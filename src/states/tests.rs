//! Lifecycle tests against fake vLLM servers on 127.0.0.1:0 (see
//! `engines::sleep::testing`). A model "running" in these tests is a real
//! owned `sleep 600` process (so ownership checks hold) whose registry port
//! points at a fake engine.

use super::idle::{IdleAction, IdleSleeper};
use super::*;
use crate::config::{load_scene_file, ScopesMode};
use crate::engines::sleep::testing::FakeVllm;
use crate::health::{NoInFlight, SupervisionTracker, Trigger};
use crate::proc;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const T0: u64 = 1_800_000_000_000;

struct TestHost {
    now: AtomicU64,
    memory: Mutex<Option<(f64, f64)>>,
    in_flight: Mutex<HashMap<String, u32>>,
    waiting: Mutex<HashMap<String, u32>>,
    last_finished: Mutex<HashMap<String, u64>>,
}

impl TestHost {
    fn new() -> Arc<TestHost> {
        Arc::new(TestHost {
            now: AtomicU64::new(T0),
            memory: Mutex::new(Some((128.0, 120.0))),
            in_flight: Mutex::new(HashMap::new()),
            waiting: Mutex::new(HashMap::new()),
            last_finished: Mutex::new(HashMap::new()),
        })
    }
    fn at(&self, offset_sec: u64) {
        self.now.store(T0 + offset_sec * 1000, Ordering::SeqCst);
    }
    fn busy(&self, id: &str, n: u32) {
        self.in_flight.lock().unwrap().insert(id.to_string(), n);
    }
    /// `n` of the in-flight requests are held for a wake.
    fn held(&self, id: &str, n: u32) {
        self.waiting.lock().unwrap().insert(id.to_string(), n);
    }
    fn finished(&self, id: &str, offset_sec: u64) {
        self.last_finished
            .lock()
            .unwrap()
            .insert(id.to_string(), T0 + offset_sec * 1000);
    }
}

impl Host for TestHost {
    fn now_ms(&self) -> u64 {
        self.now.load(Ordering::SeqCst)
    }
    fn memory(&self, _pool: &governor::Pool) -> Option<(f64, f64)> {
        *self.memory.lock().unwrap()
    }
    fn in_flight(&self, id: &str) -> u32 {
        self.in_flight.lock().unwrap().get(id).copied().unwrap_or(0)
    }
    fn waiting(&self, id: &str) -> u32 {
        self.waiting.lock().unwrap().get(id).copied().unwrap_or(0)
    }
    fn last_finished_ms(&self, id: &str) -> Option<u64> {
        self.last_finished.lock().unwrap().get(id).copied()
    }
    fn poll(&self) -> Duration {
        Duration::from_millis(20)
    }
}

/// A Lumbridge Computer root with a registry, one Scene `test` desired, and owned
/// stand-in processes. Everything is stopped and removed on drop.
struct Fixture {
    root: PathBuf,
    host: Arc<TestHost>,
}

impl Fixture {
    fn new(registry_models: &str, scene_models: &[&str], budget_gb: f64, modes: &str) -> Fixture {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("lumbridge-computer-states-{unique}"));
        fs::create_dir_all(root.join("registry")).unwrap();
        fs::create_dir_all(root.join("scenes")).unwrap();
        fs::write(
            root.join("registry/models.yaml"),
            format!("apiVersion: lumbridge/v1\nscopes: off\nmodels:\n{registry_models}"),
        )
        .unwrap();
        fs::write(
            root.join("scenes/test.scene.yaml"),
            format!(
                "apiVersion: lumbridge/v1\nmetadata:\n  name: test\n  version: 1\nmodels: [{}]\nbudget_gb: {budget_gb}\nmodes:\n{modes}",
                scene_models.join(", ")
            ),
        )
        .unwrap();
        State {
            desired_scene: Some("test".into()),
            active_scene: Some("test".into()),
            ..State::default()
        }
        .save(&root)
        .unwrap();
        Fixture {
            root,
            host: TestHost::new(),
        }
    }

    /// Record an owned stand-in process for `id` in `state`.
    fn own(&self, id: &str, state: Option<ModelState>) -> Proc {
        let stand_in: Model = serde_yaml::from_str(
            "name: stand-in\nfootprint_gb: 0\nserve: {kind: exec, command: [/usr/bin/sleep, '600']}\n",
        )
        .unwrap();
        let pid = proc::spawn(
            &self.root,
            &format!("{id}-stand-in"),
            &stand_in,
            ScopesMode::Off,
        )
        .unwrap();
        let mut saved = self.state();
        saved.seq += 1;
        let mut process = Proc::capture(pid, saved.seq, None).unwrap();
        if let Some(model_state) = state {
            process.set_state(model_state, T0);
        }
        saved.procs.insert(id.to_string(), process.clone());
        saved.save(&self.root).unwrap();
        process
    }

    fn state(&self) -> State {
        State::load_checked(&self.root).unwrap()
    }

    fn model_state(&self, id: &str) -> ModelState {
        self.lifecycle().state(id)
    }

    fn lifecycle(&self) -> ModelLifecycle {
        ModelLifecycle::new(&self.root, self.host.clone())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Ok(state) = State::load_checked(&self.root) {
            for process in state.procs.values() {
                let _ = proc::stop_owned(process);
            }
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn soon(sec: u64) -> Instant {
    Instant::now() + Duration::from_secs(sec)
}

fn asleep() -> Option<ModelState> {
    Some(ModelState::Sleeping { level: 2 })
}

// ---- state transitions -------------------------------------------------------

#[test]
fn a_sleeping_model_wakes_through_waking_to_ready_and_sleeps_again() {
    let engine = FakeVllm::start();
    engine.set(|s| s.sleeping = true);
    let fx = Fixture::new(
        &engine.registry_entry("a", 10.0, ""),
        &["a"],
        100.0,
        "  a: {mode: sleepable}\n",
    );
    let owned = fx.own("a", asleep());
    let lifecycle = fx.lifecycle();
    assert_eq!(lifecycle.state("a"), ModelState::Sleeping { level: 2 });

    fx.host.at(5);
    lifecycle.ensure_ready("a", soon(10)).unwrap();
    assert_eq!(lifecycle.state("a"), ModelState::Ready);
    let record = fx.state().procs["a"].clone();
    assert_eq!(record.pid, owned.pid, "a wake keeps the process");
    assert_eq!(record.state_since_ms, Some(T0 + 5_000));
    assert_eq!(
        engine.calls(),
        [
            "POST /wake_up?tags=weights",
            "POST /collective_rpc",
            "POST /wake_up?tags=kv_cache"
        ]
    );
    // Idempotent: a second call on a ready model touches nothing.
    lifecycle.ensure_ready("a", soon(10)).unwrap();
    assert_eq!(engine.count("POST /wake_up"), 2);

    fx.host.at(9);
    lifecycle.sleep("a", "test").unwrap();
    assert_eq!(lifecycle.state("a"), ModelState::Sleeping { level: 2 });
    assert!(engine.sleeping());
    assert_eq!(
        engine.calls().last().unwrap(),
        "POST /sleep?level=2&mode=wait"
    );
    assert_eq!(fx.state().procs["a"].state_since_ms, Some(T0 + 9_000));
    // Sleeping an asleep model is a no-op.
    lifecycle.sleep("a", "again").unwrap();
    assert_eq!(engine.count("POST /sleep"), 1);
}

#[test]
fn a_failed_reload_stops_the_model_and_the_next_request_takes_the_cold_path() {
    let engine = FakeVllm::start();
    engine.set(|s| {
        s.sleeping = true;
        s.fail_rpc = true;
    });
    let fx = Fixture::new(
        &engine.registry_entry("a", 10.0, ""),
        &["a"],
        100.0,
        "  a: {mode: sleepable}\n",
    );
    let owned = fx.own("a", asleep());
    let lifecycle = fx.lifecycle();
    let error = format!("{:#}", lifecycle.ensure_ready("a", soon(10)).unwrap_err());
    assert!(error.contains("stopped it"), "{error}");
    assert!(error.contains("/collective_rpc"), "{error}");
    // Still "sleeping" to vLLM (kv_cache never woke), yet stopped: torn
    // weights must never serve, and a retry would not repair them.
    assert!(engine.sleeping());
    assert_eq!(engine.count("POST /wake_up?tags=kv_cache"), 0);
    assert!(!fx.state().procs.contains_key("a"));
    assert_eq!(lifecycle.state("a"), ModelState::Stopped);
    let gone = (0..50).any(|_| {
        std::thread::sleep(Duration::from_millis(20));
        !owned.owned_alive()
    });
    assert!(gone, "the engine process was stopped");

    // The next request starts it cold (here refused: the fake, standing in
    // for a dead engine, still holds the port), and never tries to wake the
    // old engine again.
    engine.set(|s| s.unready = true);
    let wakes = engine.count("POST /wake_up");
    assert!(lifecycle.ensure_ready("a", soon(10)).is_err());
    assert_eq!(engine.count("POST /wake_up"), wakes);
}

#[test]
fn a_model_with_mtp_wakes_only_if_its_drafter_reloaded() {
    let mtp =
        "      args: {speculative-config: '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'}\n";
    for (reply, ok) in [
        (crate::engines::sleep::testing::RELOAD_WITH_DRAFT, true),
        (crate::engines::sleep::testing::RELOAD_NO_DRAFT, false),
    ] {
        let engine = FakeVllm::start();
        engine.set(|s| {
            s.sleeping = true;
            s.rpc_reply = reply.to_string();
        });
        let fx = Fixture::new(
            &format!("{}{mtp}", engine.registry_entry("a", 10.0, "")),
            &["a"],
            100.0,
            "  a: {mode: sleepable}\n",
        );
        fx.own("a", asleep());
        let woke = fx.lifecycle().ensure_ready("a", soon(10));
        assert_eq!(woke.is_ok(), ok, "{reply}: {woke:?}");
        if ok {
            assert_eq!(fx.model_state("a"), ModelState::Ready);
        } else {
            let error = format!("{:#}", woke.unwrap_err());
            assert!(error.contains("drafter"), "{error}");
            assert!(!fx.state().procs.contains_key("a"), "stopped, not served");
        }
    }
}

#[test]
fn sleep_refuses_resident_models_and_models_outside_the_scene() {
    let engine = FakeVllm::start();
    let fx = Fixture::new(
        &format!(
            "{}{}",
            engine.registry_entry("res", 10.0, ""),
            engine.registry_entry("b", 10.0, "")
        ),
        &["res"],
        100.0,
        "",
    );
    fx.own("res", None);
    let error = fx.lifecycle().sleep("res", "test").unwrap_err().to_string();
    assert!(error.contains("resident"), "{error}");
    assert!(fx.lifecycle().sleep("b", "test").is_err());
    assert!(engine.calls().is_empty());
    assert_eq!(fx.model_state("res"), ModelState::Ready);
}

#[test]
fn a_drain_that_times_out_leaves_the_model_ready() {
    let engine = FakeVllm::start();
    let fx = Fixture::new(
        &engine.registry_entry("a", 10.0, "").replace(
            "sleep: {level: 2, footprint_gb: 2}",
            "sleep: {level: 2, drain_timeout_sec: 1}",
        ),
        &["a"],
        100.0,
        "  a: {mode: sleepable}\n",
    );
    fx.own("a", None);
    fx.host.busy("a", 1);
    let error = fx.lifecycle().sleep("a", "test").unwrap_err().to_string();
    assert!(error.contains("in flight"), "{error}");
    assert_eq!(fx.model_state("a"), ModelState::Ready);
    assert!(engine.calls().is_empty(), "never slept with work in flight");
}

#[test]
fn a_drain_waits_for_active_requests_only_never_for_ones_held_for_a_wake() {
    let engine = FakeVllm::start();
    let fx = Fixture::new(
        &engine.registry_entry("a", 10.0, "").replace(
            "sleep: {level: 2, footprint_gb: 2}",
            "sleep: {level: 2, drain_timeout_sec: 1}",
        ),
        &["a"],
        100.0,
        "  a: {mode: sleepable}\n",
    );
    fx.own("a", None);
    // One request in flight, and it is the one held waiting: nothing active.
    fx.host.busy("a", 1);
    fx.host.held("a", 1);
    let started = Instant::now();
    fx.lifecycle().sleep("a", "test").unwrap();
    assert!(
        started.elapsed() < Duration::from_millis(900),
        "no drain wait"
    );
    assert_eq!(fx.model_state("a"), ModelState::Sleeping { level: 2 });

    // Two in flight, one held: the active one blocks the drain.
    let fx2 = Fixture::new(
        &engine.registry_entry("b", 10.0, "").replace(
            "sleep: {level: 2, footprint_gb: 2}",
            "sleep: {level: 2, drain_timeout_sec: 1}",
        ),
        &["b"],
        100.0,
        "  b: {mode: sleepable}\n",
    );
    fx2.own("b", None);
    fx2.host.busy("b", 2);
    fx2.host.held("b", 1);
    assert!(fx2.lifecycle().sleep("b", "test").is_err());
    assert_eq!(fx2.model_state("b"), ModelState::Ready);
}

#[test]
fn state_is_answered_from_memory_and_updated_by_every_transition() {
    let engine = FakeVllm::start();
    let fx = Fixture::new(
        &engine.registry_entry("a", 10.0, ""),
        &["a"],
        100.0,
        "  a: {mode: sleepable}\n",
    );
    fx.own("a", None);
    // The cache ages by a hand-moved clock, not the wall clock, so a slow machine
    // cannot let it expire between two asserts.
    let now = Arc::new(Mutex::new(Instant::now()));
    let clock = {
        let now = now.clone();
        Arc::new(move || *now.lock().unwrap()) as super::machine::Clock
    };
    let lifecycle = ModelLifecycle::with_clock(&fx.root, fx.host.clone(), clock);
    assert_eq!(lifecycle.state("a"), ModelState::Ready);
    // A transition by this lifecycle shows at once.
    lifecycle.sleep("a", "test").unwrap();
    assert_eq!(lifecycle.state("a"), ModelState::Sleeping { level: 2 });
    // A change behind its back (another process) is not read per call...
    let mut state = fx.state();
    state
        .procs
        .get_mut("a")
        .unwrap()
        .set_state(ModelState::Ready, T0);
    state.save(&fx.root).unwrap();
    assert_eq!(lifecycle.state("a"), ModelState::Sleeping { level: 2 });
    *now.lock().unwrap() += Duration::from_millis(900);
    assert_eq!(lifecycle.state("a"), ModelState::Sleeping { level: 2 });
    // ...but within the cache lifetime.
    *now.lock().unwrap() += Duration::from_millis(200);
    assert_eq!(lifecycle.state("a"), ModelState::Ready);
}

#[test]
fn an_engine_that_cannot_sleep_is_stopped_and_cold_started_on_demand() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let fx = Fixture::new(
        &format!(
            "  web:\n    name: web\n    footprint_gb: 4\n    health: http://localhost:{port}/\n    serve:\n      kind: exec\n      port: {port}\n      command: [/usr/bin/python3, -m, http.server, '{port}', --bind, 127.0.0.1]\n"
        ),
        &["web"],
        100.0,
        "  web: {mode: on_demand}\n",
    );
    let lifecycle = fx.lifecycle();
    assert_eq!(lifecycle.state("web"), ModelState::Stopped);
    lifecycle.ensure_ready("web", soon(20)).unwrap();
    assert_eq!(lifecycle.state("web"), ModelState::Ready);
    assert!(governor::port_open(port));

    lifecycle.sleep("web", "idle").unwrap();
    assert_eq!(lifecycle.state("web"), ModelState::Stopped);
    assert!(!fx.state().procs.contains_key("web"));
    assert!(!governor::port_open(port));
}

#[test]
fn ensure_ready_never_starts_a_stopped_resident_model() {
    let engine = FakeVllm::start();
    engine.set(|s| s.unready = true);
    let fx = Fixture::new(&engine.registry_entry("res", 10.0, ""), &["res"], 100.0, "");
    let error = format!(
        "{:#}",
        fx.lifecycle().ensure_ready("res", soon(5)).unwrap_err()
    );
    assert!(error.contains("resident"), "{error}");
}

#[test]
fn a_state_file_from_before_m3_reads_a_live_process_as_ready() {
    let engine = FakeVllm::start();
    let fx = Fixture::new(&engine.registry_entry("res", 10.0, ""), &["res"], 100.0, "");
    let owned = fx.own("res", None);
    // Exactly what a pre-M3 binary writes: no state keys at all.
    let old = format!(
        "seq: 1\nprocs:\n  res:\n    pid: {}\n    seq: 1\n    port: null\n    boot_id: {}\n    start_time_ticks: {}\ndesired_scene: test\nactive_scene: test\n",
        owned.pid,
        owned.boot_id.clone().unwrap(),
        owned.start_time_ticks.unwrap()
    );
    fs::write(fx.root.join(".compute/state.yaml"), old).unwrap();
    assert_eq!(fx.model_state("res"), ModelState::Ready);
    assert_eq!(fx.state().procs["res"].model_state(), ModelState::Ready);
}

// ---- single flight -------------------------------------------------------------

#[test]
fn concurrent_ensure_ready_callers_share_one_wake() {
    let engine = FakeVllm::start();
    engine.set(|s| {
        s.sleeping = true;
        s.wake_delay = Duration::from_millis(300);
    });
    let fx = Fixture::new(
        &engine.registry_entry("a", 10.0, ""),
        &["a"],
        100.0,
        "  a: {mode: sleepable}\n",
    );
    fx.own("a", asleep());
    let lifecycle = fx.lifecycle();
    let callers: Vec<_> = (0..8)
        .map(|_| {
            let lifecycle = lifecycle.clone();
            std::thread::spawn(move || lifecycle.ensure_ready("a", soon(15)))
        })
        .collect();
    for caller in callers {
        caller.join().unwrap().unwrap();
    }
    assert_eq!(
        engine.count("POST /wake_up?tags=weights"),
        1,
        "{:?}",
        engine.calls()
    );
    assert_eq!(engine.count("POST /collective_rpc"), 1);
    assert_eq!(lifecycle.state("a"), ModelState::Ready);
}

#[test]
fn an_impatient_caller_gets_an_error_while_the_wake_carries_on() {
    let engine = FakeVllm::start();
    engine.set(|s| {
        s.sleeping = true;
        s.wake_delay = Duration::from_millis(600);
    });
    let fx = Fixture::new(
        &engine.registry_entry("a", 10.0, ""),
        &["a"],
        100.0,
        "  a: {mode: sleepable}\n",
    );
    fx.own("a", asleep());
    let lifecycle = fx.lifecycle();
    let error = lifecycle
        .ensure_ready("a", Instant::now() + Duration::from_millis(100))
        .unwrap_err()
        .to_string();
    assert!(error.contains("deadline"), "{error}");
    // A patient caller joins the same wake.
    lifecycle.ensure_ready("a", soon(10)).unwrap();
    assert_eq!(engine.count("POST /wake_up?tags=weights"), 1);
}

// ---- admission and eviction -----------------------------------------------------

/// Target `t` (on_demand, asleep at 2 GB) plus: `res` resident, `busy` with a
/// request in flight, `x` shed first (priority 10), `y` and `z` at priority 0
/// with `y` idle longer. Each on its own fake engine.
struct Crowd {
    fx: Fixture,
    engines: HashMap<&'static str, FakeVllm>,
}

fn crowd(budget_gb: f64) -> Crowd {
    let mut engines = HashMap::new();
    let mut registry = String::new();
    for (id, footprint, extra) in [
        ("t", 20.0, ""),
        ("res", 30.0, ""),
        ("busy", 10.0, ""),
        ("x", 10.0, "    shed_priority: 10\n"),
        ("y", 10.0, ""),
        ("z", 10.0, ""),
    ] {
        let engine = FakeVllm::start();
        registry.push_str(&engine.registry_entry(id, footprint, extra));
        engines.insert(id, engine);
    }
    engines["t"].set(|s| s.sleeping = true);
    let fx = Fixture::new(
        &registry,
        &["t", "res", "busy", "x", "y", "z"],
        budget_gb,
        "  t: {mode: on_demand}\n  busy: {mode: sleepable}\n  x: {mode: sleepable}\n  y: {mode: sleepable}\n  z: {mode: sleepable}\n",
    );
    fx.own("t", asleep());
    fx.own("res", None);
    for id in ["busy", "x", "y", "z"] {
        fx.own(id, Some(ModelState::Ready));
    }
    fx.host.busy("busy", 1);
    fx.host.finished("y", 10);
    fx.host.finished("z", 20);
    Crowd { fx, engines }
}

#[test]
fn eviction_order_sleeps_then_stops_by_shed_priority_then_lru_never_resident_or_busy() {
    let crowd = crowd(1000.0);
    let plan = crowd.fx.lifecycle().plan_evictions("t").unwrap();
    let order: Vec<(String, bool)> = plan.iter().map(|e| (e.id.clone(), e.stop)).collect();
    let expect = |id: &str, stop| (id.to_string(), stop);
    assert_eq!(
        order,
        [
            expect("x", false),
            expect("y", false),
            expect("z", false),
            expect("x", true),
            expect("y", true),
            expect("z", true),
        ]
    );
    // Sleeping frees full minus sleeping footprint (10 - 2); stopping the rest.
    assert_eq!(plan[0].freed_gb, 8.0);
    assert_eq!(plan[3].freed_gb, 2.0);
}

#[test]
fn waking_evicts_only_as_much_as_it_needs() {
    // Committed 2 + 30 + 4 × 10 = 72. Waking t needs 18 more: 72 + 18 + 8 margin
    // = 98 > 92, and sleeping x alone frees 8.
    let crowd = crowd(92.0);
    let lifecycle = crowd.fx.lifecycle();
    lifecycle.ensure_ready("t", soon(10)).unwrap();
    assert_eq!(lifecycle.state("t"), ModelState::Ready);
    assert_eq!(lifecycle.state("x"), ModelState::Sleeping { level: 2 });
    for id in ["res", "busy", "y", "z"] {
        assert_eq!(lifecycle.state(id), ModelState::Ready, "{id}");
        assert!(crowd.engines[id].calls().is_empty(), "{id} untouched");
    }
}

#[test]
fn when_idle_models_cannot_free_enough_nothing_is_evicted() {
    // Even stopping x, y and z frees 30: 72 - 30 + 18 + 8 = 68 > 60.
    let crowd = crowd(60.0);
    let lifecycle = crowd.fx.lifecycle();
    let error = format!("{:#}", lifecycle.ensure_ready("t", soon(10)).unwrap_err());
    assert!(error.contains("no room for 't'"), "{error}");
    for (id, engine) in &crowd.engines {
        assert!(
            engine.calls().is_empty(),
            "{id} was touched: {:?}",
            engine.calls()
        );
    }
    assert_eq!(lifecycle.state("t"), ModelState::Sleeping { level: 2 });
}

#[test]
fn observed_memory_also_limits_a_wake() {
    let crowd = crowd(1000.0);
    // 18 GB more needed + 8 margin, only 20 available: sleep x (frees 8).
    *crowd.fx.host.memory.lock().unwrap() = Some((128.0, 20.0));
    let lifecycle = crowd.fx.lifecycle();
    // The fake host's memory does not move when x sleeps, so the post-eviction
    // re-check keeps refusing: the wake fails, but only after evicting x.
    let error = format!("{:#}", lifecycle.ensure_ready("t", soon(30)).unwrap_err());
    assert!(error.contains("still does not fit"), "{error}");
    assert_eq!(lifecycle.state("x"), ModelState::Sleeping { level: 2 });
    assert_eq!(lifecycle.state("t"), ModelState::Sleeping { level: 2 });
}

#[test]
fn a_sleeping_model_counts_at_its_sleeping_footprint() {
    let engine = FakeVllm::start();
    let fx = Fixture::new(
        &format!(
            "{}{}",
            engine.registry_entry("a", 10.0, ""),
            engine
                .registry_entry("b", 10.0, "")
                .replace("sleep: {level: 2, footprint_gb: 2}", "sleep: {level: 2}")
        ),
        &["a", "b"],
        100.0,
        "  a: {mode: sleepable}\n  b: {mode: sleepable}\n",
    );
    fx.own("a", asleep());
    fx.own("b", asleep());
    let registry = Registry::load(&fx.root).unwrap();
    let state = fx.state();
    assert_eq!(
        counted_gb(&fx.root, &registry, &state, "a"),
        2.0,
        "registry"
    );
    // Default max(6, 0.3 x 10): a sleeping vLLM keeps its host side and CUDA
    // context (6.4 GB measured for a 4B model).
    assert_eq!(counted_gb(&fx.root, &registry, &state, "b"), 6.0, "default");
    let pool = governor::Pool::unified();
    assert_eq!(committed_gb(&fx.root, &registry, &state, &pool), 8.0);

    // A sample taken during the current sleep beats the registry figure.
    let samples = crate::footprint::Samples::open(&fx.root).unwrap();
    let measurement = crate::footprint::Measurement {
        source: "scope",
        cgroup: None,
        host_gb: 1.0,
        file_gb: 0.0,
        gpu_gb: Some(0.5),
        swap_gb: 0.0,
        peak_host_gb: None,
        pids: 1,
    };
    samples
        .record((T0 - 60_000) as i64, "a", &measurement)
        .unwrap(); // before the sleep
    assert_eq!(counted_gb(&fx.root, &registry, &state, "a"), 2.0);
    samples
        .record((T0 + 30_000) as i64, "a", &measurement)
        .unwrap();
    assert_eq!(counted_gb(&fx.root, &registry, &state, "a"), 1.5);
}

// ---- supervision -------------------------------------------------------------------

const SUPERVISED: &str =
    "    supervision: {restart: always, check_interval_sec: 1, failure_threshold: 1, backoff_sec: 1}\n";

#[test]
fn supervision_never_probes_or_restarts_a_sleeping_or_stopped_non_resident_model() {
    let _memory = crate::mem::test_memory(128.0, 100.0);
    let sleeper = FakeVllm::start();
    let waker = FakeVllm::start();
    let demand = FakeVllm::start();
    // Asleep, waking and stopped: none of them answers its health check.
    for engine in [&sleeper, &waker, &demand] {
        engine.set(|s| {
            s.unready = true;
            s.sleeping = true;
        });
    }
    let fx = Fixture::new(
        &format!(
            "{}{}{}",
            sleeper.registry_entry("a", 10.0, SUPERVISED),
            waker.registry_entry("b", 10.0, SUPERVISED),
            demand.registry_entry("c", 10.0, SUPERVISED),
        ),
        &["a", "b", "c"],
        100.0,
        "  a: {mode: sleepable}\n  b: {mode: sleepable}\n  c: {mode: on_demand}\n",
    );
    let a = fx.own("a", asleep());
    let b = fx.own("b", Some(ModelState::Waking));
    let mut tracker = SupervisionTracker::new(Arc::new(NoInFlight));
    for now in [0, 5, 10, 20, 40, 80] {
        tracker.tick(&fx.root, now).unwrap();
    }
    let state = fx.state();
    assert_eq!(state.procs["a"].pid, a.pid);
    assert_eq!(state.procs["b"].pid, b.pid);
    assert!(a.owned_alive() && b.owned_alive(), "nothing was stopped");
    assert!(!state.procs.contains_key("c"), "nothing was started");
    assert_eq!(state.active_scene.as_deref(), Some("test"));
    assert!(state.last_error.is_none(), "{:?}", state.last_error);
    for engine in [&sleeper, &waker, &demand] {
        assert!(engine.calls().is_empty());
    }

    // The restart path re-checks under the lock, too.
    let skipped =
        crate::lifecycle::restart_supervised_model(&fx.root, "a", Trigger::Exited, &NoInFlight)
            .unwrap();
    assert!(
        matches!(&skipped, crate::lifecycle::Restart::Skipped(why) if why.contains("sleeping")),
        "{skipped:?}"
    );
    assert!(a.owned_alive());
}

/// Spike pitfall 7: vLLM's `/health` (and `/v1/models`, `/metrics`) answer 200
/// while the engine sleeps, and a chat probe would queue behind the paused
/// scheduler and time out. Health can't see sleep, so supervision must skip a
/// sleeping model on its recorded state alone, without probing it at all.
#[test]
fn supervision_skips_a_sleeping_model_even_though_its_health_answers_200() {
    let _memory = crate::mem::test_memory(128.0, 100.0);
    let engine = FakeVllm::start();
    engine.set(|s| s.sleeping = true);
    let port = engine.port;
    let liveness = format!(
        "{SUPERVISED}    liveness:\n      kind: vllm-metrics\n      stall_sec: 60\n      chat_probe: {{interval_sec: 30, timeout_sec: 5, model: fake-vllm}}\n"
    );
    let fx = Fixture::new(
        &engine.registry_entry("a", 10.0, &liveness),
        &["a"],
        100.0,
        "  a: {mode: sleepable}\n",
    );
    let model = Registry::load(&fx.root).unwrap().models["a"].clone();
    // The premise: asleep, yet /health says 200 and the exact health check
    // (/v1/models containing the marker, as for the brain) passes.
    assert!(crate::engines::sleep::is_sleeping(&model).unwrap());
    let (status, _) = crate::engines::http::request_within(
        "GET",
        &format!("http://127.0.0.1:{port}/health"),
        None,
        Duration::from_secs(1),
        Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(status, 200);
    assert_eq!(
        governor::readiness(&model, Duration::from_secs(2)),
        governor::Readiness::Ready
    );
    let probes_before = engine.state.lock().unwrap().probes;

    let a = fx.own("a", asleep());
    let lifecycle: Arc<dyn Lifecycle> = Arc::new(fx.lifecycle());
    let mut plain = SupervisionTracker::new(Arc::new(NoInFlight));
    let mut agent = SupervisionTracker::new(Arc::new(NoInFlight)).with_lifecycle(lifecycle);
    for now in [0, 1, 2, 5, 10, 31, 62, 125] {
        plain.tick(&fx.root, now).unwrap();
        agent.tick(&fx.root, now).unwrap();
    }
    let seen = engine.state.lock().unwrap();
    assert_eq!(seen.probes, probes_before, "never probed while asleep");
    assert_eq!(
        seen.chats, 0,
        "no chat probe queued behind the paused scheduler"
    );
    drop(seen);
    assert!(
        engine.calls().is_empty(),
        "never woken or slept by supervision"
    );
    assert_eq!(fx.state().procs["a"].pid, a.pid);
    assert!(a.owned_alive(), "never restarted");

    // Control: the same model recorded ready is probed, so the silence above
    // is supervision skipping it, not a probe that can't reach the fake.
    let mut state = fx.state();
    state
        .procs
        .get_mut("a")
        .unwrap()
        .set_state(ModelState::Ready, T0);
    state.save(&fx.root).unwrap();
    engine.set(|s| s.sleeping = false);
    plain.tick(&fx.root, 200).unwrap();
    assert!(engine.state.lock().unwrap().probes > probes_before);
}

#[test]
fn liveness_resumes_once_a_woken_model_is_ready() {
    let _memory = crate::mem::test_memory(128.0, 100.0);
    let engine = FakeVllm::start();
    engine.set(|s| s.sleeping = true);
    let fx = Fixture::new(
        &engine.registry_entry("a", 10.0, SUPERVISED),
        &["a"],
        100.0,
        "  a: {mode: sleepable}\n",
    );
    let a = fx.own("a", asleep());
    let mut tracker = SupervisionTracker::new(Arc::new(NoInFlight));
    tracker.tick(&fx.root, 0).unwrap();
    fx.lifecycle().ensure_ready("a", soon(10)).unwrap();
    tracker.tick(&fx.root, 10).unwrap();
    assert!(a.owned_alive(), "a ready, healthy model is left alone");

    // Now awake and failing its health check: supervision acts again. (Its
    // restart then fails to launch `vllm`, which is beside the point here.)
    engine.set(|s| s.unready = true);
    tracker.tick(&fx.root, 20).unwrap();
    assert!(!a.owned_alive(), "supervision stopped the failed process");
    assert!(!fx.state().procs.contains_key("a"));
}

// ---- idle sleeper ------------------------------------------------------------------

#[test]
fn the_idle_sleeper_sleeps_after_idle_sec_and_stops_on_demand_after_stop_after_sec() {
    let a = FakeVllm::start();
    let b = FakeVllm::start();
    let res = FakeVllm::start();
    let never = FakeVllm::start();
    let fx = Fixture::new(
        &format!(
            "{}{}{}{}",
            a.registry_entry("a", 10.0, ""),
            b.registry_entry("b", 10.0, ""),
            res.registry_entry("res", 10.0, ""),
            never.registry_entry("never", 10.0, ""),
        ),
        &["a", "b", "res", "never"],
        100.0,
        "  a: {mode: sleepable, idle_sec: 60}\n  b: {mode: on_demand, idle_sec: 60, stop_after_sec: 300}\n  never: {mode: sleepable, idle_sec: 0}\n",
    );
    fx.own("a", Some(ModelState::Ready)); // ready since T0
    fx.own("b", asleep()); // asleep since T0
    fx.own("res", None);
    fx.own("never", Some(ModelState::Ready));
    let host = fx.host.clone();
    let mut sleeper = IdleSleeper::default();
    let mut due = |offset: u64| {
        host.at(offset);
        sleeper.due(&fx.root, host.as_ref()).unwrap()
    };

    assert_eq!(due(59), []);
    assert_eq!(due(60), [IdleAction::Sleep("a".into())]);

    // Busy resets the clock; a finished request restarts it from then. A
    // request held for a wake counts as busy too: never sleep a model then.
    host.busy("a", 1);
    host.held("a", 1);
    assert_eq!(due(61), []);
    host.held("a", 0);
    host.busy("a", 0);
    host.finished("a", 100);
    assert_eq!(due(159), []);
    // Direct clients count as busy too (the engine reports them).
    a.set(|s| s.running = 1);
    assert_eq!(due(170), []);
    a.set(|s| s.running = 0);
    assert_eq!(due(229), []);
    assert_eq!(due(230), [IdleAction::Sleep("a".into())]);

    // on_demand asleep since T0: stopped at stop_after_sec.
    assert_eq!(due(299), [IdleAction::Sleep("a".into())]);
    assert_eq!(
        due(300),
        [IdleAction::Sleep("a".into()), IdleAction::Stop("b".into())]
    );
}

#[test]
fn a_model_started_by_the_scene_gets_a_full_idle_period_before_sleeping() {
    let a = FakeVllm::start();
    let fx = Fixture::new(
        &a.registry_entry("a", 10.0, ""),
        &["a"],
        100.0,
        "  a: {mode: sleepable, idle_sec: 60}\n",
    );
    fx.own("a", None); // no state, no timestamp: started by `scene activate`
    let mut sleeper = IdleSleeper::default();
    fx.host.at(1_000);
    assert_eq!(sleeper.due(&fx.root, fx.host.as_ref()).unwrap(), []);
    fx.host.at(1_059);
    assert_eq!(sleeper.due(&fx.root, fx.host.as_ref()).unwrap(), []);
    fx.host.at(1_060);
    assert_eq!(
        sleeper.due(&fx.root, fx.host.as_ref()).unwrap(),
        [IdleAction::Sleep("a".into())]
    );
}

#[test]
fn the_idle_sleeper_tick_acts_and_a_resident_found_asleep_is_woken() {
    let a = FakeVllm::start();
    let res = FakeVllm::start();
    res.set(|s| s.sleeping = true);
    let fx = Fixture::new(
        &format!(
            "{}{}",
            a.registry_entry("a", 10.0, ""),
            res.registry_entry("res", 10.0, "")
        ),
        &["a", "res"],
        100.0,
        "  a: {mode: sleepable, idle_sec: 30}\n",
    );
    fx.own("a", Some(ModelState::Ready));
    fx.own("res", asleep()); // its Scene made it resident while it slept
    fx.host.at(31);
    let lifecycle = fx.lifecycle();
    let results = IdleSleeper::default().tick(&lifecycle);
    assert_eq!(results.len(), 2, "{results:?}");
    assert!(results.iter().all(|(_, r)| r.is_ok()), "{results:?}");
    assert_eq!(lifecycle.state("a"), ModelState::Sleeping { level: 2 });
    assert_eq!(lifecycle.state("res"), ModelState::Ready);
}

// ---- scene modes -------------------------------------------------------------------

fn scene(yaml_modes: &str, models: &str) -> Scene {
    serde_yaml::from_str(&format!(
        "apiVersion: lumbridge/v1\nmetadata: {{name: s}}\nmodels: [{models}]\nbudget_gb: 100\nmodes:\n{yaml_modes}"
    ))
    .unwrap()
}

fn mode_registry() -> Registry {
    Registry::parse(
        "apiVersion: lumbridge/v1\nmodels:\n  llm:\n    name: l\n    footprint_gb: 40\n    serve: {kind: vllm, port: 1, sleep: {level: 2}}\n  plain:\n    name: p\n    footprint_gb: 40\n    serve: {kind: vllm, port: 2}\n  asr:\n    name: a\n    footprint_gb: 20\n    serve: {kind: exec, port: 3, command: [/bin/true]}\n",
    )
    .unwrap()
}

#[test]
fn scene_modes_default_to_resident_and_old_scenes_validate_as_before() {
    let registry = mode_registry();
    let old: Scene = serde_yaml::from_str(
        "apiVersion: lumbridge/v1\nmetadata: {name: s}\nmodels: [llm, asr]\nbudget_gb: 100\n",
    )
    .unwrap();
    assert!(old.modes.is_empty());
    assert_eq!(old.mode_of("llm").mode, Mode::Resident);
    crate::lifecycle::validate_scene(&registry, &old).unwrap();
    let pool = governor::Pool::unified();
    assert_eq!(scene_need_gb(&registry, &old, &pool), 60.0);
}

#[test]
fn only_engines_that_can_sleep_may_be_sleepable_but_any_may_be_on_demand() {
    let registry = mode_registry();
    validate_modes(&registry, &scene("  llm: {mode: sleepable}\n", "llm, asr")).unwrap();
    validate_modes(&registry, &scene("  asr: {mode: on_demand}\n", "llm, asr")).unwrap();
    validate_modes(&registry, &scene("  plain: {mode: on_demand}\n", "plain")).unwrap();
    for (modes, models, needle) in [
        ("  asr: {mode: sleepable}\n", "asr", "cannot be sleepable"),
        (
            "  plain: {mode: sleepable}\n",
            "plain",
            "cannot be sleepable",
        ),
        ("  llm: {mode: on_demand}\n", "asr", "not one of its models"),
        ("  llm: {idle_sec: 60}\n", "llm", "resident"),
        (
            "  llm: {mode: sleepable, stop_after_sec: 9}\n",
            "llm",
            "only applies to on_demand",
        ),
    ] {
        let error = validate_modes(&registry, &scene(modes, models))
            .unwrap_err()
            .to_string();
        assert!(error.contains(needle), "{modes}: {error}");
    }
    // Unknown keys and modes are parse errors, not silently resident.
    assert!(serde_yaml::from_str::<Scene>(
        "apiVersion: lumbridge/v1\nmetadata: {name: s}\nmodels: [llm]\nmodes:\n  llm: {mode: sometimes}\n"
    )
    .is_err());
    assert!(serde_yaml::from_str::<Scene>(
        "apiVersion: lumbridge/v1\nmetadata: {name: s}\nmodels: [llm]\nmodes:\n  llm: {idle: 5}\n"
    )
    .is_err());
}

#[test]
fn idle_sec_has_a_floor_and_zero_means_never_for_idleness() {
    let modes = scene(
        "  llm: {mode: sleepable, idle_sec: 5}\n  asr: {mode: on_demand}\n  plain: {mode: sleepable, idle_sec: 0}\n",
        "llm, asr, plain",
    );
    assert_eq!(modes.mode_of("llm").idle_sec(), Some(30));
    assert_eq!(modes.mode_of("asr").idle_sec(), Some(600));
    assert_eq!(modes.mode_of("asr").stop_after_sec(), 1800);
    assert_eq!(modes.mode_of("plain").idle_sec(), None);
}

#[test]
fn the_budget_counts_sleepers_at_their_floor_plus_room_for_one_awake() {
    let registry = mode_registry();
    let pool = governor::Pool::unified();
    // llm asleep 3 + asr stopped 0 + the widest awake (llm: 37, asr: 20).
    let both = scene(
        "  llm: {mode: sleepable}\n  asr: {mode: on_demand}\n",
        "llm, asr",
    );
    assert_eq!(scene_need_gb(&registry, &both, &pool), 40.0);
    // Only asr non-resident: 40 resident + 0 + 20.
    let one = scene("  asr: {mode: on_demand}\n", "llm, asr");
    assert_eq!(scene_need_gb(&registry, &one, &pool), 60.0);
}

#[test]
fn a_registry_sleep_block_is_vllm_only_and_checked() {
    let parse = |serve: &str| {
        Registry::parse(&format!(
            "apiVersion: lumbridge/v1\nmodels:\n  m:\n    name: m\n    footprint_gb: 1\n    serve: {serve}\n"
        ))
    };
    parse("{kind: vllm, sleep: {}}").unwrap();
    let error = format!(
        "{:#}",
        parse("{kind: exec, command: [x], sleep: {}}").unwrap_err()
    );
    assert!(error.contains("only supported for vLLM"), "{error}");
    assert!(parse("{kind: vllm, sleep: {level: 3}}").is_err());
    // Level 1 is refused at load time, pointing at the measurement.
    let error = format!(
        "{:#}",
        parse("{kind: vllm, sleep: {level: 1}}").unwrap_err()
    );
    assert!(error.contains("level must be 2"), "{error}");
    assert!(error.contains("docs/sleep.md"), "{error}");
    // The builder is required: a hand-written argv can't get the extension
    // or the loopback bind.
    let error = format!(
        "{:#}",
        parse("{kind: vllm, command: [vllm, serve, /w, --enable-sleep-mode], sleep: {}}")
            .unwrap_err()
    );
    assert!(error.contains("vllm builder"), "{error}");
    let error = format!(
        "{:#}",
        parse("{kind: vllm, args: {worker-extension-cls: x.Y}, sleep: {}}").unwrap_err()
    );
    assert!(error.contains("worker-extension-cls"), "{error}");
    assert!(parse("{kind: vllm, sleep: {wake: [{path: 'http://x/'}]}}").is_err());
    assert!(parse("{kind: vllm, sleep: {levle: 2}}").is_err());
}

/// The example Scene: the fixture box's registry with the `two-brains-lite` delta.
#[test]
fn the_two_brains_lite_example_validates_with_brain_alt_on_demand() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    let live: serde_yaml::Value = serde_yaml::from_str(
        &fs::read_to_string(base.join("tests/fixtures/box/registry/models.yaml")).unwrap(),
    )
    .unwrap();
    let delta: serde_yaml::Value = serde_yaml::from_str(
        &fs::read_to_string(base.join("examples/two-brains-lite.registry.yaml")).unwrap(),
    )
    .unwrap();
    let mut merged = live.clone();
    let models = merged
        .get_mut("models")
        .and_then(|m| m.as_mapping_mut())
        .unwrap();
    for (id, model) in delta["models"].as_mapping().unwrap() {
        models.insert(id.clone(), model.clone());
    }
    let registry = Registry::parse(&serde_yaml::to_string(&merged).unwrap()).unwrap();
    let scene = load_scene_file(&base.join("examples/two-brains-lite.scene.yaml")).unwrap();

    crate::lifecycle::validate_scene(&registry, &scene).unwrap();
    assert_eq!(scene.mode_of("brain-alt").mode, Mode::OnDemand);
    assert_eq!(scene.mode_of("brain").mode, Mode::Sleepable);
    assert_eq!(scene.mode_of("brain").idle_sec(), None);
    assert_eq!(scene.mode_of("asr").mode, Mode::Resident);
    let alt = &registry.models["brain-alt"];
    assert!(
        crate::engines::sleep::supports_sleep(alt),
        "inherits serve.sleep"
    );
    let argv = proc::build_argv(alt, Some(121.6)).unwrap();
    assert!(argv.contains(&"--enable-sleep-mode".to_string()));
    assert!(argv.contains(&crate::config::SLEEP_EXTENSION_CLS.to_string()));
    let host = argv.iter().position(|a| a == "--host").unwrap();
    assert_eq!(argv[host + 1], "127.0.0.1");
    assert!(crate::engines::sleep::has_speculative_config(alt));
    let pool = governor::Pool::unified();
    // Residents 53 + both brains asleep at max(6, 0.3 x 40) = 12 + one awake (40 - 12).
    assert_eq!(scene_need_gb(&registry, &scene, &pool), 105.0);
    // Everything resident would not fit: that is why the modes exist.
    let total: f64 = scene
        .models
        .iter()
        .map(|id| registry.models[id].footprint_gb)
        .sum();
    assert!(total + pool.safety_margin_gb > scene.budget_gb.unwrap());
}

/// The sleep example: the brain with sleep on, loopback only, still routed
/// through its proxy.
#[test]
fn the_brain_sleep_example_binds_loopback_and_keeps_its_proxy() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    let registry = Registry::load_file(&base.join("examples/brain-sleep.registry.yaml")).unwrap();
    let brain = &registry.models["brain"];
    assert!(crate::engines::sleep::supports_sleep(brain));
    assert!(crate::engines::sleep::has_speculative_config(brain));
    assert_eq!(brain.gateway.upstream.as_deref(), Some("127.0.0.1:8041"));
    let sleep = brain.serve.sleep.as_ref().unwrap();
    assert_eq!(sleep.wake_timeout(), Duration::from_secs(60));
    assert_eq!(sleep.footprint_for(brain.footprint_gb), 12.0);
    let argv = proc::build_argv(brain, Some(121.6)).unwrap();
    let flag = |name: &str| {
        argv.iter()
            .position(|a| a == name)
            .map(|i| argv[i + 1].clone())
    };
    assert_eq!(flag("--host").as_deref(), Some("127.0.0.1"));
    assert_eq!(flag("--port").as_deref(), Some("8001"));
    assert_eq!(
        flag("--worker-extension-cls").as_deref(),
        Some(crate::config::SLEEP_EXTENSION_CLS)
    );
    assert!(argv.contains(&"--enable-sleep-mode".to_string()));
}
