//! [`ModelLifecycle`]: the `Lifecycle` implementation.
//!
//! ## Locking
//!
//! Every read-modify-write of `state.yaml` happens under the Scene transition
//! lock (`lifecycle::TransitionLock`, a non-blocking `flock`), the same lock
//! Scene activation, supervised restarts and the memory watchdog take. Those
//! three only ever *try* the lock and back off, so waiting for it here can't
//! deadlock with them. The lifecycle waits for the lock by polling up to a
//! deadline and never holds its own in-process mutex while doing so.
//!
//! The lock is held only for decisions and short engine calls (admission,
//! eviction, kicking off a start or wake), never across a cold start or the
//! readiness wait after a wake. Meanwhile the model is persisted as `Starting`
//! or `Waking`, which supervision, eviction and the idle sleeper all leave
//! alone; whoever finishes takes the lock again to record `Ready`.
//!
//! ## Single flight
//!
//! Concurrent `ensure_ready` callers for one model in this process share one
//! worker thread; each waits on it up to its own deadline, and the worker
//! carries on past an impatient caller (a cold start outlives a 60 s gateway
//! deadline). Across processes (the CLI and the agent) the lock serialises
//! them and the second one finds the model ready, or mid-start, and waits.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use super::{counted_gb, sleeping_gb, Host, Lifecycle, ModelState};
use crate::config::{find_scene, Mode, Model, Registry, Scene, DEFAULT_DRAIN_TIMEOUT_SEC};
use crate::engines::sleep as engine;
use crate::governor::{self, Readiness};
use crate::lifecycle::TransitionLock;
use crate::proc::{self, Proc, State};

/// How long a finishing step waits for the lock to record its outcome.
const FINISH_LOCK_PATIENCE: Duration = Duration::from_secs(120);
/// How long `sleep`/`stop` wait for the lock before giving up.
const SLEEP_LOCK_PATIENCE: Duration = Duration::from_secs(30);
/// The shortest lock wait a worker gets, however close the caller's deadline.
const MIN_LOCK_PATIENCE: Duration = Duration::from_secs(5);
/// Eviction only picks idle models, so a drain longer than this means a
/// request raced in: leave that model and look elsewhere.
const EVICTION_DRAIN: Duration = Duration::from_secs(5);
/// After evicting, how long freed memory may take to show in MemAvailable.
const RELEASE_WAIT: Duration = Duration::from_secs(15);
/// A `Waking` record older than its wake timeout plus this is an interrupted
/// wake (the waker died), not one in progress.
const STALE_WAKE_GRACE: Duration = Duration::from_secs(30);
/// Cold-start no-progress window for models without a supervision block.
const DEFAULT_STARTUP_WINDOW: Duration = Duration::from_secs(900);

type Outcome = Result<(), String>;

#[derive(Default)]
struct Flight {
    outcome: Mutex<Option<Outcome>>,
    done: Condvar,
}

struct Inner {
    root: PathBuf,
    host: Arc<dyn Host>,
    flights: Mutex<HashMap<String, Arc<Flight>>>,
    /// `state()` answers from here: the gateway calls it on its async request
    /// path, so it must not read files or probe ports per request.
    cache: RwLock<StateCache>,
    /// The time the state cache ages by: `Instant::now` outside tests.
    clock: Clock,
}

/// A source of "now" for the state cache, so tests can age it without sleeping.
pub(crate) type Clock = Arc<dyn Fn() -> Instant + Send + Sync>;

/// Every owned model's state as of the last save by this lifecycle, or the
/// last read of the state file. Re-read when older than [`CACHE_TTL`], which
/// picks up transitions made by another process (the CLI) and processes that died.
#[derive(Default)]
struct StateCache {
    loaded: Option<Instant>,
    states: HashMap<String, ModelState>,
}

const CACHE_TTL: Duration = Duration::from_secs(1);

/// The lifecycle for one Lumbridge Computer root. A cheap clone: clones share flights.
#[derive(Clone)]
pub struct ModelLifecycle {
    inner: Arc<Inner>,
}

/// One planned eviction step.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Eviction {
    pub id: String,
    pub stop: bool,
    pub freed_gb: f64,
}

impl ModelLifecycle {
    pub fn new(root: &Path, host: Arc<dyn Host>) -> Self {
        Self::with_clock(root, host, Arc::new(Instant::now))
    }

    /// As [`ModelLifecycle::new`], with the state cache aged by `clock`.
    pub(crate) fn with_clock(root: &Path, host: Arc<dyn Host>, clock: Clock) -> Self {
        Self {
            inner: Arc::new(Inner {
                root: root.to_path_buf(),
                host,
                flights: Mutex::new(HashMap::new()),
                cache: RwLock::new(StateCache::default()),
                clock,
            }),
        }
    }

    pub fn root(&self) -> &Path {
        &self.inner.root
    }

    pub fn host(&self) -> &dyn Host {
        self.inner.host.as_ref()
    }

    /// Drain then stop `id` (any engine). Refuses resident models: the Scene and
    /// supervision own those.
    pub fn stop(&self, id: &str, reason: &str) -> Result<()> {
        self.inner.put_down(id, reason, true)
    }
    /// The eviction order `ensure_ready(target)` would use right now.
    #[cfg(test)]
    pub(super) fn plan_evictions(&self, target: &str) -> Result<Vec<Eviction>> {
        let registry = Registry::load(&self.inner.root)?;
        let state = State::load_checked(&self.inner.root)?;
        let scene = desired_scene(&self.inner.root, &state)?;
        Ok(self
            .inner
            .eviction_candidates(&registry, &state, &scene, target))
    }
}

impl Lifecycle for ModelLifecycle {
    fn state(&self, id: &str) -> ModelState {
        self.inner.state(id)
    }

    fn ensure_ready(&self, id: &str, deadline: Instant) -> Result<()> {
        if self.inner.state(id) == ModelState::Ready {
            return Ok(());
        }
        let flight = {
            let mut flights = self.inner.flights.lock().unwrap_or_else(|e| e.into_inner());
            match flights.get(id) {
                Some(flight) => Arc::clone(flight),
                None => {
                    let flight = Arc::new(Flight::default());
                    flights.insert(id.to_string(), Arc::clone(&flight));
                    let inner = Arc::clone(&self.inner);
                    let worker_flight = Arc::clone(&flight);
                    let worker_id = id.to_string();
                    let lock_deadline = deadline.max(Instant::now() + MIN_LOCK_PATIENCE);
                    std::thread::spawn(move || {
                        let outcome = inner
                            .bring_up(&worker_id, lock_deadline)
                            .map_err(|error| format!("{error:#}"));
                        // Leave the map first: a caller arriving after this sees
                        // the new state and starts fresh rather than reusing us.
                        {
                            let mut flights =
                                inner.flights.lock().unwrap_or_else(|e| e.into_inner());
                            if flights
                                .get(&worker_id)
                                .is_some_and(|f| Arc::ptr_eq(f, &worker_flight))
                            {
                                flights.remove(&worker_id);
                            }
                        }
                        let mut slot = worker_flight
                            .outcome
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        *slot = Some(outcome);
                        worker_flight.done.notify_all();
                    });
                    flight
                }
            }
        };
        let mut slot = flight.outcome.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(outcome) = slot.as_ref() {
                return outcome.clone().map_err(|error| anyhow::anyhow!(error));
            }
            let now = Instant::now();
            if now >= deadline {
                bail!("'{id}' was not ready by the deadline; its wake or start continues");
            }
            slot = flight
                .done
                .wait_timeout(slot, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    fn sleep(&self, id: &str, reason: &str) -> Result<()> {
        self.inner.put_down(id, reason, false)
    }
}

/// Poll the transition lock until `deadline`. The other holders only try it
/// once and back off, so polling here never deadlocks with them.
fn acquire_lock(root: &Path, deadline: Instant, poll: Duration) -> Result<TransitionLock> {
    loop {
        match TransitionLock::acquire(root) {
            Ok(lock) => return Ok(lock),
            Err(error) if Instant::now() >= deadline => {
                return Err(error).context("waiting for the transition lock")
            }
            Err(_) => std::thread::sleep(poll.min(Duration::from_millis(100))),
        }
    }
}

fn desired_scene(root: &Path, state: &State) -> Result<Scene> {
    let desired = state
        .desired_scene
        .as_deref()
        .context("no desired Scene is persisted; activate one first")?;
    find_scene(root, desired)
}

fn lookup<'a>(registry: &'a Registry, id: &str) -> Result<&'a Model> {
    registry
        .models
        .get(id)
        .with_context(|| format!("no registered model '{id}'"))
}

fn drain_timeout(model: &Model) -> Duration {
    model
        .serve
        .sleep
        .as_ref()
        .map_or(Duration::from_secs(DEFAULT_DRAIN_TIMEOUT_SEC), |sleep| {
            sleep.drain_timeout()
        })
}

fn wake_timeout(model: &Model) -> Duration {
    model.serve.sleep.as_ref().map_or(
        Duration::from_secs(crate::config::DEFAULT_SLEEP_WAKE_TIMEOUT_SEC),
        |sleep| sleep.wake_timeout(),
    )
}

fn startup_window(model: &Model) -> Duration {
    model
        .supervision
        .as_ref()
        .map_or(DEFAULT_STARTUP_WINDOW, |policy| {
            Duration::from_secs(policy.startup_timeout_sec())
        })
}

impl Inner {
    fn now(&self) -> u64 {
        self.host.now_ms()
    }

    /// Any request at all: in flight at the gateway (held ones included) or
    /// reported by the engine. Eviction never picks such a model.
    fn busy(&self, id: &str, model: &Model) -> bool {
        self.host.in_flight(id) > 0 || self.host.engine_in_flight(model).unwrap_or(0) > 0
    }

    /// Requests actually being served: gateway in-flight minus those held for
    /// a wake, or the engine's own count. A drain waits only for these, so a
    /// request held for this very model can never block its own drain.
    fn active(&self, id: &str, model: &Model) -> bool {
        self.host
            .in_flight(id)
            .saturating_sub(self.host.waiting(id))
            > 0
            || self.host.engine_in_flight(model).unwrap_or(0) > 0
    }

    /// From the in-memory cache; re-read from disk at most once per [`CACHE_TTL`].
    /// A model with no live owned record is `Stopped` here, even one serving
    /// outside Lumbridge Computer's ownership: `ensure_ready` on it finds it answering and
    /// returns at once, without touching it.
    fn state(&self, id: &str) -> ModelState {
        {
            let cache = self.cache.read().unwrap_or_else(|e| e.into_inner());
            let now = (self.clock)();
            if cache
                .loaded
                .is_some_and(|at| now.saturating_duration_since(at) < CACHE_TTL)
            {
                return cache.states.get(id).copied().unwrap_or(ModelState::Stopped);
            }
        }
        match State::load_checked(&self.root) {
            Ok(state) => self.remember(&state),
            Err(_) => {
                // Keep the last known states, but don't re-read on every call.
                let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
                cache.loaded = Some((self.clock)());
            }
        }
        let cache = self.cache.read().unwrap_or_else(|e| e.into_inner());
        cache.states.get(id).copied().unwrap_or(ModelState::Stopped)
    }

    fn remember(&self, state: &State) {
        let states = state
            .procs
            .iter()
            .filter(|(_, process)| process.owned_alive())
            .map(|(id, process)| (id.clone(), process.model_state()))
            .collect();
        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
        *cache = StateCache {
            loaded: Some((self.clock)()),
            states,
        };
    }

    /// Persist and update the cache: every transition goes through here.
    fn save(&self, state: &State) -> Result<()> {
        state.save(&self.root)?;
        self.remember(state);
        Ok(())
    }

    // ---- bring up ---------------------------------------------------------

    fn bring_up(&self, id: &str, lock_deadline: Instant) -> Result<()> {
        let poll = self.host.poll();
        let lock = acquire_lock(&self.root, lock_deadline, poll)?;
        let registry = Registry::load(&self.root)?;
        let mut state = State::load_checked(&self.root)?;
        self.remember(&state);
        let scene = desired_scene(&self.root, &state)?;
        if !scene.models.iter().any(|m| m == id) {
            bail!(
                "'{id}' is not part of desired Scene '{}'",
                scene.metadata.name
            );
        }
        let model = lookup(&registry, id)?.clone();
        let mode = scene.mode_of(id).mode;
        let owned = state
            .procs
            .get(id)
            .filter(|process| process.owned_alive())
            .cloned();

        let Some(process) = owned else {
            if governor::is_running(&model) {
                return Ok(());
            }
            if mode == Mode::Resident {
                bail!(
                    "resident model '{id}' is not running; Scene activation and supervision \
                     start resident models, not ensure_ready"
                );
            }
            return self.cold_start(lock, &registry, &mut state, &scene, id, &model);
        };

        match process.model_state() {
            ModelState::Ready => Ok(()),
            ModelState::Starting => {
                drop(lock);
                self.finish_start(id, &model, &process)
            }
            ModelState::Waking if !self.stale_wake(&process, &model) => {
                // Another process is running the wake calls; wait for readiness
                // on the same budget, counted from when that wake began.
                drop(lock);
                let since = process.state_since_ms.unwrap_or_else(|| self.now());
                let spent = Duration::from_millis(self.now().saturating_sub(since));
                let deadline = Instant::now() + wake_timeout(&model).saturating_sub(spent);
                self.finish_wake(id, &model, &process, deadline, Ok(()))
            }
            ModelState::Draining if engine::is_sleeping(&model).ok() == Some(false) => {
                // A drain under the lock means its owner died mid-drain before
                // sleeping: the engine is still awake, so it is simply ready.
                let now = self.now();
                if let Some(record) = state.procs.get_mut(id) {
                    record.set_state(ModelState::Ready, now);
                }
                self.save(&state)?;
                Ok(())
            }
            ModelState::Stopped => {
                // Never written for a live process; treat it as an orphan record.
                state.procs.remove(id);
                self.cold_start(lock, &registry, &mut state, &scene, id, &model)
            }
            ModelState::Sleeping { .. } | ModelState::Draining | ModelState::Waking => {
                self.wake(lock, &registry, &mut state, &scene, id, &model, &process)
            }
        }
    }

    fn stale_wake(&self, process: &Proc, model: &Model) -> bool {
        let since = process.state_since_ms.unwrap_or(0);
        let limit = wake_timeout(model) + STALE_WAKE_GRACE;
        self.now().saturating_sub(since) > limit.as_millis() as u64
    }

    #[allow(clippy::too_many_arguments)]
    fn wake(
        &self,
        lock: TransitionLock,
        registry: &Registry,
        state: &mut State,
        scene: &Scene,
        id: &str,
        model: &Model,
        process: &Proc,
    ) -> Result<()> {
        if !engine::supports_sleep(model) {
            bail!("'{id}' is recorded asleep but its engine cannot sleep or wake");
        }
        self.admit(registry, state, scene, id)?;
        let now = self.now();
        let record = state
            .procs
            .get_mut(id)
            .context("the process record vanished during admission")?;
        record.set_state(ModelState::Waking, now);
        self.save(state)?;
        drop(lock);
        eprintln!("states: waking '{id}'");
        // One budget for the calls and the readiness wait together.
        let deadline = Instant::now() + wake_timeout(model);
        let called = engine::wake(model, deadline).map(|reload| {
            if let Some(reload) = reload {
                eprintln!(
                    "states: '{id}' reloaded weights: target {}s, drafter {}s (loaded: {})",
                    reload.target_secs.map_or("?".into(), |s| s.to_string()),
                    reload.draft_secs.map_or("-".into(), |s| s.to_string()),
                    reload.draft_loaded.map_or("-".into(), |b| b.to_string()),
                );
            }
        });
        self.finish_wake(id, model, process, deadline, called)
    }

    /// Wait until a woken engine answers and no longer reports sleeping, then
    /// record the outcome under the lock. Any failure stops the engine: after a
    /// level-2 sleep the weights may be torn, and the spike saw the next request
    /// kill the EngineCore. The next request cold-starts it instead.
    fn finish_wake(
        &self,
        id: &str,
        model: &Model,
        process: &Proc,
        deadline: Instant,
        called: Result<()>,
    ) -> Result<()> {
        let started = Instant::now();
        let outcome = called.and_then(|()| loop {
            if governor::readiness(model, model.probe_timeout()) == Readiness::Ready
                && engine::is_sleeping(model).ok() == Some(false)
            {
                return Ok(());
            }
            if !process.owned_alive() {
                bail!("'{id}' exited while waking");
            }
            if Instant::now() >= deadline {
                bail!(
                    "'{id}' did not become ready within its {}s wake timeout",
                    wake_timeout(model).as_secs()
                );
            }
            std::thread::sleep(self.host.poll());
        });

        let _lock = acquire_lock(
            &self.root,
            Instant::now() + FINISH_LOCK_PATIENCE,
            self.host.poll(),
        )?;
        let mut state = State::load_checked(&self.root)?;
        let now = self.now();
        let same = state
            .procs
            .get(id)
            .is_some_and(|record| record.pid == process.pid);
        if !same {
            return outcome.and_then(|()| bail!("'{id}' was replaced while waking"));
        }
        match outcome {
            Ok(()) => {
                if let Some(record) = state.procs.get_mut(id) {
                    if record.model_state() != ModelState::Ready {
                        record.set_state(ModelState::Ready, now);
                    }
                }
                self.save(&state)?;
                eprintln!(
                    "states: '{id}' is ready after waking ({:.1}s)",
                    started.elapsed().as_secs_f64()
                );
                Ok(())
            }
            Err(error) => {
                // Never serve after a partial wake: whether it still reports
                // sleeping or not, the weights (or the MTP drafter) may not be
                // what the checkpoint says. Stop it; the next request starts it
                // cold with the long deadline.
                let stopped = state
                    .procs
                    .remove(id)
                    .map(|record| proc::stop_owned(&record));
                self.save(&state)?;
                let error = error.context(format!(
                    "waking '{id}' failed; stopped it rather than serve from a partial wake"
                ));
                match stopped {
                    Some(Err(stop)) => {
                        Err(error.context(format!("and stopping it failed: {stop:#}")))
                    }
                    _ => Err(error),
                }
            }
        }
    }

    fn cold_start(
        &self,
        lock: TransitionLock,
        registry: &Registry,
        state: &mut State,
        scene: &Scene,
        id: &str,
        model: &Model,
    ) -> Result<()> {
        if let Some(port) = model.serve.port {
            if governor::port_open(port) {
                bail!("port {port} is occupied by a process Lumbridge Computer does not own; refusing to start '{id}'");
            }
        }
        state.procs.remove(id);
        self.admit(registry, state, scene, id)?;
        let pid = proc::spawn(&self.root, id, model, crate::scope::mode(registry.scopes))?;
        state.seq += 1;
        let mut process = Proc::capture(pid, state.seq, model.serve.port)
            .with_context(|| format!("capturing ownership for '{id}'"))?;
        process.container_id = proc::await_container(&self.root, id, model, &process);
        process.set_state(ModelState::Starting, self.now());
        state.procs.insert(id.to_string(), process.clone());
        self.save(state)?;
        drop(lock);
        eprintln!("states: cold-starting '{id}' (pid {pid})");
        self.finish_start(id, model, &process)
    }

    /// Wait for a starting model (ours or another process's) with the same
    /// progress rules as a Scene start, then record the outcome under the lock.
    fn finish_start(&self, id: &str, model: &Model, process: &Proc) -> Result<()> {
        let started = Instant::now();
        let outcome = crate::lifecycle::wait_started(
            model,
            process,
            &proc::log_path(&self.root, id),
            startup_window(model),
        );
        let _lock = acquire_lock(
            &self.root,
            Instant::now() + FINISH_LOCK_PATIENCE,
            self.host.poll(),
        )?;
        let mut state = State::load_checked(&self.root)?;
        let now = self.now();
        let same = state
            .procs
            .get(id)
            .is_some_and(|record| record.pid == process.pid);
        match outcome {
            Ok(()) if same => {
                if let Some(record) = state.procs.get_mut(id) {
                    if record.model_state() == ModelState::Starting {
                        record.set_state(ModelState::Ready, now);
                    }
                }
                self.save(&state)?;
                eprintln!(
                    "states: '{id}' is ready after a cold start ({:.0}s)",
                    started.elapsed().as_secs_f64()
                );
                Ok(())
            }
            Ok(()) => bail!("'{id}' was replaced while starting"),
            Err(error) => {
                if same {
                    if let Some(record) = state.procs.remove(id) {
                        let _ = proc::stop_owned(&record);
                    }
                    self.save(&state)?;
                }
                Err(error.context(format!("cold start of '{id}' failed")))
            }
        }
    }

    // ---- admission --------------------------------------------------------

    /// Make room for `id` at its full footprint, evicting per the plan if needed.
    /// Called under the lock; `state` is kept current through every eviction.
    fn admit(&self, registry: &Registry, state: &mut State, scene: &Scene, id: &str) -> Result<()> {
        let model = lookup(registry, id)?;
        let pool = model.pool.pool();
        let full = crate::footprint::admission_gb(&self.root, registry, id);
        let current = counted_gb(&self.root, registry, state, id);
        let delta = (full - current).max(0.0);
        let check = |state: &State| -> (bool, f64, f64, Option<f64>) {
            let committed = super::committed_gb(&self.root, registry, state, &pool);
            let memory = self.host.memory(&pool);
            let budget = pool.effective_budget_gb(
                pool.scene_budget_gb(scene.budget_gb),
                memory.map(|(total, _)| total),
                pool.reserved_gb(registry),
            );
            let available =
                memory.map(|(_, available)| pool.model_available_gb(registry, available));
            (
                pool.can_admit(delta, committed, budget, available),
                committed,
                budget,
                available,
            )
        };
        let (fits, committed, budget, available) = check(state);
        if fits {
            return Ok(());
        }

        let candidates = self.eviction_candidates(registry, state, scene, id);
        let mut plan = Vec::new();
        let mut freed = 0.0;
        let mut enough = false;
        for step in candidates {
            freed += step.freed_gb;
            plan.push(step);
            if pool.can_admit(
                delta,
                committed - freed,
                budget,
                available.map(|available| available + freed),
            ) {
                enough = true;
                break;
            }
        }
        if !enough {
            bail!(
                "no room for '{id}' ({full:.0} GB, {delta:.1} GB more than it holds now): \
                 {committed:.1} GB committed against a {budget:.0} GB budget{}, {:.0} GB margin; \
                 sleeping or stopping every idle non-resident model frees only {freed:.1} GB",
                available.map_or(String::new(), |a| format!(", {a:.1} GB available")),
                pool.safety_margin_gb
            );
        }

        for step in &plan {
            let victim = lookup(registry, &step.id)?;
            eprintln!(
                "states: {} '{}' to make room for '{id}' (frees ~{:.1} GB)",
                if step.stop { "stopping" } else { "sleeping" },
                step.id,
                step.freed_gb
            );
            let reason = format!("making room for '{id}'");
            self.put_down_locked(state, &step.id, victim, step.stop, EVICTION_DRAIN, &reason)
                .with_context(|| format!("evicting '{}' for '{id}'", step.id))?;
        }

        // Freed memory can take a moment to show up in MemAvailable.
        let deadline = Instant::now() + RELEASE_WAIT;
        loop {
            let (fits, committed, budget, available) = check(state);
            if fits {
                return Ok(());
            }
            if Instant::now() >= deadline {
                bail!(
                    "made room for '{id}' but it still does not fit: {committed:.1} GB committed \
                     against {budget:.0} GB{}",
                    available.map_or(String::new(), |a| format!(", {a:.1} GB available"))
                );
            }
            std::thread::sleep(self.host.poll());
        }
    }

    /// Who may be evicted for `target`, in the order the plan evicts them:
    /// first sleep ready models (stop them if their engine can't sleep), then
    /// stop sleeping ones. Within each phase the highest `shed_priority` (the
    /// watchdog's "shed me first") goes first, then the least recently used.
    /// Never a resident model, never one with requests in flight, never one
    /// mid-transition, and only models in `target`'s pool.
    pub(super) fn eviction_candidates(
        &self,
        registry: &Registry,
        state: &State,
        scene: &Scene,
        target: &str,
    ) -> Vec<Eviction> {
        let Some(pool) = registry.models.get(target).map(|m| m.pool) else {
            return Vec::new();
        };
        struct Candidate {
            id: String,
            asleep: bool,
            can_sleep: bool,
            priority: i32,
            last_active: u64,
            counted: f64,
            sleeping: f64,
        }
        let mut candidates = Vec::new();
        for id in &scene.models {
            if id == target || scene.mode_of(id).mode == Mode::Resident {
                continue;
            }
            let Some(model) = registry.models.get(id) else {
                continue;
            };
            if model.pool != pool {
                continue;
            }
            let Some(process) = state.procs.get(id).filter(|p| p.owned_alive()) else {
                continue;
            };
            let asleep = match process.model_state() {
                ModelState::Ready => false,
                ModelState::Sleeping { .. } => true,
                _ => continue,
            };
            if self.busy(id, model) {
                continue;
            }
            let last_active = self
                .host
                .last_finished_ms(id)
                .into_iter()
                .chain(process.state_since_ms)
                .max()
                .unwrap_or(0);
            candidates.push(Candidate {
                id: id.clone(),
                asleep,
                can_sleep: engine::supports_sleep(model),
                priority: model.shed_priority.unwrap_or(0),
                last_active,
                counted: counted_gb(&self.root, registry, state, id),
                sleeping: sleeping_gb(&self.root, registry, id, Some(process)),
            });
        }
        // Highest shed_priority first, then oldest activity, then id for a stable order.
        candidates.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(a.last_active.cmp(&b.last_active))
                .then(a.id.cmp(&b.id))
        });
        let mut plan = Vec::new();
        // Phase 1: sleep (or stop) the ready ones.
        for c in candidates.iter().filter(|c| !c.asleep) {
            if c.can_sleep {
                plan.push(Eviction {
                    id: c.id.clone(),
                    stop: false,
                    freed_gb: (c.counted - c.sleeping).max(0.0),
                });
            } else {
                plan.push(Eviction {
                    id: c.id.clone(),
                    stop: true,
                    freed_gb: c.counted,
                });
            }
        }
        // Phase 2: stop the sleeping ones, including those phase 1 just slept.
        for c in &candidates {
            if c.asleep {
                plan.push(Eviction {
                    id: c.id.clone(),
                    stop: true,
                    freed_gb: c.counted,
                });
            } else if c.can_sleep {
                plan.push(Eviction {
                    id: c.id.clone(),
                    stop: true,
                    freed_gb: c.sleeping,
                });
            }
        }
        plan
    }

    // ---- put down (sleep / stop) -------------------------------------------

    fn put_down(&self, id: &str, reason: &str, stop: bool) -> Result<()> {
        let _lock = acquire_lock(
            &self.root,
            Instant::now() + SLEEP_LOCK_PATIENCE,
            self.host.poll(),
        )?;
        let registry = Registry::load(&self.root)?;
        let mut state = State::load_checked(&self.root)?;
        let scene = desired_scene(&self.root, &state)?;
        if !scene.models.iter().any(|m| m == id) {
            bail!(
                "'{id}' is not part of desired Scene '{}'",
                scene.metadata.name
            );
        }
        if scene.mode_of(id).mode == Mode::Resident {
            bail!(
                "'{id}' is resident in Scene '{}'; refusing to {} it (make it sleepable or \
                 on_demand in the Scene's modes)",
                scene.metadata.name,
                if stop { "stop" } else { "sleep" }
            );
        }
        let model = lookup(&registry, id)?.clone();
        let Some(process) = state.procs.get(id).filter(|p| p.owned_alive()).cloned() else {
            if governor::is_running(&model) {
                bail!(
                    "'{id}' is serving but not owned by Lumbridge Computer; refusing to signal it"
                );
            }
            return Ok(()); // already stopped
        };
        match process.model_state() {
            ModelState::Ready => {}
            ModelState::Sleeping { .. } if !stop => return Ok(()),
            ModelState::Sleeping { .. } => {}
            other => bail!("'{id}' is {other}; try again once it settles"),
        }
        self.put_down_locked(&mut state, id, &model, stop, drain_timeout(&model), reason)
    }

    /// Drain, then sleep (or stop: when asked, or when the engine can't sleep).
    /// Under the lock. A drain that times out puts the model back to `Ready`.
    fn put_down_locked(
        &self,
        state: &mut State,
        id: &str,
        model: &Model,
        stop: bool,
        drain: Duration,
        reason: &str,
    ) -> Result<()> {
        let record = state
            .procs
            .get_mut(id)
            .with_context(|| format!("no process record for '{id}'"))?;
        let was = record.model_state();
        let was_since = record.state_since_ms;
        let asleep = matches!(was, ModelState::Sleeping { .. });
        if !asleep {
            record.set_state(ModelState::Draining, self.now());
            self.save(state)?;
            let deadline = Instant::now() + drain;
            while self.active(id, model) {
                if Instant::now() >= deadline {
                    if let Some(record) = state.procs.get_mut(id) {
                        record.state = Some(was);
                        record.state_since_ms = was_since;
                    }
                    self.save(state)?;
                    bail!(
                        "'{id}' still had requests in flight after {}s of draining; left it {was}",
                        drain.as_secs()
                    );
                }
                std::thread::sleep(self.host.poll());
            }
        }

        if !stop && engine::supports_sleep(model) {
            let level = model.serve.sleep.as_ref().map_or(2, |sleep| sleep.level);
            let slept = engine::sleep(model, level);
            let now = self.now();
            let record = state
                .procs
                .get_mut(id)
                .with_context(|| format!("no process record for '{id}'"))?;
            match slept {
                Ok(()) => {
                    record.set_state(ModelState::Sleeping { level }, now);
                    self.save(state)?;
                    eprintln!("states: '{id}' is asleep (level {level}): {reason}");
                    Ok(())
                }
                Err(error) if engine::is_sleeping(model).ok() == Some(true) => {
                    record.set_state(ModelState::Sleeping { level }, now);
                    self.save(state)?;
                    eprintln!("states: '{id}' is asleep despite {error:#}: {reason}");
                    Ok(())
                }
                Err(error) => {
                    record.set_state(ModelState::Ready, now);
                    self.save(state)?;
                    Err(error.context(format!("sleeping '{id}' failed; left it ready")))
                }
            }
        } else {
            let process = state
                .procs
                .get(id)
                .cloned()
                .with_context(|| format!("no process record for '{id}'"))?;
            proc::stop_owned(&process).with_context(|| format!("stopping '{id}'"))?;
            state.procs.remove(id);
            self.save(state)?;
            eprintln!("states: '{id}' is stopped: {reason}");
            Ok(())
        }
    }
}
