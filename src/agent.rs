//! Resident Lumbridge Computer supervisor.

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use crate::config::Registry;
use crate::health::SupervisionTracker;
use crate::{footprint, gateway, governor, lifecycle, proc, states};

pub struct TelemetryConfig {
    /// Scrape only this endpoint; `None` scrapes every running model.
    pub metrics_url: Option<String>,
    pub model: String,
    pub interval: Duration,
    pub timeout: Duration,
}

pub fn run(
    root: &Path,
    gateway_options: gateway::Options,
    pool: governor::Pool,
    telemetry: Option<TelemetryConfig>,
) -> Result<()> {
    println!("Lumbridge Computer agent starting");
    lifecycle::resume(root).context("resuming persisted desired Scene")?;

    let (gateway_done_tx, gateway_done_rx) = mpsc::channel();
    // Shared with supervision so a model with requests in flight reads as busy.
    let in_flight = gateway::InFlight::default();
    let gateway_in_flight = in_flight.clone();
    // M3: one lifecycle for the whole agent. The gateway wakes models through
    // it, the idle sleeper sleeps them through it, and supervision asks it which
    // models are mid-transition. All of them read the same InFlight counters.
    let model_lifecycle = Arc::new(states::ModelLifecycle::new(
        root,
        Arc::new(states::AgentHost::new(in_flight.clone())),
    ));
    let gateway_lifecycle: Arc<dyn states::Lifecycle> = model_lifecycle.clone();
    thread::spawn(move || {
        let result = gateway::run(&gateway_options, gateway_in_flight, Some(gateway_lifecycle));
        gateway_done_tx.send(result).ok();
    });

    if let Some(telemetry) = telemetry {
        let telemetry_root = root.to_path_buf();
        println!(
            "Lumbridge Computer telemetry {} every {}s",
            telemetry
                .metrics_url
                .as_deref()
                .unwrap_or("of running models by engine"),
            telemetry.interval.as_secs()
        );
        thread::spawn(move || {
            crate::telemetry::run_collector(
                telemetry_root,
                telemetry.metrics_url,
                telemetry.model,
                telemetry.interval,
                telemetry.timeout,
            )
        });
    } else {
        println!("Lumbridge Computer telemetry disabled");
    }

    // Footprint sampling is independent of engine telemetry: it reads cgroups,
    // /proc and nvidia-smi, never a model's HTTP port.
    let footprint_root = root.to_path_buf();
    println!(
        "Lumbridge Computer footprint sampling every {}s",
        footprint::SAMPLE_INTERVAL.as_secs()
    );
    thread::spawn(move || footprint::run_sampler(footprint_root));

    // The idle sleeper puts idle sleepable/on_demand models to sleep (and stops
    // on_demand ones asleep too long) on its own thread, so a slow drain or
    // engine call never delays the memory floor.
    println!(
        "Lumbridge Computer idle sleeper every {}s",
        states::idle::TICK.as_secs()
    );
    let sleeper_lifecycle = model_lifecycle.clone();
    thread::spawn(move || states::idle::run(sleeper_lifecycle));

    println!(
        "Lumbridge Computer agent supervising the {} pool's memory floor {:.1} GB",
        pool.id, pool.floor_gb
    );
    let supervisor_started = Instant::now();
    let mut supervisor = SupervisionTracker::new(Arc::new(in_flight))
        .with_lifecycle(model_lifecycle as Arc<dyn states::Lifecycle>);
    let mut pressure_watch = crate::pressure::Watch::default();
    let mut pressure_checked_at = 0;
    loop {
        match gateway_done_rx.try_recv() {
            Ok(result) => return result.context("stable gateway stopped"),
            Err(mpsc::TryRecvError::Disconnected) => {
                anyhow::bail!("stable gateway supervisor disconnected")
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
        enforce_memory_floor(root, &pool)?;
        let elapsed = supervisor_started.elapsed().as_secs();
        if elapsed >= pressure_checked_at + PRESSURE_CHECK_SEC {
            pressure_checked_at = elapsed;
            warn_on_memory_pressure(root, &pool, &mut pressure_watch, elapsed);
        }
        if let Err(error) = supervisor.tick(root, supervisor_started.elapsed().as_secs()) {
            // A failed recovery is a degraded model, not a failed resident agent.
            // The policy's backoff will try again; taking down the gateway and
            // memory watchdog here would turn one voice outage into a node outage.
            eprintln!("agent: supervision tick failed: {error:#}");
        }
        thread::sleep(Duration::from_secs(1));
    }
}

/// How often the agent looks at memory pressure. The kernel's shortest average is
/// over 10 seconds, so looking more often than this adds nothing.
const PRESSURE_CHECK_SEC: u64 = 5;

/// Log when the kernel reports tasks stalling on memory (see `pressure`). An early
/// warning beside the floor: it never stops or refuses anything.
fn warn_on_memory_pressure(
    root: &Path,
    pool: &governor::Pool,
    watch: &mut crate::pressure::Watch,
    now: u64,
) {
    let reading = crate::pressure::read();
    // Only pay for sensing and a registry load when there is something to say.
    let worth_sensing = reading.is_some_and(|pressure| pressure.elevated());
    let available = worth_sensing
        .then(|| {
            let memory = pool.sense().ok()?;
            Some(match Registry::load(root).ok() {
                Some(registry) => pool.model_available_gb(&registry, memory.available_gb),
                None => memory.available_gb,
            })
        })
        .flatten();
    if let Some(line) = watch.observe(reading, available, now) {
        eprintln!("agent: {line}");
    }
}

fn enforce_memory_floor(root: &Path, pool: &governor::Pool) -> Result<()> {
    let floor = pool.floor_gb;
    let mut memory = pool.sense()?;
    // Guard the room reserved tenants (the Server's Postgres) may still grow into, not
    // just the raw pool: otherwise models can eat that room and the floor only trips
    // once the database is already starving. A registry that fails to parse must not
    // disable the floor, so it degrades to the unreserved behaviour.
    let registry = Registry::load(root).ok();
    if let Some(registry) = registry.as_ref() {
        memory.available_gb = pool.model_available_gb(registry, memory.available_gb);
    }
    if !pool.below_floor(memory.available_gb) {
        return Ok(());
    }
    // Take the same lock a Scene transition takes, and skip this tick if a transition
    // already holds it. Without this the watchdog races activation: activation is
    // deliberately a stop-all-then-start-all sequence, so mid-transition the pool is
    // legitimately tight and `state.procs` is being rewritten underneath us. Firing then
    // would kill a model the transition just started, and then write a stale `state`
    // over the transition's own — losing track of a process that is still alive.
    //
    // A tick skipped here costs one second. The floor is a backstop, and the transition
    // holding the lock is doing its own admission checks.
    let _lock = match lifecycle::TransitionLock::acquire(root) {
        Ok(lock) => lock,
        Err(_) => {
            eprintln!(
                "agent: memory available to models {:.1} GB < floor {:.1}, but a Scene transition holds the \
                 lock; deferring to it for this tick",
                memory.available_gb, floor
            );
            return Ok(());
        }
    };
    let mut state = proc::State::load_checked(root)?;
    // Measured only once the floor is breached: the order among equal
    // shed_priority prefers whoever is furthest over its declared footprint.
    let overage = registry.as_ref().map_or_else(Default::default, |registry| {
        let measured =
            footprint::measure_owned(&footprint::Roots::default(), &state, &footprint::NvidiaSmi);
        footprint::overage_gb(registry, &measured)
    });
    match state.shed_candidate_by(registry.as_ref(), &overage) {
        Some((id, process)) => {
            eprintln!(
                "agent: memory available to models {:.1} GB < floor {:.1}; stopping owned model '{}' (pid {}, {:.1} GB over declared) by shed_priority/overage/newest",
                memory.available_gb,
                floor,
                id,
                process.pid,
                overage.get(&id).copied().unwrap_or(0.0)
            );
            proc::shed_owned(&process)?;
            state.procs.remove(&id);
            state.active_scene = None;
            state.last_error = Some(format!(
                "watchdog stopped '{id}' after memory available to models fell to {:.1} GB",
                memory.available_gb
            ));
            state.save(root)?;
        }
        None => eprintln!(
            "agent: memory available to models {:.1} GB < floor {:.1}, but no identity-owned model can be stopped",
            memory.available_gb, floor
        ),
    }
    Ok(())
}
