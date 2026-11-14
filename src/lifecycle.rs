//! Transactional Scene lifecycle and persisted desired-state recovery.

use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::Duration;

use crate::config::{find_scene, Model, Registry, Scene};
use crate::governor::{self, Readiness};
use crate::health::{self, InFlight, Trigger};
use crate::proc::{self, Proc, State};

/// What activating a Scene would do, decided before anything is mutated.
///
/// `activate` renders this for the CLI and the MCP server returns it verbatim,
/// so a plan an operator reads and a plan an agent reads can never drift apart.
#[derive(Debug, Clone)]
pub struct Plan {
    pub scene: String,
    pub budget_gb: f64,
    /// Registered models serving outside the target Scene; stopped first.
    pub stop: Vec<String>,
    /// Scene models not yet serving, in the order they would be admitted.
    pub start: Vec<String>,
}

pub struct TransitionLock {
    file: File,
}

impl TransitionLock {
    pub fn acquire(root: &Path) -> Result<Self> {
        let dir = root.join(".compute");
        fs::create_dir_all(&dir)?;
        let path = dir.join("transition.lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("opening transition lock {}", path.display()))?;
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result != 0 {
            return Err(std::io::Error::last_os_error())
                .context("another Scene transition is already running; wait for it to finish");
        }
        Ok(Self { file })
    }
}

impl Drop for TransitionLock {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

pub fn adopt(root: &Path, name: &str) -> Result<()> {
    let _lock = TransitionLock::acquire(root)?;
    let registry = Registry::load(root)?;
    let scene = find_scene(root, name)?;
    validate_scene(&registry, &scene)?;

    let target: HashSet<&str> = scene.models.iter().map(String::as_str).collect();
    let running = governor::running_ids(&registry);
    let extras: Vec<String> = running
        .iter()
        .filter(|id| !target.contains(id.as_str()))
        .cloned()
        .collect();
    if !extras.is_empty() {
        bail!(
            "cannot adopt '{name}': registered model(s) outside the Scene are serving: {}",
            extras.join(", ")
        );
    }

    let previous = State::load_checked(root)?;
    let mut adopted = BTreeMap::new();
    for id in &scene.models {
        let model = &registry.models[id];
        // An on_demand model that is down is simply idle; nothing to adopt.
        if scene.mode_of(id).mode == crate::config::Mode::OnDemand && !governor::is_running(model) {
            continue;
        }
        if !governor::is_running(model) {
            bail!("cannot adopt '{name}': exact model '{id}' is not healthy");
        }
        let legacy = previous.procs.get(id).with_context(|| {
            format!(
                "cannot adopt '{name}': no process record for '{id}'; start it through Lumbridge Computer"
            )
        })?;
        let captured = Proc::capture(legacy.pid, legacy.seq, model.serve.port)
            .with_context(|| format!("adopting '{id}' pid {}", legacy.pid))?;
        adopted.insert(id.clone(), captured);
    }

    let mut state = previous;
    state.procs = adopted;
    state.desired_scene = Some(name.to_string());
    state.active_scene = Some(name.to_string());
    state.last_known_good_scene = Some(name.to_string());
    state.transition_scene = None;
    state.last_error = None;
    state.save(root)?;
    println!("adopted exact running Scene '{name}' and captured process identities");
    Ok(())
}

pub fn activate(root: &Path, name: &str, dry_run: bool) -> Result<()> {
    let _lock = TransitionLock::acquire(root)?;
    let prior = State::load_checked(root)?;
    let prior_active = prior.active_scene.clone();

    if dry_run {
        return activate_once(root, name, true);
    }

    match activate_once(root, name, false) {
        Ok(()) => {
            let mut state = State::load_checked(root)?;
            if prior_active.as_deref() != Some(name) {
                if let Some(previous) = prior_active {
                    state.last_known_good_scene = Some(previous);
                }
            }
            state.desired_scene = Some(name.to_string());
            state.active_scene = Some(name.to_string());
            if state.last_known_good_scene.is_none() {
                state.last_known_good_scene = Some(name.to_string());
            }
            state.transition_scene = None;
            state.last_error = None;
            state.save(root)?;
            println!("Scene '{name}' is active and persisted as desired");
            Ok(())
        }
        Err(target_error) => {
            let target_message = format!("activating '{name}' failed: {target_error:#}");
            eprintln!("{target_message}");
            let transition_started =
                State::load_checked(root)?.transition_scene.as_deref() == Some(name);
            // Where a failed transition would roll back to. Absent when the scene being
            // activated is already the active one, which is the ordinary case for
            // "start the models in this scene that are currently down".
            let rollback_target = prior_active.as_deref().filter(|previous| *previous != name);
            // Only tear the box down when there is somewhere to put it back.
            //
            // This used to run whenever a transition had begun, including when the target
            // scene *was* the prior scene — so a refused admission stopped every running
            // model and then skipped the rollback, because the rollback target had been
            // filtered out for being the same scene. That turned one
            // model failing admission into an empty box, taking the brain and music
            // down with it.
            //
            // With no rollback target, leaving the partially-started scene running is
            // strictly better: those models are healthy and they belong to the scene that
            // was asked for. The refusal is reported either way.
            let cleanup_error = if transition_started && rollback_target.is_some() {
                stop_all_owned(root).err()
            } else {
                None
            };
            let rollback = if transition_started && cleanup_error.is_none() {
                rollback_target
                    .map(|previous| (previous.to_string(), activate_once(root, previous, false)))
            } else {
                None
            };

            let mut state = State::load_checked(root)?;
            state.transition_scene = None;
            state.last_error = Some(target_message.clone());
            match rollback {
                Some((previous, Ok(()))) => {
                    state.desired_scene = Some(previous.clone());
                    state.active_scene = Some(previous.clone());
                    if state.last_known_good_scene.is_none() {
                        state.last_known_good_scene = Some(previous.clone());
                    }
                    state.save(root)?;
                    eprintln!("rolled back to Scene '{previous}'");
                    bail!("{target_message}; rolled back to '{previous}'")
                }
                Some((previous, Err(rollback_error))) => {
                    state.active_scene = None;
                    state.save(root)?;
                    bail!(
                        "{target_message}; rollback to '{previous}' also failed: {rollback_error:#}"
                    )
                }
                None if !transition_started => {
                    state.save(root)?;
                    bail!("{target_message}")
                }
                None if cleanup_error.is_none() => {
                    state.active_scene = None;
                    state.save(root)?;
                    bail!("{target_message}")
                }
                None => {
                    state.active_scene = None;
                    state.save(root)?;
                    let cleanup_error = cleanup_error.expect("guarded by match condition");
                    bail!("{target_message}; cleanup also failed: {cleanup_error:#}")
                }
            }
        }
    }
}

fn stop_all_owned(root: &Path) -> Result<()> {
    let mut state = State::load_checked(root)?;
    let owned: Vec<(String, Proc)> = state
        .procs
        .iter()
        .filter(|(_, process)| process.owned_alive())
        .map(|(id, process)| (id.clone(), process.clone()))
        .collect();
    for (id, process) in owned {
        proc::stop_owned(&process).with_context(|| format!("cleaning up '{id}'"))?;
        state.procs.remove(&id);
        state.save(root)?;
    }
    Ok(())
}

pub fn resume(root: &Path) -> Result<()> {
    let state = State::load_checked(root)?;
    let desired = state
        .desired_scene
        .clone()
        .or_else(|| state.last_known_good_scene.clone())
        .context("no desired Scene is persisted; activate or adopt one first")?;
    let fallback = state.last_known_good_scene.clone();

    match activate(root, &desired, false) {
        Ok(()) => Ok(()),
        Err(desired_error) => {
            let Some(fallback) = fallback.filter(|fallback| fallback != &desired) else {
                return Err(desired_error).context("resuming desired Scene");
            };
            eprintln!(
                "desired Scene '{desired}' did not resume; trying last-known-good '{fallback}'"
            );
            activate(root, &fallback, false).with_context(|| {
                format!(
                    "desired Scene '{desired}' failed ({desired_error:#}) and fallback '{fallback}' failed"
                )
            })
        }
    }
}

/// Outcome of a supervised restart that did not fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Restart {
    /// A new process was started and reported exact health.
    Restarted,
    /// The re-check under the lock found the trigger no longer holds.
    Skipped(String),
}

/// Re-check the agent's trigger under the transition lock. The probes that
/// triggered it ran before the lock; a late request, a runtime finishing its
/// own recovery, or real traffic may have changed the picture in that gap.
/// Returns why the restart should be skipped, or `None` to go ahead.
fn recheck(
    model: &Model,
    process: Option<&Proc>,
    trigger: Trigger,
    in_flight: u32,
) -> Option<String> {
    let readiness = governor::readiness(model, model.probe_timeout());
    let busy = |load: Option<health::Load>| {
        let (running, waiting) = load.map_or((0, 0), |load| (load.running, load.waiting));
        (running.saturating_add(waiting).max(in_flight) > 0)
            .then(|| format!("busy: running={running} waiting={waiting} gateway={in_flight}"))
    };
    match trigger {
        // An exited process cannot come back; a live owned one that answers
        // means something already replaced it.
        Trigger::Exited => (process.is_some_and(Proc::owned_alive)
            && readiness == Readiness::Ready)
            .then(|| "a live owned process is serving it".to_string()),
        Trigger::Unready(_) => match readiness {
            Readiness::Ready => Some("it answers readiness again".to_string()),
            Readiness::Timeout => busy(health::engine_load(model)),
            Readiness::Refused | Readiness::Mismatch => None,
        },
        // A wedged engine still passes readiness, so look at progress. Only a
        // drained engine or a moved token count clears it; unreadable metrics
        // keep the verdict the agent reached over the whole stall window.
        Trigger::Stalled { tokens, .. } => {
            if readiness != Readiness::Ready {
                return None;
            }
            let load = health::engine_load(model)?;
            if load.in_flight().max(in_flight) == 0 {
                Some("it drained its requests".to_string())
            } else if load.tokens.is_some_and(|now| now != tokens) {
                Some("its token count moved".to_string())
            } else {
                None
            }
        }
        // An idle-looking hang is only still a hang if, while still idle, it
        // still won't answer one token.
        Trigger::ProbeFailed => {
            if readiness != Readiness::Ready {
                return None;
            }
            if let Some(busy) = busy(health::engine_load(model)) {
                return Some(busy);
            }
            let chat = model
                .supervision
                .as_ref()
                .and_then(|policy| policy.liveness.as_ref())
                .and_then(|liveness| liveness.chat_probe.as_ref())?;
            governor::chat_probe(model, chat).then(|| "it answered a 1-token chat".to_string())
        }
    }
}

/// A supervised restart that memory admission turned down. Nothing was spawned and
/// nothing about the model is broken: the pool is full right now. Supervision tells
/// this apart from a start that crashed (`health::SupervisionTracker::complete_refused_restart`)
/// because the two want opposite retry behaviour. A crash loop must back off; a full
/// pool should be asked again soon, since asking costs one read of `/proc/meminfo`
/// and memory frees up on its own schedule.
#[derive(Debug)]
pub struct AdmissionRefused(pub String);

impl std::fmt::Display for AdmissionRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AdmissionRefused {}

/// Restart one explicitly supervised model without disturbing the other models
/// in the desired Scene.
///
/// This is deliberately narrower than `activate`: a resident recovery attempt
/// must not turn one failed voice runtime into a stop-all rollback of an
/// otherwise healthy Scene. The same transition lock, identity ownership,
/// declared/observed admission checks and exact health gate still apply.
pub fn restart_supervised_model(
    root: &Path,
    id: &str,
    trigger: Trigger,
    in_flight: &dyn InFlight,
) -> Result<Restart> {
    let _lock = TransitionLock::acquire(root)?;
    let result = restart_supervised_model_locked(root, id, trigger, in_flight);
    if let Err(error) = &result {
        if let Ok(mut state) = State::load_checked(root) {
            state.active_scene = None;
            state.last_error = Some(format!("supervision could not restart '{id}': {error:#}"));
            let _ = state.save(root);
        }
    }
    result
}

fn restart_supervised_model_locked(
    root: &Path,
    id: &str,
    trigger: Trigger,
    in_flight: &dyn InFlight,
) -> Result<Restart> {
    let registry = Registry::load(root)?;
    let model = registry
        .models
        .get(id)
        .with_context(|| format!("no registered model '{id}'"))?;
    let policy = model
        .supervision
        .as_ref()
        .with_context(|| format!("model '{id}' did not opt into supervision"))?;
    let port = model
        .serve
        .port
        .with_context(|| format!("supervised model '{id}' has no serving port"))?;

    let mut state = State::load_checked(root)?;
    let desired = state
        .desired_scene
        .clone()
        .context("no desired Scene is persisted")?;
    let scene = find_scene(root, &desired)?;
    validate_scene(&registry, &scene)?;
    if !scene.models.iter().any(|candidate| candidate == id) {
        bail!("model '{id}' is not part of desired Scene '{desired}'");
    }
    // The agent checked this before the lock; the lifecycle may have put the
    // model to sleep (or begun waking it) since. Never restart it for that.
    if crate::states::supervision_paused(scene.mode_of(id).mode, state.procs.get(id)) {
        let what = state
            .procs
            .get(id)
            .map_or("stopped".to_string(), |p| p.model_state().to_string());
        return Ok(Restart::Skipped(format!(
            "it is {what} under the lifecycle"
        )));
    }

    if let Some(reason) = recheck(model, state.procs.get(id), trigger, in_flight.get(id)) {
        return Ok(Restart::Skipped(reason));
    }

    match state.procs.get(id).cloned() {
        Some(process) if process.owned_alive() || process.container_id.is_some() => {
            // Health is bad but this is still the exact process Lumbridge Computer started
            // (or, for a docker model whose CLI died, the exact container it
            // started). Stop only that; no sibling in the Scene is touched.
            proc::stop_owned(&process)?;
        }
        Some(_) if governor::port_open(port) => {
            bail!(
                "port {port} is occupied but the ownership record for '{id}' is stale; refusing to signal it"
            );
        }
        None if governor::port_open(port) => {
            bail!(
                "port {port} is occupied by an unowned process; refusing to replace supervised model '{id}'"
            );
        }
        Some(_) | None => {}
    }

    state.procs.remove(id);
    state.active_scene = None;
    state.last_error = Some(format!("supervision is restarting '{id}'"));
    state.save(root)?;

    // Re-read both ceilings after the old process has released its memory. The
    // registry number prevents paper overcommit; MemAvailable prevents an
    // unregistered workload from turning a nominally valid restart into a wedged
    // unified-memory node.
    // Footprint says how many GB this model and the running ones count for
    // (`admission: measured` may raise them); the pool decides whether they fit.
    let pool = model.pool.pool();
    let committed = crate::footprint::committed_gb(root, &registry, &pool);
    let add_gb = crate::footprint::admission_gb(root, &registry, id);
    // The observed ceiling asks for what the model really takes, not its typed figure.
    let observed_gb = crate::footprint::observed_gb(root, &registry, id);
    let memory = pool.sense().ok();
    let available = memory
        .as_ref()
        .map(|memory| pool.model_available_gb(&registry, memory.available_gb));
    let budget = pool.effective_budget_gb(
        pool.scene_budget_gb(scene.budget_gb),
        memory.as_ref().map(|memory| memory.total_gb),
        pool.reserved_gb(&registry),
    );
    if !pool.can_admit_observed(add_gb, observed_gb, committed, budget, available) {
        let reason = match available {
            Some(available) => format!(
                "'{id}' ({:.0} GB, {:.1} GB measured) restart refused: {:.0} GB committed against \
                 a {:.0} GB budget, {:.1} GB actually available, {:.0} GB margin required",
                add_gb, observed_gb, committed, budget, available, pool.safety_margin_gb
            ),
            None => format!(
                "'{id}' restart would exceed the {:.0} GB Scene budget with the safety margin",
                budget
            ),
        };
        return Err(AdmissionRefused(reason).into());
    }

    let pid = proc::spawn(root, id, model, crate::scope::mode(registry.scopes))?;
    state.seq += 1;
    let mut process = Proc::capture(pid, state.seq, model.serve.port)
        .with_context(|| format!("capturing ownership for supervised '{id}'"))?;
    process.container_id = proc::await_container(root, id, model, &process);
    state.procs.insert(id.to_string(), process.clone());
    state.save(root)?;

    let window = Duration::from_secs(policy.startup_timeout_sec());
    if let Err(error) = wait_started(model, &process, &proc::log_path(root, id), window) {
        let _ = proc::stop_owned(&process);
        state.procs.remove(id);
        state.last_error = Some(format!("supervised model '{id}' {error:#}"));
        state.save(root)?;
        bail!("supervised model '{id}' {error:#}");
    }

    let unhealthy: Vec<String> = scene
        .models
        .iter()
        .filter(|candidate| crate::states::expects_healthy(&scene, &state, candidate))
        .filter(|candidate| !governor::is_running(&registry.models[*candidate]))
        .cloned()
        .collect();
    if unhealthy.is_empty() {
        state.active_scene = Some(desired);
        state.last_error = None;
    } else {
        state.active_scene = None;
        state.last_error = Some(format!(
            "supervised model '{id}' recovered, but desired Scene is still unhealthy: {}",
            unhealthy.join(", ")
        ));
    }
    state.save(root)?;
    Ok(Restart::Restarted)
}

/// What counts as startup progress: the port opening, the log growing, or the
/// metrics endpoint answering. Each only ever moves forward.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct StartupProgress {
    port_open: bool,
    log_len: u64,
    metrics: bool,
}

impl StartupProgress {
    fn sample(model: &Model, log: &Path) -> Self {
        let port_open = model.serve.port.is_some_and(governor::port_open);
        StartupProgress {
            port_open,
            log_len: fs::metadata(log).map_or(0, |meta| meta.len()),
            metrics: port_open && health::engine_load(model).is_some(),
        }
    }

    fn advanced_from(&self, before: &Self) -> bool {
        (self.port_open && !before.port_open)
            || self.log_len > before.log_len
            || (self.metrics && !before.metrics)
    }
}

/// How long a start may keep making progress without becoming ready, as a
/// multiple of the no-progress window. A runtime that logs forever without
/// ever answering must not hold the agent (and its memory floor) for good.
const STARTUP_CEILING_FACTOR: u32 = 4;

fn startup_poll() -> Duration {
    Duration::from_millis(if cfg!(test) { 100 } else { 1000 })
}

/// Wait for a freshly spawned model to report exact health.
///
/// Gives up only when the process exits, when nothing moved for the whole
/// `window` (each sign of progress restarts it), or after
/// `STARTUP_CEILING_FACTOR` windows in total. A fixed deadline cost 45 minutes
/// on 2026-11-07, waiting on an engine that had faulted three minutes in.
pub fn wait_started(model: &Model, process: &Proc, log: &Path, window: Duration) -> Result<()> {
    if model.serve.port.is_none() {
        return Ok(());
    }
    watch_startup(
        window,
        &mut SystemClock,
        || StartupLook {
            ready: governor::readiness(model, model.probe_timeout()) == Readiness::Ready,
            alive: process.owned_alive(),
            progress: StartupProgress::sample(model, log),
        },
        startup_poll(),
    )
}

/// One look at a starting model.
struct StartupLook {
    ready: bool,
    alive: bool,
    progress: StartupProgress,
}

/// Time for [`watch_startup`]: the real clock, or a fake one in tests, so the
/// window rules are checked exactly rather than against a loaded machine's
/// wall clock.
trait StartupClock {
    fn now(&mut self) -> std::time::Instant;
    fn pause(&mut self, poll: Duration);
}

struct SystemClock;

impl StartupClock for SystemClock {
    fn now(&mut self) -> std::time::Instant {
        std::time::Instant::now()
    }
    fn pause(&mut self, poll: Duration) {
        std::thread::sleep(poll);
    }
}

/// The rules of [`wait_started`], over any clock and any way of looking.
fn watch_startup(
    window: Duration,
    clock: &mut dyn StartupClock,
    mut look: impl FnMut() -> StartupLook,
    poll: Duration,
) -> Result<()> {
    let started = clock.now();
    let ceiling = window.saturating_mul(STARTUP_CEILING_FACTOR);
    let mut last_progress = started;
    let mut seen: Option<StartupProgress> = None;
    loop {
        let current = look();
        if current.ready {
            return Ok(());
        }
        let now = clock.now();
        if !current.alive {
            bail!(
                "exited during startup after {}s",
                now.saturating_duration_since(started).as_secs()
            );
        }
        let progress = current.progress;
        seen = Some(match seen {
            None => progress,
            Some(before) => {
                if progress.advanced_from(&before) {
                    last_progress = now;
                }
                StartupProgress {
                    port_open: before.port_open || progress.port_open,
                    log_len: before.log_len.max(progress.log_len),
                    metrics: before.metrics || progress.metrics,
                }
            }
        });
        if now.saturating_duration_since(last_progress) >= window {
            bail!(
                "made no startup progress (port, log, metrics) for {}s",
                window.as_secs()
            );
        }
        if now.saturating_duration_since(started) >= ceiling {
            bail!(
                "was still not ready after {}s despite progress",
                ceiling.as_secs()
            );
        }
        clock.pause(poll);
    }
}

pub fn deactivate(root: &Path) -> Result<()> {
    let _lock = TransitionLock::acquire(root)?;
    let mut state = State::load_checked(root)?;
    let ids: Vec<String> = state.procs.keys().cloned().collect();
    for id in ids {
        let process = state.procs[&id].clone();
        // A docker model's container is stopped by id even if its CLI died.
        if process.owned_alive() || process.container_id.is_some() {
            println!("  stopping {id} (pid {})", process.pid);
            proc::stop_owned(&process)?;
        }
        state.procs.remove(&id);
        state.save(root)?;
    }
    state.desired_scene = None;
    state.active_scene = None;
    state.transition_scene = None;
    state.last_error = None;
    state.save(root)?;
    println!("all Lumbridge Computer-managed models stopped; no Scene is desired");
    Ok(())
}

/// Diff the target Scene against what is actually serving. Pure: it decides
/// *what* would change, never *whether* the change is allowed.
///
/// M3: `on_demand` models are never started by a transition, and a model the
/// lifecycle holds asleep (or mid-wake) is present, not missing. An owned
/// process outside the Scene is stopped even when it does not answer health,
/// so a sleeping model is never orphaned by a switch.
fn plan_transition(registry: &Registry, scene: &Scene, state: &State) -> Plan {
    let target: HashSet<&str> = scene.models.iter().map(String::as_str).collect();
    let running = governor::running_ids(registry);
    let mut stop: Vec<String> = running
        .iter()
        .filter(|id| !target.contains(id.as_str()))
        .cloned()
        .collect();
    for (id, process) in &state.procs {
        if !target.contains(id.as_str())
            && !stop.contains(id)
            && registry.models.contains_key(id)
            && process.owned_alive()
        {
            stop.push(id.clone());
        }
    }
    let held = |id: &str| {
        state
            .procs
            .get(id)
            .is_some_and(|p| p.owned_alive() && p.model_state().in_transition_or_asleep())
    };
    let mut start: Vec<String> = scene
        .models
        .iter()
        .filter(|id| scene.mode_of(id).mode != crate::config::Mode::OnDemand)
        .filter(|id| !held(id))
        .filter(|id| !governor::is_running(&registry.models[*id]))
        .cloned()
        .collect();
    let listed = scene
        .activation
        .as_ref()
        .and_then(|activation| activation.order.as_deref())
        .map(|order| order == "listed")
        .unwrap_or(false);
    if !listed {
        start.sort_by(|a, b| {
            registry.models[a]
                .footprint_gb
                .partial_cmp(&registry.models[b].footprint_gb)
                .unwrap()
        });
    }
    // A Scene's `budget_gb` is the unified pool's ceiling: the only pool a Spark has.
    let pool = governor::Pool::unified();
    Plan {
        scene: scene.metadata.name.clone(),
        budget_gb: pool.effective_budget_gb(
            pool.scene_budget_gb(scene.budget_gb),
            pool.sense().ok().map(|memory| memory.total_gb),
            pool.reserved_gb(registry),
        ),
        stop,
        start,
    }
}

/// Everything that must hold before the first process is signalled. Runs to
/// completion with nothing mutated, so a rejection here leaves the currently
/// active Scene exactly as it was.
fn preflight(registry: &Registry, scene: &Scene, plan: &Plan, state: &State) -> Result<()> {
    for id in &scene.models {
        if governor::is_running(&registry.models[id]) {
            let process = state.procs.get(id).with_context(|| {
                format!(
                    "'{id}' is already serving but is not identity-owned by Lumbridge Computer; adopt the active Scene first"
                )
            })?;
            if !process.owned_alive() {
                bail!(
                    "ownership record for serving model '{id}' is stale; adopt the active Scene first"
                );
            }
        }
    }

    for id in &plan.stop {
        let process = state.procs.get(id).with_context(|| {
            format!(
                "'{id}' is serving but is not identity-owned by Lumbridge Computer; adopt the active Scene before switching"
            )
        })?;
        if !process.owned_alive() {
            bail!("ownership record for serving model '{id}' is stale; refusing to signal its pid");
        }
    }
    for id in &plan.start {
        let model = &registry.models[id];
        if let Some(port) = model.serve.port {
            // A model in `plan.stop` may currently hold this port. Every stop runs before
            // any start, so that is a handoff, not a conflict. Without this exemption every
            // same-port swap is rejected — including brain -> brain-laguna/brain-gemma,
            // which share :8001 by design because the alias downstream agents call must
            // survive a weight swap.
            let freed_by_stop = plan.stop.iter().any(|stopping| {
                registry
                    .models
                    .get(stopping)
                    .and_then(|stopped| stopped.serve.port)
                    == Some(port)
            });
            if !freed_by_stop && governor::port_open(port) && !governor::is_running(model) {
                bail!("port {port} is occupied by a different model; refusing to start '{id}'");
            }
        }
    }
    Ok(())
}

/// `scene activate --dry-run` as data rather than as printed lines: the same
/// validation, the same plan, nothing written. Callers that need the plan
/// programmatically use this instead of scraping stdout.
pub fn plan(root: &Path, name: &str) -> Result<Plan> {
    let _lock = TransitionLock::acquire(root)?;
    plan_read_only(root, name)
}

/// `plan` without the transition lock, for `lumbridge-computer apply --dry-run`: taking the lock
/// writes under the root and, held at the wrong moment, would make the live
/// supervisor's own transition fail. Reads only; a plan computed mid-transition may
/// be stale, which the caller reports via `State::transition_scene`.
pub fn plan_read_only(root: &Path, name: &str) -> Result<Plan> {
    let registry = Registry::load(root)?;
    let scene = find_scene(root, name)?;
    validate_scene(&registry, &scene)?;
    let mut state = State::load_checked(root)?;
    state.procs.retain(|_, process| process.owned_alive());
    let plan = plan_transition(&registry, &scene, &state);
    preflight(&registry, &scene, &plan, &state)?;
    Ok(plan)
}

fn activate_once(root: &Path, name: &str, dry_run: bool) -> Result<()> {
    let registry = Registry::load(root)?;
    let scene = find_scene(root, name)?;
    validate_scene(&registry, &scene)?;
    let mut state = State::load_checked(root)?;
    state.procs.retain(|_, process| process.owned_alive());
    let plan = plan_transition(&registry, &scene, &state);

    println!("activate '{name}' (budget {:.0} GB)", plan.budget_gb);
    println!("  stop : {}", format_ids(&plan.stop));
    println!("  start: {}", format_ids(&plan.start));

    preflight(&registry, &scene, &plan, &state)?;

    if dry_run {
        println!("  dry run: validation passed; nothing changed");
        return Ok(());
    }

    state.transition_scene = Some(name.to_string());
    state.save(root)?;

    for id in &plan.stop {
        let process = state.procs[id].clone();
        println!("  stopping {id} (pid {})", process.pid);
        proc::stop_owned(&process)?;
        state.procs.remove(id);
        state.save(root)?;
    }

    // Committed memory per pool: a model only competes with models in its own pool,
    // each counted at its admission figure (declared, or measured when opted in).
    // A sleeping model counts at its sleeping footprint (M3).
    let mut committed: BTreeMap<governor::PoolId, f64> = BTreeMap::new();
    for id in &scene.models {
        let model = &registry.models[id];
        *committed.entry(model.pool).or_default() +=
            crate::states::counted_gb(root, &registry, &state, id);
    }
    let wait_healthy = scene
        .activation
        .as_ref()
        .and_then(|activation| activation.wait_healthy)
        .unwrap_or(true);

    for id in &plan.start {
        let model: &Model = &registry.models[id];
        // Re-read the pool before every start rather than once per activation: each model
        // that comes up consumes real memory, and its true appetite is only knowable after
        // it has allocated. A footprint that was optimistic shows up here, on the next
        // model, instead of taking the box down.
        let pool = model.pool.pool();
        let committed_in_pool = committed.get(&model.pool).copied().unwrap_or(0.0);
        let available_gb = pool
            .sense()
            .ok()
            .map(|m| pool.model_available_gb(&registry, m.available_gb));
        // `admission: measured` budgets max(declared, learned); default is declared.
        let add_gb = crate::footprint::admission_gb(root, &registry, id);
        // The observed ceiling asks for what the model really takes, not its typed figure.
        let observed_gb = crate::footprint::observed_gb(root, &registry, id);
        if !pool.can_admit_observed(
            add_gb,
            observed_gb,
            committed_in_pool,
            plan.budget_gb,
            available_gb,
        ) {
            match available_gb {
                Some(available) => bail!(
                    "'{id}' ({:.0} GB, {:.1} GB measured) refused: {:.0} GB committed against a \
                     {:.0} GB budget, {:.1} GB actually available, {:.0} GB margin required",
                    add_gb,
                    observed_gb,
                    committed_in_pool,
                    plan.budget_gb,
                    available,
                    pool.safety_margin_gb
                ),
                None => bail!(
                    "'{id}' would exceed the {:.0} GB Scene budget with the safety margin",
                    plan.budget_gb
                ),
            }
        }
        let pid = proc::spawn(root, id, model, crate::scope::mode(registry.scopes))?;
        state.seq += 1;
        let mut process = Proc::capture(pid, state.seq, model.serve.port)
            .with_context(|| format!("capturing ownership for newly started '{id}'"))?;
        process.container_id = proc::await_container(root, id, model, &process);
        state.procs.insert(id.clone(), process.clone());
        state.save(root)?;
        *committed.entry(model.pool).or_default() += add_gb;
        println!("  started {id} (pid {pid}); waiting for exact health");
        let window = model
            .supervision
            .as_ref()
            .map_or(health_timeout(), |policy| {
                Duration::from_secs(policy.startup_timeout_sec())
            });
        if wait_healthy {
            if let Err(error) = wait_started(model, &process, &proc::log_path(root, id), window) {
                let _ = proc::stop_owned(&process);
                state.procs.remove(id);
                state.save(root)?;
                bail!("'{id}' did not report its exact health marker: it {error:#}");
            }
        }
    }

    let unhealthy: Vec<String> = scene
        .models
        .iter()
        .filter(|id| crate::states::expects_healthy(&scene, &state, id))
        .filter(|id| !governor::is_running(&registry.models[*id]))
        .cloned()
        .collect();
    if !unhealthy.is_empty() {
        bail!(
            "Scene '{name}' is incomplete; exact health failed for: {}",
            unhealthy.join(", ")
        );
    }
    Ok(())
}

pub(crate) fn validate_scene(registry: &Registry, scene: &Scene) -> Result<()> {
    let missing: Vec<&String> = scene
        .models
        .iter()
        .filter(|id| !registry.models.contains_key(*id))
        .collect();
    if !missing.is_empty() {
        bail!(
            "Scene '{}' references unknown model id(s): {}",
            scene.metadata.name,
            missing
                .iter()
                .map(|id| id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    scene.check_ports(registry)?;
    crate::states::validate_modes(registry, scene)?;
    let pool = governor::Pool::unified();
    let budget = pool.scene_budget_gb(scene.budget_gb);
    // Without `modes:` this is the plain sum of footprints; with them, sleeping
    // models count at their sleeping footprint plus room for one awake (M3).
    let footprint = crate::states::scene_need_gb(registry, scene, &pool);
    if !pool.fits(footprint, budget) {
        bail!(
            "Scene '{}' needs {:.1} GB including safety margin, above its {:.1} GB budget",
            scene.metadata.name,
            footprint + pool.safety_margin_gb,
            budget
        );
    }
    Ok(())
}

fn health_timeout() -> Duration {
    Duration::from_secs(if cfg!(test) { 5 } else { 900 })
}

fn format_ids(ids: &[String]) -> String {
    if ids.is_empty() {
        "(none)".to_string()
    } else {
        ids.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::NoInFlight;
    use crate::mem;
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_root() -> (PathBuf, u16) {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("lumbridge-computer-{unique}"));
        fs::create_dir_all(root.join("registry")).unwrap();
        fs::create_dir_all(root.join("scenes")).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let bad_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let bad_port = bad_listener.local_addr().unwrap().port();
        drop(bad_listener);
        let registry = format!(
            "apiVersion: lumbridge/v1\nmodels:\n  fake:\n    name: Fake\n    footprint_gb: 1\n    health: http://localhost:{port}/\n    supervision:\n      restart: always\n    serve:\n      kind: exec\n      port: {port}\n      command: [/usr/bin/python3, -m, http.server, '{port}', --bind, 127.0.0.1]\n  bad:\n    name: Bad\n    footprint_gb: 2\n    health: http://localhost:{bad_port}/\n    serve:\n      kind: exec\n      port: {bad_port}\n      command: [/usr/bin/false]\n"
        );
        fs::write(root.join("registry/models.yaml"), registry).unwrap();
        fs::write(
            root.join("scenes/test.scene.yaml"),
            "apiVersion: lumbridge/v1\nmetadata:\n  name: test\n  version: 1\nmodels: [fake]\nbudget_gb: 100\n",
        )
        .unwrap();
        (root, port)
    }

    #[test]
    fn low_available_memory_refuses_activation_without_starting_a_model() {
        let _memory = mem::test_memory(128.0, 5.0);
        let (root, port) = fixture_root();
        let error = activate(&root, "test", false).unwrap_err();
        assert!(format!("{error:#}").contains("actually available"));
        assert!(State::load_checked(&root).unwrap().procs.is_empty());
        assert!(!governor::port_open(port));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn activation_persists_and_deactivation_stops_owned_process() {
        let _memory = mem::test_memory(128.0, 100.0);
        let (root, _port) = fixture_root();
        activate(&root, "test", false).unwrap();
        let state = State::load_checked(&root).unwrap();
        assert_eq!(state.desired_scene.as_deref(), Some("test"));
        assert_eq!(state.active_scene.as_deref(), Some("test"));
        assert!(state.procs["fake"].owned_alive());
        deactivate(&root).unwrap();
        assert!(State::load_checked(&root).unwrap().procs.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_fresh_agent_resumes_by_adopting_the_running_model_not_restarting_it() {
        // What a new agent does on its first start after the old one stopped:
        // the same root, the same state file, a live model it did not spawn.
        let _memory = mem::test_memory(128.0, 100.0);
        let (root, _port) = fixture_root();
        activate(&root, "test", false).unwrap();
        let before = State::load_checked(&root).unwrap().procs["fake"].clone();
        resume(&root).unwrap();
        let after = State::load_checked(&root).unwrap();
        let adopted = &after.procs["fake"];
        assert_eq!(adopted.pid, before.pid, "the running process was kept");
        assert_eq!(adopted.start_time_ticks, before.start_time_ticks);
        assert_eq!(adopted.seq, before.seq, "nothing was launched");
        assert!(adopted.owned_alive());
        assert_eq!(after.active_scene.as_deref(), Some("test"));
        deactivate(&root).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn supervision_restarts_only_the_failed_owned_model() {
        let _memory = mem::test_memory(128.0, 100.0);
        let (root, _port) = fixture_root();
        activate(&root, "test", false).unwrap();
        let before = State::load_checked(&root).unwrap();
        let old = before.procs["fake"].clone();
        proc::stop_owned(&old).unwrap();

        assert_eq!(
            restart_supervised_model(&root, "fake", Trigger::Exited, &NoInFlight).unwrap(),
            Restart::Restarted
        );
        let after = State::load_checked(&root).unwrap();
        assert_ne!(after.procs["fake"].pid, old.pid);
        assert!(after.procs["fake"].owned_alive());
        assert_eq!(after.desired_scene.as_deref(), Some("test"));
        assert_eq!(after.active_scene.as_deref(), Some("test"));

        deactivate(&root).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_liveness_trigger_restarts_a_model_that_still_passes_readiness() {
        let _memory = mem::test_memory(128.0, 100.0);
        let (root, _port) = fixture_root();
        let registry_path = root.join("registry/models.yaml");
        let contents = fs::read_to_string(&registry_path).unwrap();
        fs::write(
            &registry_path,
            contents.replacen(
                "      restart: always\n",
                "      restart: always\n      liveness:\n        kind: vllm-metrics\n",
                1,
            ),
        )
        .unwrap();
        activate(&root, "test", false).unwrap();
        let old = State::load_checked(&root).unwrap().procs["fake"].clone();
        let registry = Registry::load(&root).unwrap();
        assert!(registry.models["fake"]
            .supervision
            .as_ref()
            .unwrap()
            .liveness
            .is_some());
        assert!(governor::is_running(&registry.models["fake"]));

        // A readiness-only trigger on a healthy model is a no-op, as before.
        assert!(matches!(
            restart_supervised_model(
                &root,
                "fake",
                Trigger::Unready(Readiness::Timeout),
                &NoInFlight
            )
            .unwrap(),
            Restart::Skipped(_)
        ));
        assert_eq!(
            State::load_checked(&root).unwrap().procs["fake"].pid,
            old.pid
        );

        // A liveness trigger is not undone by readiness passing. This server has
        // no vLLM metrics, which keeps the agent's stalled verdict.
        let stalled = Trigger::Stalled {
            tokens: 42.0,
            in_flight: 1,
        };
        assert_eq!(
            restart_supervised_model(&root, "fake", stalled, &NoInFlight).unwrap(),
            Restart::Restarted
        );
        let after = State::load_checked(&root).unwrap();
        assert_ne!(after.procs["fake"].pid, old.pid);
        assert!(after.procs["fake"].owned_alive());

        deactivate(&root).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn supervision_refuses_to_replace_an_unowned_process() {
        let (root, port) = fixture_root();
        let registry = Registry::load(&root).unwrap();
        let model = &registry.models["fake"];
        let pid = proc::spawn(&root, "fake", model, crate::config::ScopesMode::Off).unwrap();
        assert!(governor::wait_healthy(model, Duration::from_secs(5)));

        // Make exact health fail while the unowned server continues to hold the
        // port. Supervision must not infer ownership from a port number.
        let registry_path = root.join("registry/models.yaml");
        let contents = fs::read_to_string(&registry_path).unwrap();
        fs::write(
            &registry_path,
            contents.replace(
                &format!("health: http://localhost:{port}/"),
                &format!(
                    "health: http://localhost:{port}/\n    health_contains: marker-that-is-not-served"
                ),
            ),
        )
        .unwrap();
        fs::create_dir_all(root.join(".compute")).unwrap();
        State {
            desired_scene: Some("test".to_string()),
            active_scene: Some("test".to_string()),
            last_known_good_scene: Some("test".to_string()),
            ..State::default()
        }
        .save(&root)
        .unwrap();

        let error = restart_supervised_model(
            &root,
            "fake",
            Trigger::Unready(Readiness::Mismatch),
            &NoInFlight,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("unowned process"));
        assert!(governor::port_open(port));

        let owned = Proc::capture(pid, 1, Some(port)).unwrap();
        proc::stop_owned(&owned).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    /// Two models sharing one port, distinguishable by `health_contains` — the
    /// brain/brain-laguna shape. Each serves its own directory so the health probe
    /// can tell which one is actually up.
    fn same_port_root() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("lumbridge-computer-swap-{unique}"));
        fs::create_dir_all(root.join("registry")).unwrap();
        fs::create_dir_all(root.join("scenes")).unwrap();
        let dir_a = root.join("srv-alpha");
        let dir_b = root.join("srv-beta");
        fs::create_dir_all(&dir_a).unwrap();
        fs::create_dir_all(&dir_b).unwrap();
        fs::write(dir_a.join("alpha-marker.txt"), "a").unwrap();
        fs::write(dir_b.join("beta-marker.txt"), "b").unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let (a, b) = (dir_a.display(), dir_b.display());
        let registry = format!(
            "apiVersion: lumbridge/v1\nmodels:\n  alpha:\n    name: Alpha\n    footprint_gb: 1\n    health: http://localhost:{port}/\n    health_contains: alpha-marker\n    serve:\n      kind: exec\n      port: {port}\n      command: [/usr/bin/python3, -m, http.server, '{port}', --bind, 127.0.0.1, --directory, '{a}']\n  beta:\n    name: Beta\n    footprint_gb: 1\n    health: http://localhost:{port}/\n    health_contains: beta-marker\n    serve:\n      kind: exec\n      port: {port}\n      command: [/usr/bin/python3, -m, http.server, '{port}', --bind, 127.0.0.1, --directory, '{b}']\n"
        );
        fs::write(root.join("registry/models.yaml"), registry).unwrap();
        fs::write(
            root.join("scenes/a.scene.yaml"),
            "apiVersion: lumbridge/v1\nmetadata:\n  name: a\n  version: 1\nmodels: [alpha]\nbudget_gb: 100\n",
        )
        .unwrap();
        fs::write(
            root.join("scenes/b.scene.yaml"),
            "apiVersion: lumbridge/v1\nmetadata:\n  name: b\n  version: 1\nmodels: [beta]\nbudget_gb: 100\n",
        )
        .unwrap();
        root
    }

    #[test]
    fn same_port_swap_is_allowed_when_the_occupant_is_being_stopped() {
        let _memory = mem::test_memory(128.0, 100.0);
        let root = same_port_root();
        activate(&root, "a", false).unwrap();
        assert!(State::load_checked(&root)
            .unwrap()
            .procs
            .contains_key("alpha"));

        // Regression: this used to fail with "port N is occupied by a different model;
        // refusing to start 'beta'". The pre-flight port check ran over `plan.start`
        // without exempting ports released by `plan.stop`, so every same-port swap was
        // rejected even though stops precede starts.
        activate(&root, "b", false).unwrap();

        let state = State::load_checked(&root).unwrap();
        assert_eq!(state.active_scene.as_deref(), Some("b"));
        assert!(state.procs.contains_key("beta"));
        assert!(!state.procs.contains_key("alpha"));

        deactivate(&root).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn already_serving_scene_requires_explicit_adoption() {
        let (root, port) = fixture_root();
        let registry = Registry::load(&root).unwrap();
        let model = &registry.models["fake"];
        let pid = proc::spawn(&root, "fake", model, crate::config::ScopesMode::Off).unwrap();
        assert!(governor::wait_healthy(model, Duration::from_secs(5)));
        let error = activate(&root, "test", true).unwrap_err().to_string();
        assert!(error.contains("adopt the active Scene"));
        let owned = Proc::capture(pid, 1, Some(port)).unwrap();
        proc::stop_owned(&owned).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn preflight_rejection_preserves_healthy_active_scene() {
        let _memory = mem::test_memory(128.0, 100.0);
        let (root, _port) = fixture_root();
        activate(&root, "test", false).unwrap();
        let before = State::load_checked(&root).unwrap();
        let pid = before.procs["fake"].pid;
        fs::write(
            root.join("scenes/invalid.scene.yaml"),
            "apiVersion: lumbridge/v1\nmetadata:\n  name: invalid\n  version: 1\nmodels: [missing]\nbudget_gb: 100\n",
        )
        .unwrap();
        assert!(activate(&root, "invalid", false).is_err());
        let after = State::load_checked(&root).unwrap();
        assert_eq!(after.active_scene.as_deref(), Some("test"));
        assert_eq!(after.desired_scene.as_deref(), Some("test"));
        assert_eq!(after.procs["fake"].pid, pid);
        assert!(after.procs["fake"].owned_alive());
        deactivate(&root).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_activation_keeps_healthy_partial_scene_without_rollback_target() {
        let _memory = mem::test_memory(128.0, 100.0);
        let (root, port) = fixture_root();
        fs::write(
            root.join("scenes/failing.scene.yaml"),
            "apiVersion: lumbridge/v1\nmetadata:\n  name: failing\n  version: 1\nmodels: [fake, bad]\nbudget_gb: 100\n",
        )
        .unwrap();
        assert!(activate(&root, "failing", false).is_err());
        let state = State::load_checked(&root).unwrap();
        assert!(state.procs.contains_key("fake"));
        assert!(!state.procs.contains_key("bad"));
        assert!(state.active_scene.is_none());
        assert!(governor::port_open(port));
        deactivate(&root).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    /// A model whose command is `argv` (YAML flow list), on a fresh free port.
    fn startup_model(argv: &str) -> (PathBuf, Model, u16) {
        let (root, port) = fixture_root();
        let model: Model = serde_yaml::from_str(&format!(
            "name: Starting\nfootprint_gb: 1\nhealth: http://localhost:{port}/\nserve:\n  kind: exec\n  port: {port}\n  command: {}\n",
            argv.replace("PORT", &port.to_string())
        ))
        .unwrap();
        (root, model, port)
    }

    fn spawn_owned(root: &Path, model: &Model) -> Proc {
        let pid = proc::spawn(root, "starting", model, crate::config::ScopesMode::Off).unwrap();
        Proc::capture(pid, 1, model.serve.port).unwrap()
    }

    #[test]
    fn startup_progress_extends_the_deadline() {
        // Loads for ~2.4 s, logging every 0.3 s, then serves. A fixed 0.8 s
        // deadline would have killed it; the log growth keeps resetting it.
        let (root, model, _port) = startup_model(
            "[/usr/bin/python3, -u, -c, \"import time, http.server as h\\nfor i in range(8):\\n    print('loading', i)\\n    time.sleep(0.3)\\nh.ThreadingHTTPServer(('127.0.0.1', PORT), h.SimpleHTTPRequestHandler).serve_forever()\"]",
        );
        let process = spawn_owned(&root, &model);
        let started = std::time::Instant::now();
        let log = proc::log_path(&root, "starting");
        wait_started(&model, &process, &log, Duration::from_millis(800)).unwrap();
        assert!(started.elapsed() > Duration::from_secs(2));
        proc::stop_owned(&process).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    /// A clock that only moves when the watcher pauses.
    struct FakeClock {
        now: std::time::Instant,
        start: std::time::Instant,
    }

    impl FakeClock {
        fn new() -> FakeClock {
            let now = std::time::Instant::now();
            FakeClock { now, start: now }
        }
        fn elapsed(&self) -> Duration {
            self.now - self.start
        }
    }

    impl StartupClock for FakeClock {
        fn now(&mut self) -> std::time::Instant {
            self.now
        }
        fn pause(&mut self, poll: Duration) {
            self.now += poll;
        }
    }

    fn still(log_len: u64) -> StartupLook {
        StartupLook {
            ready: false,
            alive: true,
            progress: StartupProgress {
                port_open: false,
                log_len,
                metrics: false,
            },
        }
    }

    #[test]
    fn startup_without_progress_gives_up_after_exactly_one_window() {
        // The rule itself, on a clock that a loaded machine can't stretch: a
        // live process that never moves is given up on once a full window has
        // passed since the start, not before and not a poll later.
        let poll = Duration::from_millis(100);
        let window = Duration::from_millis(800);
        let mut clock = FakeClock::new();
        let error = watch_startup(window, &mut clock, || still(42), poll)
            .unwrap_err()
            .to_string();
        assert!(error.contains("no startup progress"), "{error}");
        assert!(clock.elapsed() >= window, "{:?}", clock.elapsed());
        assert!(clock.elapsed() < window + poll, "{:?}", clock.elapsed());
    }

    #[test]
    fn each_sign_of_progress_restarts_the_window_up_to_the_ceiling() {
        let poll = Duration::from_millis(100);
        let window = Duration::from_millis(800);
        // The log grows once, 0.5 s in: the window restarts from there.
        let mut clock = FakeClock::new();
        let mut looks = 0u32;
        let error = watch_startup(
            window,
            &mut clock,
            || {
                looks += 1;
                still(if looks > 5 { 2 } else { 1 })
            },
            poll,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("no startup progress"), "{error}");
        assert!(clock.elapsed() >= Duration::from_millis(500) + window);
        assert!(clock.elapsed() < Duration::from_millis(500) + window + poll);

        // Progress on every look never trips the window, only the ceiling.
        let mut clock = FakeClock::new();
        let mut len = 0;
        let error = watch_startup(
            window,
            &mut clock,
            || {
                len += 1;
                still(len)
            },
            poll,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("despite progress"), "{error}");
        assert!(clock.elapsed() >= window * STARTUP_CEILING_FACTOR);
    }

    #[test]
    fn a_real_process_that_makes_no_progress_is_given_up_on_by_the_window_rule() {
        // End to end with a real, silent process. Port 0 can never accept, so
        // no other test's server can look like this model opening its port
        // (the 2026-11-08 flake). Wall time only has to be ordered: it waited
        // at least the window, and the message says the window rule fired, not
        // the ceiling or an exit.
        let (root, _port) = fixture_root();
        let model: Model = serde_yaml::from_str(
            "name: Silent\nfootprint_gb: 1\nhealth: http://localhost:0/\nserve:\n  kind: exec\n  port: 0\n  command: [/usr/bin/sleep, '30']\n",
        )
        .unwrap();
        let process = spawn_owned(&root, &model);
        let window = Duration::from_millis(800);
        let started = std::time::Instant::now();
        let log = proc::log_path(&root, "starting");
        let error = wait_started(&model, &process, &log, window)
            .unwrap_err()
            .to_string();
        assert!(error.contains("no startup progress"), "{error}");
        assert!(started.elapsed() >= window);
        assert!(process.owned_alive(), "a live process, given up on");
        proc::stop_owned(&process).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_process_that_dies_during_startup_fails_at_once_not_at_the_deadline() {
        // 2026-11-07 02:45: the new engine faulted three minutes in and the agent
        // waited out the rest of a 45-minute deadline.
        let (root, model, _port) =
            startup_model("[/bin/sh, -c, 'echo starting; sleep 0.3; exit 3']");
        let process = spawn_owned(&root, &model);
        let started = std::time::Instant::now();
        let log = proc::log_path(&root, "starting");
        let error = wait_started(&model, &process, &log, Duration::from_secs(600))
            .unwrap_err()
            .to_string();
        assert!(error.contains("exited during startup"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(5));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_respawn_keeps_the_previous_log() {
        let (root, model, _port) = startup_model("[/bin/sh, -c, 'echo first run; exec sleep 30']");
        let first = spawn_owned(&root, &model);
        let log = proc::log_path(&root, "starting");
        for _ in 0..50 {
            if fs::read_to_string(&log)
                .unwrap_or_default()
                .contains("first run")
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = proc::stop_owned(&first);
        let second = spawn_owned(&root, &model);
        let kept = fs::read_to_string(log.with_extension("log.1")).unwrap();
        assert!(kept.contains("first run"));
        let _ = proc::stop_owned(&second);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_restart_recheck_stands_down_for_a_busy_engine() {
        let engine = health::testing::FakeEngine::start();
        engine.set(|s| {
            s.running = 1;
            s.waiting = 2;
            s.models_delay = Duration::from_millis(1_200);
        });
        let model = engine.model("probe_timeout_ms: 500\n");
        let skipped = recheck(&model, None, Trigger::Unready(Readiness::Timeout), 0);
        assert_eq!(
            skipped.as_deref(),
            Some("busy: running=1 waiting=2 gateway=0")
        );

        // Idle and still slow: go ahead.
        engine.set(|s| {
            s.running = 0;
            s.waiting = 0;
        });
        assert_eq!(
            recheck(&model, None, Trigger::Unready(Readiness::Timeout), 0),
            None
        );
        // Gateway work counts as busy too.
        assert!(recheck(&model, None, Trigger::Unready(Readiness::Timeout), 4).is_some());
    }

    #[test]
    fn the_restart_recheck_reprobes_an_idle_hang_and_respects_new_work() {
        let engine = health::testing::FakeEngine::start();
        engine.set(|s| s.chat_ok = false);
        let model = engine
            .model("liveness:\n    kind: vllm-metrics\n    chat_probe:\n      model: brain\n");
        assert_eq!(recheck(&model, None, Trigger::ProbeFailed, 0), None);
        assert_eq!(engine.chats(), 1);

        // Work arrived before the lock was taken: not a hang, and no probe.
        engine.set(|s| s.running = 1);
        assert!(recheck(&model, None, Trigger::ProbeFailed, 0).is_some());
        assert_eq!(engine.chats(), 1);

        // Idle again and the chat answers: recovered.
        engine.set(|s| {
            s.running = 0;
            s.chat_ok = true;
        });
        assert_eq!(
            recheck(&model, None, Trigger::ProbeFailed, 0).as_deref(),
            Some("it answered a 1-token chat")
        );
    }

    #[test]
    fn activation_never_starts_an_on_demand_model_and_does_not_wait_for_it() {
        let _memory = mem::test_memory(128.0, 100.0);
        let (root, port) = fixture_root();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let lazy_port = listener.local_addr().unwrap().port();
        drop(listener);
        let registry_path = root.join("registry/models.yaml");
        let mut registry = fs::read_to_string(&registry_path).unwrap();
        registry.push_str(&format!(
            "  lazy:\n    name: Lazy\n    footprint_gb: 1\n    health: http://localhost:{lazy_port}/\n    serve:\n      kind: exec\n      port: {lazy_port}\n      command: [/usr/bin/python3, -m, http.server, '{lazy_port}', --bind, 127.0.0.1]\n"
        ));
        fs::write(&registry_path, registry).unwrap();
        fs::write(
            root.join("scenes/lazy.scene.yaml"),
            "apiVersion: lumbridge/v1\nmetadata:\n  name: lazy\n  version: 1\nmodels: [fake, lazy]\nmodes:\n  lazy: {mode: on_demand}\nbudget_gb: 100\n",
        )
        .unwrap();
        let plan = plan(&root, "lazy").unwrap();
        assert_eq!(plan.start, ["fake"]);
        activate(&root, "lazy", false).unwrap();
        let state = State::load_checked(&root).unwrap();
        assert_eq!(state.active_scene.as_deref(), Some("lazy"));
        assert!(state.procs.contains_key("fake"));
        assert!(!state.procs.contains_key("lazy"));
        assert!(governor::port_open(port));
        assert!(!governor::port_open(lazy_port));
        deactivate(&root).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checked_state_load_rejects_malformed_yaml() {
        let (root, _port) = fixture_root();
        fs::create_dir_all(root.join(".compute")).unwrap();
        fs::write(root.join(".compute/state.yaml"), "desired_scene: [").unwrap();
        assert!(State::load_checked(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn state_write_replaces_complete_yaml_atomically() {
        let (root, _port) = fixture_root();
        let state = State {
            desired_scene: Some("test".to_string()),
            ..State::default()
        };
        state.save(&root).unwrap();
        let loaded = State::load_checked(&root).unwrap();
        assert_eq!(loaded.desired_scene.as_deref(), Some("test"));
        fs::remove_dir_all(root).unwrap();
    }
}
