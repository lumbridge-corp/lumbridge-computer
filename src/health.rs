//! Supervision health rules: when a supervised model counts as failed, and when
//! it is merely busy.
//!
//! The rules, in the order a due check applies them:
//!
//! 1. **Process exit is fatal.** An owned process that is gone is restarted at
//!    once (still subject to backoff), without waiting for readiness to notice.
//! 2. **Stall.** Work in flight (engine `running + waiting`, or the gateway's
//!    in-flight count) while the token counters stay flat for `stall_sec` is a
//!    hang and counts as a failed check, whatever readiness says.
//! 3. **Readiness.** A refused connection or the wrong model on the port counts
//!    as a failed check. A *timeout* while work is in flight is busy, not failed:
//!    it neither counts nor resets the failure streak.
//! 4. **Chat probe.** Only when the engine is idle by every source and its
//!    counters have not moved. One failure is only suspicious; a second in a row
//!    confirms the hang. After a failed probe the load is re-read at once, so a
//!    request that arrived while the probe waited voids the result.
//!
//! Each rule comes from a real failure; the replay test below walks through one day of them.

use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

use crate::config::{find_scene, ChatProbe, Model, Registry, RestartPolicy, Supervision};
use crate::governor::{self, Readiness};
use crate::{lifecycle, proc};

/// Engine work as supervision sees it. Mirrors the fields of lane B's
/// `engines::EngineLoad` that supervision needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Load {
    pub running: u32,
    pub waiting: u32,
    /// Prompt plus generation tokens since the engine started; `None` when the
    /// engine reports load but not token counters.
    pub tokens: Option<f64>,
}

impl Load {
    pub fn in_flight(&self) -> u32 {
        self.running.saturating_add(self.waiting)
    }
}

impl From<governor::EngineProgress> for Load {
    fn from(progress: governor::EngineProgress) -> Self {
        let count = |value: f64| value.max(0.0).round().min(f64::from(u32::MAX)) as u32;
        Load {
            running: count(progress.running),
            waiting: count(progress.waiting),
            tokens: Some(progress.tokens),
        }
    }
}

/// The single source of engine load for supervision. `engine_progress` reads
/// through the `engines` module and also carries the token counters the stall
/// rule needs, which `engines::scrape`'s `EngineLoad` does not guarantee.
pub fn engine_load(model: &Model) -> Option<Load> {
    governor::engine_progress(model).map(Load::from)
}

/// Requests a front door (lane C's gateway) has accepted for a model and not
/// yet finished. Above zero, the model is busy.
pub trait InFlight: Send + Sync {
    fn get(&self, model_id: &str) -> u32;
}

/// Used until the gateway provides real counts.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoInFlight;

impl InFlight for NoInFlight {
    fn get(&self, _model_id: &str) -> u32 {
        0
    }
}

/// Why supervision restarts a model. Passed to the restart so it can re-check
/// the same condition under the transition lock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Trigger {
    Exited,
    Unready(Readiness),
    Stalled { tokens: f64, in_flight: u32 },
    ProbeFailed,
}

impl Trigger {
    pub fn describe(&self, policy: &Supervision) -> String {
        match self {
            Trigger::Exited => "process exited".to_string(),
            Trigger::Unready(readiness) => format!(
                "not ready: {} (timeout {} ms)",
                readiness.describe(),
                policy.probe_timeout().as_millis()
            ),
            Trigger::Stalled { tokens, in_flight } => format!(
                "stalled: {in_flight} request(s) in flight but the token count has stayed at {tokens} for at least {}s",
                policy.liveness.as_ref().map_or(0, |l| l.stall_sec())
            ),
            Trigger::ProbeFailed => format!(
                "looks idle but failed {REQUIRED_PROBE_FAILURES} 1-token chats in a row"
            ),
        }
    }
}

/// A chat probe confirms a hang only after this many consecutive failures
/// while idle. `failure_threshold` does not stack on top of it.
pub const REQUIRED_PROBE_FAILURES: u32 = 2;

/// What one due check concluded.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Healthy,
    /// Readiness timed out while work was in flight. Not counted either way.
    Busy {
        running: u32,
        waiting: u32,
        gateway: u32,
    },
    /// One idle chat probe failed; the next check probes again.
    Suspect,
    /// A chat probe failed but work arrived while it ran. Not counted.
    ProbeVoided {
        running: u32,
        waiting: u32,
        gateway: u32,
    },
    /// Counts toward `failure_threshold`.
    Failed(Trigger),
    /// Restart now (subject to backoff): the process exited, or the chat
    /// probe failed twice in a row while idle.
    Confirmed(Trigger),
}

impl Verdict {
    fn kind(&self) -> &'static str {
        match self {
            Verdict::Healthy => "healthy",
            Verdict::Busy { .. } => "busy",
            Verdict::Suspect => "suspect",
            Verdict::ProbeVoided { .. } => "probe-voided",
            Verdict::Failed(Trigger::Exited) | Verdict::Confirmed(Trigger::Exited) => "exited",
            Verdict::Failed(Trigger::Unready(_)) | Verdict::Confirmed(Trigger::Unready(_)) => {
                "unready"
            }
            Verdict::Failed(Trigger::Stalled { .. })
            | Verdict::Confirmed(Trigger::Stalled { .. }) => "stalled",
            Verdict::Failed(Trigger::ProbeFailed) | Verdict::Confirmed(Trigger::ProbeFailed) => {
                "probe-failed"
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub verdict: Verdict,
    pub restart: Option<Trigger>,
}

/// Everything a due check may ask about one model. The agent uses
/// `LiveProbes`; tests script the answers.
pub trait Probes {
    /// True only on positive evidence that the owned process is gone.
    fn exited(&mut self) -> bool;
    fn readiness(&mut self) -> Readiness;
    fn load(&mut self) -> Option<Load>;
    fn in_flight(&mut self) -> u32;
    fn chat(&mut self) -> bool;
}

pub struct LiveProbes<'a> {
    pub id: &'a str,
    pub model: &'a Model,
    pub process: Option<&'a proc::Proc>,
    pub chat: Option<&'a ChatProbe>,
    pub in_flight: &'a dyn InFlight,
}

impl Probes for LiveProbes<'_> {
    fn exited(&mut self) -> bool {
        self.process.is_some_and(proc::Proc::exited)
    }
    fn readiness(&mut self) -> Readiness {
        governor::readiness(self.model, self.model.probe_timeout())
    }
    fn load(&mut self) -> Option<Load> {
        engine_load(self.model)
    }
    fn in_flight(&mut self) -> u32 {
        self.in_flight.get(self.id)
    }
    fn chat(&mut self) -> bool {
        self.chat
            .is_none_or(|chat| governor::chat_probe(self.model, chat))
    }
}

/// How soon a restart that admission refused for memory is asked again.
pub const ADMISSION_RETRY_SEC: u64 = 30;
/// A model that stays refused is logged this often, not on every retry.
pub const ADMISSION_LOG_EVERY_SEC: u64 = 300;

#[derive(Debug, Default)]
struct ModelTrack {
    consecutive_failures: u32,
    next_check_at: u64,
    retry_at: u64,
    next_backoff_sec: u64,
    /// Last token count seen while work was in flight, and since when.
    progress: Option<Progress>,
    /// Chat probe bookkeeping: when it last ran, the token count at the last
    /// due check (no probe while real traffic moves it), and the current run
    /// of consecutive idle failures.
    last_probe_at: u64,
    last_tokens: Option<f64>,
    probe_failures: u32,
    /// When a memory refusal was last logged (see `complete_refused_restart`).
    refusal_logged_at: Option<u64>,
    log: LogLimit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Progress {
    tokens: f64,
    since: u64,
}

/// Per-model log rate limit: a line prints when the verdict kind changes, or
/// at most every `LOG_REPEAT_SEC` while it stays the same.
#[derive(Debug, Default)]
struct LogLimit {
    kind: Option<&'static str>,
    printed_at: u64,
    suppressed: u32,
}

const LOG_REPEAT_SEC: u64 = 300;

impl LogLimit {
    /// `Some(suppressed)` when a line should print now.
    fn admit(&mut self, kind: &'static str, now: u64) -> Option<u32> {
        if self.kind == Some(kind) && now.saturating_sub(self.printed_at) < LOG_REPEAT_SEC {
            self.suppressed += 1;
            return None;
        }
        let suppressed = if self.kind == Some(kind) {
            self.suppressed
        } else {
            0
        };
        self.kind = Some(kind);
        self.printed_at = now;
        self.suppressed = 0;
        Some(suppressed)
    }
}

/// The stall rule. Stalled only when work is in flight and the token count has
/// not changed for `stall_sec`. Idle resets the window; unreadable counters
/// neither build nor reset it (unknown is not wedged).
fn progress_stalled(
    progress: &mut Option<Progress>,
    tokens: Option<f64>,
    in_flight: u32,
    now: u64,
    stall_sec: u64,
) -> bool {
    let Some(tokens) = tokens else {
        return false;
    };
    if in_flight == 0 {
        *progress = None;
        return false;
    }
    match progress {
        // Any change counts as progress, including a drop when the runtime
        // restarted its counters.
        Some(seen) if seen.tokens == tokens => now.saturating_sub(seen.since) >= stall_sec,
        _ => {
            *progress = Some(Progress { tokens, since: now });
            false
        }
    }
}

pub struct SupervisionTracker {
    models: BTreeMap<String, ModelTrack>,
    in_flight: Arc<dyn InFlight>,
    /// The agent's lifecycle (M3): a model it holds mid-transition is skipped
    /// even before that transition reaches the state file.
    lifecycle: Option<Arc<dyn crate::states::Lifecycle>>,
}

impl Default for SupervisionTracker {
    fn default() -> Self {
        Self::new(Arc::new(NoInFlight))
    }
}

impl SupervisionTracker {
    pub fn new(in_flight: Arc<dyn InFlight>) -> Self {
        Self {
            models: BTreeMap::new(),
            in_flight,
            lifecycle: None,
        }
    }

    pub fn with_lifecycle(mut self, lifecycle: Arc<dyn crate::states::Lifecycle>) -> Self {
        self.lifecycle = Some(lifecycle);
        self
    }

    /// One health check, if one is due. Probes run only when a check is due,
    /// never on every agent tick.
    pub fn evaluate(
        &mut self,
        id: &str,
        policy: &Supervision,
        probes: &mut impl Probes,
        now: u64,
    ) -> Option<Decision> {
        let track = self.models.entry(id.to_string()).or_default();
        if now < track.next_check_at {
            return None;
        }
        let verdict = if probes.exited() {
            track.progress = None;
            track.last_tokens = None;
            Verdict::Confirmed(Trigger::Exited)
        } else {
            judge(track, policy, probes, now)
        };
        let restart = observe(track, policy, &verdict, now);
        Some(Decision { verdict, restart })
    }

    pub fn tick(&mut self, root: &Path, now: u64) -> Result<()> {
        let registry = Registry::load(root)?;
        let state = proc::State::load_checked(root)?;
        let Some(desired) = state.desired_scene.as_deref() else {
            self.models.clear();
            return Ok(());
        };
        let scene = find_scene(root, desired)?;
        let supervised = supervised_model_ids(&registry, &scene.models);
        self.models.retain(|id, _| supervised.contains(id));
        let in_flight = Arc::clone(&self.in_flight);

        for id in supervised {
            let Some(model) = registry.models.get(&id) else {
                continue;
            };
            let Some(policy) = model.supervision.as_ref() else {
                continue;
            };
            if policy.restart != RestartPolicy::Always {
                continue;
            }
            // M3: asleep, draining, waking or starting under the lifecycle, or a
            // non-resident model that is simply stopped. Not a failure: skip it,
            // and forget its track so liveness starts fresh once it is ready.
            let held = self
                .lifecycle
                .as_ref()
                .is_some_and(|lifecycle| lifecycle.state(&id).in_transition_or_asleep());
            if held
                || crate::states::supervision_paused(scene.mode_of(&id).mode, state.procs.get(&id))
            {
                self.models.remove(&id);
                continue;
            }
            let mut probes = LiveProbes {
                id: &id,
                model,
                process: state.procs.get(&id),
                chat: policy
                    .liveness
                    .as_ref()
                    .and_then(|liveness| liveness.chat_probe.as_ref()),
                in_flight: in_flight.as_ref(),
            };
            let Some(decision) = self.evaluate(&id, policy, &mut probes, now) else {
                continue;
            };
            self.log_decision(&id, policy, &decision.verdict, now);
            let Some(trigger) = decision.restart else {
                continue;
            };

            // While a model waits for memory this fires every `ADMISSION_RETRY_SEC`;
            // the refusal itself is logged at a slower pace, so stay quiet here too.
            let waiting_for_memory = self
                .models
                .get(&id)
                .is_some_and(|track| track.refusal_logged_at.is_some());
            if !waiting_for_memory {
                eprintln!(
                    "agent: supervised model '{id}' {}; restarting only that model",
                    trigger.describe(policy)
                );
            }
            let attempt_started = Instant::now();
            let result =
                lifecycle::restart_supervised_model(root, &id, trigger, in_flight.as_ref());
            let finished_at = now.saturating_add(attempt_started.elapsed().as_secs());
            let refused = result.as_ref().err().is_some_and(|error| {
                error
                    .downcast_ref::<lifecycle::AdmissionRefused>()
                    .is_some()
            });
            if refused {
                let quiet = !self.complete_refused_restart(&id, policy, finished_at);
                if quiet {
                    continue;
                }
            } else {
                self.complete_restart(&id, policy, finished_at, result.is_ok());
            }
            match result {
                Ok(lifecycle::Restart::Restarted) => {
                    eprintln!("agent: supervised model '{id}' recovered")
                }
                Ok(lifecycle::Restart::Skipped(reason)) => {
                    eprintln!("agent: supervised model '{id}' not restarted: {reason}")
                }
                Err(error) if refused => {
                    eprintln!(
                        "agent: supervised model '{id}' is waiting for memory: {error:#}; asking again every {} seconds",
                        ADMISSION_RETRY_SEC.min(policy.backoff_sec())
                    );
                }
                Err(error) => {
                    let retry_in = self.models[&id].retry_at.saturating_sub(finished_at);
                    eprintln!(
                        "agent: supervised model '{id}' restart failed: {error:#}; next attempt in no less than {retry_in} seconds"
                    );
                }
            }
        }
        Ok(())
    }

    fn log_decision(&mut self, id: &str, policy: &Supervision, verdict: &Verdict, now: u64) {
        let Some(track) = self.models.get_mut(id) else {
            return;
        };
        if *verdict == Verdict::Healthy && track.log.kind.is_none_or(|kind| kind == "healthy") {
            return;
        }
        let Some(suppressed) = track.log.admit(verdict.kind(), now) else {
            return;
        };
        let timeout_ms = policy.probe_timeout().as_millis();
        let line = match verdict {
            Verdict::Healthy => "healthy again".to_string(),
            Verdict::Busy {
                running,
                waiting,
                gateway,
            } => format!(
                "busy: running={running} waiting={waiting} gateway={gateway}; readiness took over {timeout_ms} ms; not counting it"
            ),
            Verdict::Suspect => format!(
                "looks idle and did not answer a 1-token chat; probing again before it counts ({}/{REQUIRED_PROBE_FAILURES})",
                track.probe_failures
            ),
            Verdict::ProbeVoided {
                running,
                waiting,
                gateway,
            } => format!(
                "chat probe failed but work arrived meanwhile (running={running} waiting={waiting} gateway={gateway}); not counting it"
            ),
            Verdict::Failed(trigger) => format!(
                "{}; failed check {}/{}",
                trigger.describe(policy),
                track.consecutive_failures,
                policy.failure_threshold()
            ),
            Verdict::Confirmed(trigger) => trigger.describe(policy),
        };
        let suppressed = if suppressed > 0 {
            format!(" ({suppressed} similar lines suppressed)")
        } else {
            String::new()
        };
        eprintln!("agent: supervised model '{id}' {line}{suppressed}");
    }

    /// A restart that admission refused for memory (`lifecycle::AdmissionRefused`).
    ///
    /// Not a failed start: no process was spawned, so there is no crash loop to damp.
    /// The retry comes after `ADMISSION_RETRY_SEC` (or the policy's own backoff when
    /// that is shorter) and the exponential backoff is left where it was, so a model
    /// that also crashes keeps its crash history. On 2026-11-10 the shared backoff
    /// kept an embedder down for 50 minutes while the pool was 0.2 GB short and only
    /// being asked every 10 minutes.
    ///
    /// Returns whether this refusal should be logged: the first one, then one every
    /// `ADMISSION_LOG_EVERY_SEC`, so a pool that stays full does not fill the journal.
    pub fn complete_refused_restart(&mut self, id: &str, policy: &Supervision, now: u64) -> bool {
        let track = self.models.entry(id.to_string()).or_default();
        track.next_check_at = now.saturating_add(policy.check_interval_sec());
        track.progress = None;
        track.last_tokens = None;
        track.probe_failures = 0;
        track.last_probe_at = now;
        track.retry_at = now.saturating_add(ADMISSION_RETRY_SEC.min(policy.backoff_sec()));
        // As after a failed start: the first check after the wait may retry.
        track.consecutive_failures = policy.failure_threshold();
        let log = track
            .refusal_logged_at
            .is_none_or(|at| now.saturating_sub(at) >= ADMISSION_LOG_EVERY_SEC);
        if log {
            track.refusal_logged_at = Some(now);
        }
        log
    }

    pub fn complete_restart(&mut self, id: &str, policy: &Supervision, now: u64, success: bool) {
        let track = self.models.entry(id.to_string()).or_default();
        track.next_check_at = now.saturating_add(policy.check_interval_sec());
        // Whatever ran before is gone or replaced; its counters mean nothing now.
        track.progress = None;
        track.last_tokens = None;
        track.probe_failures = 0;
        track.last_probe_at = now;
        track.refusal_logged_at = None;
        if success {
            track.consecutive_failures = 0;
            track.retry_at = 0;
            track.next_backoff_sec = policy.backoff_sec();
            return;
        }

        let backoff = track.next_backoff_sec.max(policy.backoff_sec());
        track.retry_at = now.saturating_add(backoff);
        track.next_backoff_sec = backoff.saturating_mul(2).min(policy.max_backoff_sec());
        // Keep the counter at the threshold so the first check after backoff may
        // retry, but a healthy probe at any point still resets it.
        track.consecutive_failures = policy.failure_threshold();
    }
}

/// Rules 2–4 for a model whose process has not exited.
fn judge(
    track: &mut ModelTrack,
    policy: &Supervision,
    probes: &mut impl Probes,
    now: u64,
) -> Verdict {
    let readiness = probes.readiness();
    let liveness = policy.liveness.as_ref();
    // Load is read when liveness wants it, or to tell busy from failed.
    let load = if liveness.is_some() || readiness != Readiness::Ready {
        probes.load()
    } else {
        None
    };
    let gateway = probes.in_flight();
    let (running, waiting) = load.map_or((0, 0), |load| (load.running, load.waiting));
    // The gateway's requests are the engine's too: take the larger, never the sum.
    let in_flight = (running.saturating_add(waiting)).max(gateway);
    let tokens = load.and_then(|load| load.tokens);

    match liveness {
        Some(liveness)
            if progress_stalled(
                &mut track.progress,
                tokens,
                in_flight,
                now,
                liveness.stall_sec(),
            ) =>
        {
            track.last_tokens = tokens;
            let tokens = track.progress.map_or(0.0, |seen| seen.tokens);
            return Verdict::Failed(Trigger::Stalled { tokens, in_flight });
        }
        Some(_) => {}
        None => track.progress = None,
    }

    match readiness {
        Readiness::Ready => {}
        Readiness::Timeout if in_flight > 0 => {
            track.last_tokens = tokens;
            return Verdict::Busy {
                running,
                waiting,
                gateway,
            };
        }
        failed => {
            // A model that is down has no meaningful progress.
            track.progress = None;
            track.last_tokens = None;
            track.probe_failures = 0;
            return Verdict::Failed(Trigger::Unready(failed));
        }
    }

    let Some(chat) = liveness.and_then(|liveness| liveness.chat_probe.as_ref()) else {
        track.last_tokens = tokens;
        return Verdict::Healthy;
    };
    // Without counters there is no telling idle from busy, so no probe.
    let Some(tokens) = tokens else {
        track.last_tokens = None;
        return Verdict::Healthy;
    };
    let quiet = in_flight == 0 && track.last_tokens == Some(tokens);
    track.last_tokens = Some(tokens);
    if !quiet {
        // Real traffic is moving the counters (or the stall rule owns it).
        track.probe_failures = 0;
        return Verdict::Healthy;
    }
    let due =
        track.probe_failures > 0 || now.saturating_sub(track.last_probe_at) >= chat.interval_sec();
    if !due {
        return Verdict::Healthy;
    }
    track.last_probe_at = now;
    if probes.chat() {
        track.probe_failures = 0;
        return Verdict::Healthy;
    }

    // Re-check before blaming the engine: work that arrived while the probe
    // waited explains a slow answer.
    let after = probes.load();
    let gateway = probes.in_flight();
    let (running, waiting) = after.map_or((0, 0), |load| (load.running, load.waiting));
    let tokens_after = after.and_then(|load| load.tokens);
    let moved = tokens_after.is_some_and(|now_tokens| now_tokens != tokens);
    if running.saturating_add(waiting).max(gateway) > 0 || moved {
        track.last_tokens = tokens_after.or(Some(tokens));
        return Verdict::ProbeVoided {
            running,
            waiting,
            gateway,
        };
    }
    track.probe_failures += 1;
    if track.probe_failures >= REQUIRED_PROBE_FAILURES {
        Verdict::Confirmed(Trigger::ProbeFailed)
    } else {
        Verdict::Suspect
    }
}

/// Turn a verdict into the failure streak and, maybe, a restart.
fn observe(
    track: &mut ModelTrack,
    policy: &Supervision,
    verdict: &Verdict,
    now: u64,
) -> Option<Trigger> {
    track.next_check_at = now.saturating_add(policy.check_interval_sec());
    match verdict {
        Verdict::Healthy => {
            track.consecutive_failures = 0;
            track.retry_at = 0;
            track.next_backoff_sec = policy.backoff_sec();
            None
        }
        Verdict::Busy { .. } | Verdict::Suspect | Verdict::ProbeVoided { .. } => None,
        Verdict::Failed(trigger) => {
            track.consecutive_failures = track.consecutive_failures.saturating_add(1);
            (track.consecutive_failures >= policy.failure_threshold() && now >= track.retry_at)
                .then_some(*trigger)
        }
        Verdict::Confirmed(trigger) => {
            track.consecutive_failures = track
                .consecutive_failures
                .saturating_add(1)
                .max(policy.failure_threshold());
            (now >= track.retry_at).then_some(*trigger)
        }
    }
}

pub fn supervised_model_ids(registry: &Registry, scene_models: &[String]) -> BTreeSet<String> {
    scene_models
        .iter()
        .filter_map(|id| {
            registry
                .models
                .get(id)
                .filter(|model| model.supervision.is_some())
                .map(|_| id.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    type OnChat = Box<dyn FnMut(&mut Option<Load>, &mut u32)>;

    /// A scripted model. Each probe reads the current field values; a test
    /// changes them between checks (or from inside `on_chat`).
    struct Fake {
        exited: bool,
        readiness: Readiness,
        load: Option<Load>,
        gateway: u32,
        chat_ok: bool,
        /// Called while the chat probe "runs", to simulate work arriving.
        on_chat: Option<OnChat>,
        chats: u32,
        loads: u32,
    }

    impl Fake {
        fn ready(load: Option<Load>) -> Self {
            Fake {
                exited: false,
                readiness: Readiness::Ready,
                load,
                gateway: 0,
                chat_ok: true,
                on_chat: None,
                chats: 0,
                loads: 0,
            }
        }
    }

    impl Probes for Fake {
        fn exited(&mut self) -> bool {
            self.exited
        }
        fn readiness(&mut self) -> Readiness {
            self.readiness
        }
        fn load(&mut self) -> Option<Load> {
            self.loads += 1;
            self.load
        }
        fn in_flight(&mut self) -> u32 {
            self.gateway
        }
        fn chat(&mut self) -> bool {
            self.chats += 1;
            if let Some(on_chat) = self.on_chat.as_mut() {
                on_chat(&mut self.load, &mut self.gateway);
            }
            self.chat_ok
        }
    }

    fn load(running: u32, waiting: u32, tokens: f64) -> Option<Load> {
        Some(Load {
            running,
            waiting,
            tokens: Some(tokens),
        })
    }

    fn parse(yaml: &str) -> Supervision {
        serde_yaml::from_str(yaml).unwrap()
    }

    fn policy() -> Supervision {
        parse("restart: always\ncheck_interval_sec: 10\nfailure_threshold: 3\nbackoff_sec: 60\nmax_backoff_sec: 240\nstartup_timeout_sec: 5\n")
    }

    /// A 35B vLLM brain's policy, without the chat probe.
    fn brain_policy() -> Supervision {
        parse("restart: always\ncheck_interval_sec: 15\nfailure_threshold: 2\nbackoff_sec: 60\nmax_backoff_sec: 900\nstartup_timeout_sec: 2700\nliveness:\n  kind: vllm-metrics\n  stall_sec: 300\n")
    }

    /// A 35B vLLM brain's policy, chat probe included.
    fn brain_policy_with_chat_probe() -> Supervision {
        parse("restart: always\ncheck_interval_sec: 15\nfailure_threshold: 2\nbackoff_sec: 60\nmax_backoff_sec: 900\nstartup_timeout_sec: 2700\nliveness:\n  kind: vllm-metrics\n  stall_sec: 300\n  chat_probe:\n    interval_sec: 120\n    timeout_sec: 30\n")
    }

    /// Run due checks every `step` seconds until `until`, letting `script`
    /// set the fake's state first. Returns the first restart.
    fn first_restart(
        tracker: &mut SupervisionTracker,
        policy: &Supervision,
        fake: &mut Fake,
        until: u64,
        step: u64,
        mut script: impl FnMut(u64, &mut Fake),
    ) -> Option<(u64, Trigger)> {
        (0..=until).step_by(step as usize).find_map(|now| {
            script(now, fake);
            tracker
                .evaluate("brain", policy, fake, now)
                .and_then(|decision| decision.restart)
                .map(|trigger| (now, trigger))
        })
    }

    #[test]
    fn a_slow_readiness_probe_on_a_busy_engine_is_never_counted() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy();
        // Readiness times out for two hours while the engine works through a
        // queue and keeps counting tokens.
        let mut fake = Fake::ready(None);
        fake.readiness = Readiness::Timeout;
        let restart = first_restart(&mut tracker, &policy, &mut fake, 7200, 15, |now, fake| {
            fake.load = load(2, 1, (now * 40) as f64);
        });
        assert_eq!(restart, None);
        let decision = tracker.evaluate("brain", &policy, &mut fake, 7215).unwrap();
        assert_eq!(
            decision.verdict,
            Verdict::Busy {
                running: 2,
                waiting: 1,
                gateway: 0
            }
        );
    }

    #[test]
    fn busy_by_gateway_count_alone_is_still_busy() {
        // vLLM exposes no load (say, metrics unreadable), but the gateway holds
        // three requests for the model: a readiness timeout is busy.
        let mut tracker = SupervisionTracker::default();
        let policy = policy();
        let mut fake = Fake::ready(None);
        fake.readiness = Readiness::Timeout;
        fake.gateway = 3;
        let restart = first_restart(&mut tracker, &policy, &mut fake, 3600, 10, |_, _| {});
        assert_eq!(restart, None);
    }

    #[test]
    fn a_slow_readiness_probe_on_an_idle_engine_still_counts() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy();
        let mut fake = Fake::ready(load(0, 0, 500.0));
        fake.readiness = Readiness::Timeout;
        let restart = first_restart(&mut tracker, &policy, &mut fake, 600, 15, |_, _| {});
        assert_eq!(
            restart,
            Some((15, Trigger::Unready(Readiness::Timeout))),
            "two consecutive idle timeouts reach failure_threshold 2"
        );
    }

    #[test]
    fn busy_does_not_excuse_a_closed_port_or_the_wrong_model() {
        for readiness in [Readiness::Refused, Readiness::Mismatch] {
            let mut tracker = SupervisionTracker::default();
            let policy = brain_policy();
            let mut fake = Fake::ready(None);
            fake.readiness = readiness;
            fake.gateway = 4;
            let restart = first_restart(&mut tracker, &policy, &mut fake, 600, 15, |_, _| {});
            assert_eq!(restart, Some((15, Trigger::Unready(readiness))));
        }
    }

    #[test]
    fn a_dead_process_is_restarted_at_once_with_backoff() {
        let mut tracker = SupervisionTracker::default();
        let policy = policy();
        let mut fake = Fake::ready(None);
        fake.exited = true;
        // failure_threshold is 3, but an exit does not wait for readiness.
        let decision = tracker.evaluate("voice", &policy, &mut fake, 0).unwrap();
        assert_eq!(decision.verdict, Verdict::Confirmed(Trigger::Exited));
        assert_eq!(decision.restart, Some(Trigger::Exited));

        // A failed restart backs off; the dead process is not retried before.
        tracker.complete_restart("voice", &policy, 0, false);
        for now in [10, 20, 30, 40, 50] {
            let decision = tracker.evaluate("voice", &policy, &mut fake, now).unwrap();
            assert_eq!(decision.restart, None, "retried inside backoff at {now}");
        }
        assert_eq!(
            tracker
                .evaluate("voice", &policy, &mut fake, 60)
                .unwrap()
                .restart,
            Some(Trigger::Exited)
        );
    }

    #[test]
    fn stalled_tokens_with_requests_in_flight_are_restarted() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy();
        // The 2026-11-05 11:02:48 freeze: two requests running, counters
        // frozen, /v1/models fine.
        let mut fake = Fake::ready(load(2, 0, 172_714.0));
        let restart = first_restart(&mut tracker, &policy, &mut fake, 7200, 15, |_, _| {});
        // Stalled from t=300 (first failure); the second at 315 restarts it.
        assert_eq!(
            restart,
            Some((
                315,
                Trigger::Stalled {
                    tokens: 172_714.0,
                    in_flight: 2
                }
            ))
        );

        // The restart resets the window: the new engine starts its own clock.
        tracker.complete_restart("brain", &policy, 400, true);
        fake.load = load(1, 0, 0.0);
        assert_eq!(
            tracker
                .evaluate("brain", &policy, &mut fake, 415)
                .unwrap()
                .verdict,
            Verdict::Healthy
        );
    }

    #[test]
    fn a_stall_is_caught_even_while_readiness_times_out() {
        // Busy excuses a slow probe only while tokens move.
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy();
        let mut fake = Fake::ready(load(2, 0, 9_000.0));
        fake.readiness = Readiness::Timeout;
        let restart = first_restart(&mut tracker, &policy, &mut fake, 7200, 15, |_, _| {});
        assert!(matches!(restart, Some((315, Trigger::Stalled { .. }))));
    }

    #[test]
    fn gateway_requests_against_frozen_counters_are_a_stall() {
        // 2026-11-06 shape behind a gateway: the engine reports 0 running with
        // frozen counters while the gateway holds two hung client requests.
        // In-flight must not shield it from both the probe and the stall rule.
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy_with_chat_probe();
        let mut fake = Fake::ready(load(0, 0, 9_000.0));
        fake.gateway = 2;
        fake.chat_ok = false;
        let restart = first_restart(&mut tracker, &policy, &mut fake, 7200, 15, |_, _| {});
        assert!(matches!(
            restart,
            Some((315, Trigger::Stalled { in_flight: 2, .. }))
        ));
        assert_eq!(fake.chats, 0, "never probed while the gateway held work");
    }

    #[test]
    fn a_busy_engine_that_keeps_counting_tokens_is_never_restarted() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy();
        // Tokens arrive in bursts: a cold 128K-token prefill moves neither vLLM
        // counter until its first token, so each burst follows up to 285 s of
        // silence.
        let mut fake = Fake::ready(None);
        let restart = first_restart(&mut tracker, &policy, &mut fake, 7200, 15, |now, fake| {
            fake.load = load(4, 0, (now / 300 * 300) as f64);
        });
        assert_eq!(restart, None);
    }

    #[test]
    fn an_idle_engine_is_never_restarted() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy();
        let mut fake = Fake::ready(load(0, 0, 9_000.0));
        let restart = first_restart(&mut tracker, &policy, &mut fake, 36_000, 15, |_, _| {});
        assert_eq!(restart, None);
        assert_eq!(tracker.models["brain"].progress, None);
    }

    #[test]
    fn going_idle_or_unreadable_metrics_never_build_toward_a_stall() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy();
        let mut fake = Fake::ready(None);
        let restart = first_restart(&mut tracker, &policy, &mut fake, 900, 15, |now, fake| {
            fake.load = match now {
                0..=290 => load(1, 0, 500.0),
                291..=305 => load(0, 0, 500.0),
                306..=495 => load(1, 0, 500.0),
                // /metrics vanishes: unknown is not wedged, and it does not
                // reset the window either.
                _ => None,
            };
        });
        assert_eq!(restart, None);
    }

    #[test]
    fn idle_with_the_chat_probe_failing_twice_is_restarted() {
        // 2026-11-07 06:13–10:37: counters idle, every chat hangs.
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy_with_chat_probe();
        let mut fake = Fake::ready(load(0, 0, 9_000.0));
        fake.chat_ok = false;
        let restart = first_restart(&mut tracker, &policy, &mut fake, 7200, 15, |_, _| {});
        // First probe due at 120 (suspect), retried at 135 (confirmed).
        assert_eq!(restart, Some((135, Trigger::ProbeFailed)));
        assert_eq!(fake.chats, 2);
    }

    #[test]
    fn idle_with_one_chat_failure_then_success_is_not_restarted() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy_with_chat_probe();
        let mut fake = Fake::ready(load(0, 0, 7.0));
        fake.chat_ok = false;
        let mut failed_once = false;
        let restart = first_restart(&mut tracker, &policy, &mut fake, 3600, 15, |_, fake| {
            // Fail only the first probe.
            if fake.chats == 1 && !failed_once {
                failed_once = true;
                fake.chat_ok = true;
            }
        });
        assert_eq!(restart, None);
        assert!(fake.chats > 2);
        assert_eq!(tracker.models["brain"].probe_failures, 0);
    }

    #[test]
    fn a_request_arriving_during_a_failed_probe_voids_it() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy_with_chat_probe();
        let mut fake = Fake::ready(load(0, 0, 7.0));
        fake.chat_ok = false;
        // Every time the probe runs, a real request lands and occupies the
        // engine, so the probe's timeout is explained by load.
        fake.on_chat = Some(Box::new(|load, _gateway| {
            if let Some(load) = load.as_mut() {
                load.running = 1;
            }
        }));
        let restart = first_restart(&mut tracker, &policy, &mut fake, 3600, 15, |_, fake| {
            // The request finishes before the next check without new tokens.
            if let Some(load) = fake.load.as_mut() {
                load.running = 0;
            }
        });
        assert_eq!(restart, None);
        assert!(fake.chats >= 2);
        assert_eq!(tracker.models["brain"].probe_failures, 0);
    }

    #[test]
    fn a_gateway_request_arriving_during_a_failed_probe_voids_it() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy_with_chat_probe();
        let mut fake = Fake::ready(load(0, 0, 7.0));
        fake.chat_ok = false;
        fake.on_chat = Some(Box::new(|_load, gateway| *gateway = 1));
        let mut decisions = Vec::new();
        for now in (0..=150).step_by(15) {
            fake.gateway = 0;
            if let Some(decision) = tracker.evaluate("brain", &policy, &mut fake, now) {
                decisions.push(decision.verdict);
            }
        }
        assert!(decisions.contains(&Verdict::ProbeVoided {
            running: 0,
            waiting: 0,
            gateway: 1
        }));
        assert!(!decisions
            .iter()
            .any(|verdict| matches!(verdict, Verdict::Confirmed(_) | Verdict::Suspect)));
    }

    #[test]
    fn an_idle_healthy_engine_is_probed_rarely_and_never_restarted() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy_with_chat_probe();
        let mut fake = Fake::ready(load(0, 0, 500.0));
        let restart = first_restart(&mut tracker, &policy, &mut fake, 3600, 15, |_, _| {});
        assert_eq!(restart, None);
        assert!(
            fake.chats <= 31,
            "about one probe per 2 min, got {}",
            fake.chats
        );
    }

    #[test]
    fn real_traffic_skips_the_chat_probe() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy_with_chat_probe();
        let mut fake = Fake::ready(None);
        fake.chat_ok = false;
        let restart = first_restart(&mut tracker, &policy, &mut fake, 3600, 15, |now, fake| {
            // Counters move between checks; idle at sample time.
            fake.load = load(0, 0, 1_000.0 + now as f64 * 3.0);
        });
        assert_eq!(restart, None);
        assert_eq!(fake.chats, 0);
    }

    #[test]
    fn liveness_is_opt_in_and_a_ready_model_is_not_scraped() {
        let mut tracker = SupervisionTracker::default();
        let policy = policy();
        let mut fake = Fake::ready(load(2, 0, 1.0));
        fake.chat_ok = false;
        let restart = first_restart(&mut tracker, &policy, &mut fake, 7200, 10, |_, _| {});
        assert_eq!(restart, None);
        assert_eq!(
            fake.loads, 0,
            "metrics read for a ready model without liveness"
        );
        assert_eq!(
            fake.chats, 0,
            "chat probe ran for a model that did not opt in"
        );
    }

    #[test]
    fn probes_run_only_when_a_check_is_due() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy();
        let mut fake = Fake::ready(load(0, 0, 0.0));
        let checks = (0..60)
            .filter(|&now| tracker.evaluate("brain", &policy, &mut fake, now).is_some())
            .count();
        assert_eq!(checks, 4, "one check per 15 s, not per agent tick");
        assert_eq!(fake.loads, 4);
    }

    #[test]
    fn only_consecutive_failures_trigger_a_restart() {
        let mut tracker = SupervisionTracker::default();
        let policy = policy();
        let mut fake = Fake::ready(None);
        let mut check = |ready: bool, now: u64| {
            fake.readiness = if ready {
                Readiness::Ready
            } else {
                Readiness::Refused
            };
            tracker
                .evaluate("voice", &policy, &mut fake, now)
                .and_then(|decision| decision.restart)
        };
        assert_eq!(check(false, 0), None);
        assert_eq!(check(false, 5), None, "not due: ignored entirely");
        assert_eq!(check(false, 10), None);
        assert_eq!(check(true, 20), None, "healthy resets the streak");
        assert_eq!(check(false, 30), None);
        assert_eq!(check(false, 40), None);
        assert_eq!(check(false, 50), Some(Trigger::Unready(Readiness::Refused)));
    }

    #[test]
    fn busy_holds_the_failure_streak_without_resetting_it() {
        let mut tracker = SupervisionTracker::default();
        let policy = brain_policy();
        let mut fake = Fake::ready(load(0, 0, 1.0));
        fake.readiness = Readiness::Timeout;
        assert_eq!(
            tracker
                .evaluate("brain", &policy, &mut fake, 0)
                .unwrap()
                .restart,
            None
        );
        fake.load = load(1, 0, 2.0);
        assert!(matches!(
            tracker
                .evaluate("brain", &policy, &mut fake, 15)
                .unwrap()
                .verdict,
            Verdict::Busy { .. }
        ));
        assert_eq!(tracker.models["brain"].consecutive_failures, 1);
        fake.load = load(0, 0, 2.0);
        assert_eq!(
            tracker
                .evaluate("brain", &policy, &mut fake, 30)
                .unwrap()
                .restart,
            Some(Trigger::Unready(Readiness::Timeout))
        );
    }

    #[test]
    fn failed_restarts_back_off_exponentially_and_health_resets_them() {
        let mut tracker = SupervisionTracker::default();
        let policy = policy();
        let mut fake = Fake::ready(None);
        fake.readiness = Readiness::Refused;
        let restart_at = |tracker: &mut SupervisionTracker, fake: &mut Fake, now| {
            tracker
                .evaluate("voice", &policy, fake, now)
                .and_then(|decision| decision.restart)
        };
        for now in [0, 10] {
            assert_eq!(restart_at(&mut tracker, &mut fake, now), None);
        }
        assert!(restart_at(&mut tracker, &mut fake, 20).is_some());
        tracker.complete_restart("voice", &policy, 20, false);
        assert_eq!(tracker.models["voice"].retry_at, 80);
        for now in [30, 40, 50, 60, 70] {
            assert_eq!(restart_at(&mut tracker, &mut fake, now), None);
        }
        assert!(restart_at(&mut tracker, &mut fake, 80).is_some());
        tracker.complete_restart("voice", &policy, 80, false);
        assert_eq!(tracker.models["voice"].retry_at, 200);

        fake.readiness = Readiness::Ready;
        assert_eq!(restart_at(&mut tracker, &mut fake, 90), None);
        let track = &tracker.models["voice"];
        assert_eq!(track.consecutive_failures, 0);
        assert_eq!(track.retry_at, 0);
        assert_eq!(track.next_backoff_sec, 60);
    }

    #[test]
    fn a_memory_refusal_retries_soon_and_leaves_the_crash_backoff_alone() {
        let policy = policy();
        let mut tracker = SupervisionTracker::default();
        // One real failed start first: backoff 60 s, next one doubles to 120.
        tracker.complete_restart("voice", &policy, 0, false);
        assert_eq!(tracker.models["voice"].retry_at, 60);
        assert_eq!(tracker.models["voice"].next_backoff_sec, 120);

        // Refused for memory at t=60, 90, 120...: asked again every 30 s, forever,
        // and the crash backoff does not grow.
        for now in [60, 90, 120, 150, 600, 3600] {
            tracker.complete_refused_restart("voice", &policy, now);
            let track = &tracker.models["voice"];
            assert_eq!(track.retry_at, now + ADMISSION_RETRY_SEC);
            assert_eq!(track.next_backoff_sec, 120);
            assert_eq!(track.consecutive_failures, policy.failure_threshold());
        }

        // A start that then really fails still backs off from where it was.
        tracker.complete_restart("voice", &policy, 3700, false);
        assert_eq!(tracker.models["voice"].retry_at, 3700 + 120);
    }

    #[test]
    fn a_standing_memory_refusal_is_logged_once_per_interval() {
        let policy = policy();
        let mut tracker = SupervisionTracker::default();
        assert!(tracker.complete_refused_restart("voice", &policy, 1000));
        for now in [1030, 1060, 1290] {
            assert!(!tracker.complete_refused_restart("voice", &policy, now));
        }
        assert!(tracker.complete_refused_restart("voice", &policy, 1300));
        // A real attempt in between resets it: the next refusal is news again.
        tracker.complete_restart("voice", &policy, 1310, false);
        assert!(tracker.complete_refused_restart("voice", &policy, 1400));
    }

    #[test]
    fn repeated_verdicts_are_rate_limited_and_changes_always_print() {
        let mut limit = LogLimit::default();
        assert_eq!(limit.admit("busy", 0), Some(0));
        for now in (15..300).step_by(15) {
            assert_eq!(limit.admit("busy", now), None);
        }
        assert_eq!(limit.admit("busy", 300), Some(19), "reprinted with a count");
        assert_eq!(
            limit.admit("stalled", 301),
            Some(0),
            "a new kind prints at once"
        );
        assert_eq!(limit.admit("busy", 302), Some(0));
    }

    #[test]
    fn engine_load_keeps_running_and_waiting_apart_for_the_logs() {
        let metrics = "vllm:num_requests_running{engine=\"0\",model_name=\"brain\"} 2.0\n\
vllm:num_requests_waiting{engine=\"0\",model_name=\"brain\"} 1.0\n\
vllm:prompt_tokens_total{engine=\"0\",model_name=\"brain\"} 835681.0\n\
vllm:generation_tokens_total{engine=\"0\",model_name=\"brain\"} 20432.0\n";
        assert_eq!(
            governor::vllm_progress(metrics).map(Load::from),
            Some(Load {
                running: 2,
                waiting: 1,
                tokens: Some(835_681.0 + 20_432.0)
            })
        );
    }

    #[test]
    fn models_without_supervision_are_not_reconciled() {
        let registry: Registry = serde_yaml::from_str(
            "apiVersion: lumbridge/v1\nmodels:\n  plain:\n    name: Plain\n    footprint_gb: 1\n    serve:\n      kind: exec\n      port: 9000\n  voice:\n    name: Voice\n    footprint_gb: 8\n    supervision:\n      restart: always\n    serve:\n      kind: exec\n      port: 8095\n",
        )
        .unwrap();
        let scene_models = vec!["plain".to_string(), "voice".to_string()];
        assert_eq!(
            supervised_model_ids(&registry, &scene_models),
            BTreeSet::from(["voice".to_string()])
        );
    }

    /// Replay of one bad day for a 35B vLLM brain, minute by minute from the
    /// agent journal, kernel log and gateway traffic.
    /// Old rules restarted 7 times and skipped a 4 h hang; the new rules must
    /// restart for each real fault and never for the busy stretches.
    #[test]
    fn replay_of_2026_11_07_restarts_only_for_real_faults() {
        #[derive(Clone, Copy, PartialEq, Debug)]
        enum Phase {
            /// 5-minute vision batches: busy ~15 s, then idle.
            Normal,
            /// Readiness slow while a batch runs (box under swap pressure).
            SlowWhileBusy,
            /// EngineCore died on a GPU fault: API server gone.
            Crashed,
            /// Requests in flight, counters frozen.
            Wedged,
            /// Looks idle, every chat hangs.
            IdleHang,
        }
        let hm = |h: u64, m: u64| h * 3600 + m * 60;
        // Fault windows as observed; a restart ends the fault it hit.
        let windows = [
            (hm(1, 14) + 20, hm(1, 17) + 47, Phase::Crashed),
            (hm(2, 37) + 21, hm(3, 31) + 40, Phase::Wedged),
            (hm(4, 43) + 14, hm(4, 51) + 35, Phase::Wedged),
            (hm(6, 12), hm(10, 41) + 8, Phase::IdleHang),
            (hm(15, 0), hm(16, 0), Phase::SlowWhileBusy),
            (hm(23, 41) + 40, hm(23, 45) + 4, Phase::Crashed),
        ];
        let window_at = |t: u64| {
            windows
                .iter()
                .find(|(start, end, _)| (*start..*end).contains(&t))
                .copied()
        };

        let policy = brain_policy_with_chat_probe();
        let mut tracker = SupervisionTracker::default();
        let mut restarts: Vec<(u64, Phase)> = Vec::new();
        let mut tokens = 0.0;
        let mut now = 0;
        while now < hm(24, 0) {
            let window = window_at(now);
            let phase = window.map_or(Phase::Normal, |(_, _, phase)| phase);
            let batch = now % 300 < 15;
            let mut fake = Fake::ready(None);
            match phase {
                Phase::Normal | Phase::SlowWhileBusy => {
                    if batch {
                        tokens += 400.0;
                    }
                    fake.load = load(u32::from(batch) * 2, 0, tokens);
                    if phase == Phase::SlowWhileBusy && batch {
                        fake.readiness = Readiness::Timeout;
                    }
                }
                Phase::Crashed => {
                    fake.readiness = Readiness::Refused;
                    fake.exited = true;
                }
                Phase::Wedged => fake.load = load(1, 0, tokens),
                Phase::IdleHang => {
                    fake.load = load(0, 0, tokens);
                    fake.chat_ok = false;
                }
            }
            let restart = tracker
                .evaluate("brain", &policy, &mut fake, now)
                .and_then(|decision| decision.restart);
            if restart.is_some() {
                restarts.push((now, phase));
                // A fresh engine after the ~3-minute vLLM start, fault gone.
                let back = (now + 200).max(window.map_or(0, |(_, end, _)| end));
                tracker.complete_restart("brain", &policy, back, true);
                tokens = 0.0;
                now = back;
                continue;
            }
            now += 1;
        }

        assert!(
            restarts
                .iter()
                .all(|(_, phase)| *phase != Phase::Normal && *phase != Phase::SlowWhileBusy),
            "restarted a healthy or busy engine: {restarts:?}"
        );
        let phases: Vec<Phase> = restarts.iter().map(|(_, phase)| *phase).collect();
        assert_eq!(phases.iter().filter(|p| **p == Phase::Crashed).count(), 2);
        assert!(phases.contains(&Phase::Wedged));
        assert!(phases.contains(&Phase::IdleHang));
        // The 06:12 hang is caught within minutes, not after 4 h 24 min.
        let hang = restarts
            .iter()
            .find(|(_, phase)| *phase == Phase::IdleHang)
            .unwrap()
            .0;
        assert!(
            hang - hm(6, 12) <= 300,
            "idle hang caught after {}s",
            hang - hm(6, 12)
        );
    }

    #[test]
    fn a_slow_but_busy_server_is_never_restarted() {
        // A real HTTP engine: /v1/models takes longer than the probe budget
        // while /metrics shows two running requests and moving tokens.
        let engine = testing::FakeEngine::start();
        engine.set(|s| {
            s.running = 2;
            s.models_delay = Duration::from_millis(1_200);
        });
        let model = engine.model("probe_timeout_ms: 500\n  liveness:\n    kind: vllm-metrics\n");
        let policy = model.supervision.clone().unwrap();
        let mut tracker = SupervisionTracker::default();
        for now in [0, 15, 30, 45] {
            let mut probes = LiveProbes {
                id: "fake",
                model: &model,
                process: None,
                chat: None,
                in_flight: &NoInFlight,
            };
            let decision = tracker.evaluate("fake", &policy, &mut probes, now).unwrap();
            assert!(
                matches!(decision.verdict, Verdict::Busy { running: 2, .. }),
                "at {now}: {decision:?}"
            );
            assert_eq!(decision.restart, None);
        }

        // The same slow server with nothing in flight is a failure.
        engine.set(|s| s.running = 0);
        let restart = [60, 75].into_iter().find_map(|now| {
            let mut probes = LiveProbes {
                id: "fake",
                model: &model,
                process: None,
                chat: None,
                in_flight: &NoInFlight,
            };
            tracker.evaluate("fake", &policy, &mut probes, now)?.restart
        });
        assert_eq!(restart, Some(Trigger::Unready(Readiness::Timeout)));
    }

    #[test]
    fn a_real_idle_engine_whose_chats_fail_twice_is_restarted() {
        let engine = testing::FakeEngine::start();
        engine.set(|s| s.chat_ok = false);
        let model = engine.model(
            "liveness:\n    kind: vllm-metrics\n    chat_probe:\n      interval_sec: 30\n      model: brain\n",
        );
        let policy = model.supervision.clone().unwrap();
        let chat = policy.liveness.as_ref().unwrap().chat_probe.clone();
        let mut tracker = SupervisionTracker::default();
        let mut verdicts = Vec::new();
        for now in [0, 15, 30, 45] {
            let mut probes = LiveProbes {
                id: "fake",
                model: &model,
                process: None,
                chat: chat.as_ref(),
                in_flight: &NoInFlight,
            };
            verdicts.push(tracker.evaluate("fake", &policy, &mut probes, now).unwrap());
        }
        // 0 records the counters, 15 is inside the interval, 30 and 45 probe.
        assert_eq!(verdicts[2].verdict, Verdict::Suspect);
        assert_eq!(verdicts[3].restart, Some(Trigger::ProbeFailed));
        assert_eq!(engine.chats(), 2);
    }
}

/// A tiny fake vLLM server for tests: `/metrics`, `/v1/models` (optionally
/// slow) and `/v1/chat/completions` (answers or fails), on 127.0.0.1:0.
#[cfg(test)]
pub mod testing {
    use crate::config::Model;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    pub struct EngineState {
        pub running: u32,
        pub waiting: u32,
        /// Tokens advance by this much on every `/metrics` read while running.
        pub tokens_per_read: f64,
        pub tokens: f64,
        pub models_delay: Duration,
        pub chat_ok: bool,
        pub chats: u32,
    }

    pub struct FakeEngine {
        pub port: u16,
        state: Arc<Mutex<EngineState>>,
    }

    impl FakeEngine {
        pub fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let state = Arc::new(Mutex::new(EngineState {
                running: 0,
                waiting: 0,
                tokens_per_read: 10.0,
                tokens: 100.0,
                models_delay: Duration::ZERO,
                chat_ok: true,
                chats: 0,
            }));
            let shared = Arc::clone(&state);
            thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let state = Arc::clone(&shared);
                    thread::spawn(move || serve(stream, &state));
                }
            });
            FakeEngine { port, state }
        }

        pub fn set(&self, change: impl FnOnce(&mut EngineState)) {
            change(&mut self.state.lock().unwrap());
        }

        pub fn chats(&self) -> u32 {
            self.state.lock().unwrap().chats
        }

        /// A supervised model on this port; `supervision` is extra YAML
        /// indented under the `supervision:` block.
        pub fn model(&self, supervision: &str) -> Model {
            let port = self.port;
            serde_yaml::from_str(&format!(
                "name: Fake\nfootprint_gb: 1\nhealth: http://localhost:{port}/v1/models\nhealth_contains: '\"root\":\"/models/fake\"'\nsupervision:\n  restart: always\n  check_interval_sec: 15\n  failure_threshold: 2\n  {supervision}serve:\n  kind: vllm\n  port: {port}\n  served_name: [brain]\n"
            ))
            .unwrap()
        }
    }

    fn serve(mut stream: TcpStream, state: &Mutex<EngineState>) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        // Headers, then whatever body Content-Length announces.
        while !request.windows(4).any(|w| w == b"\r\n\r\n") {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => return,
                Ok(n) => request.extend_from_slice(&buf[..n]),
            }
        }
        let text = String::from_utf8_lossy(&request).into_owned();
        let head_len = text.find("\r\n\r\n").map_or(0, |i| i + 4);
        let body_len = text
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())?
            })
            .unwrap_or(0);
        while request.len() < head_len + body_len {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => request.extend_from_slice(&buf[..n]),
            }
        }
        let path = text.split_whitespace().nth(1).unwrap_or("/").to_string();
        let (status, body) = match path.as_str() {
            "/metrics" => {
                let mut s = state.lock().unwrap();
                if s.running + s.waiting > 0 {
                    s.tokens += s.tokens_per_read;
                }
                (
                    "200 OK",
                    format!(
                        "vllm:num_requests_running{{engine=\"0\"}} {}\nvllm:num_requests_waiting{{engine=\"0\"}} {}\nvllm:prompt_tokens_total{{engine=\"0\"}} {}\nvllm:generation_tokens_total{{engine=\"0\"}} 0\n",
                        s.running, s.waiting, s.tokens
                    ),
                )
            }
            "/v1/models" => {
                let delay = state.lock().unwrap().models_delay;
                thread::sleep(delay);
                (
                    "200 OK",
                    "{\"data\":[{\"id\":\"brain\",\"root\":\"/models/fake\"}]}".to_string(),
                )
            }
            "/v1/chat/completions" => {
                let mut s = state.lock().unwrap();
                s.chats += 1;
                if s.chat_ok {
                    ("200 OK", "{\"choices\":[{\"text\":\"OK\"}]}".to_string())
                } else {
                    ("500 Internal Server Error", "{}".to_string())
                }
            }
            _ => ("404 Not Found", String::new()),
        };
        let _ = write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    }
}
