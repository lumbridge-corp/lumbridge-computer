//! The agent's idle sleeper: every few seconds, sleep the `sleepable` and
//! `on_demand` models of the desired Scene that are ready, idle and have been
//! for `idle_sec`, and stop `on_demand` models asleep for `stop_after_sec`.
//!
//! Idle means no request in flight at the gateway *and* none the engine itself
//! reports (a client on the model's port directly). The idle clock starts at
//! the latest of: the gateway's last finished request, the model becoming
//! ready, and the last time this sleeper saw it busy.

use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::{Host, Lifecycle, ModelLifecycle, ModelState};
use crate::config::{find_scene, Mode, Registry};
use crate::proc::State;

/// How often the agent runs the sleeper.
pub const TICK: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdleAction {
    Sleep(String),
    Stop(String),
    /// A resident model found asleep (its Scene's mode changed): wake it.
    Wake(String),
}

#[derive(Default)]
pub struct IdleSleeper {
    /// Last time each model was seen busy, or first seen with no clock at all.
    seen: HashMap<String, u64>,
}

impl IdleSleeper {
    /// What is due now. Reads only: the registry, the state file, the host.
    pub fn due(&mut self, root: &std::path::Path, host: &dyn Host) -> Result<Vec<IdleAction>> {
        let state = State::load_checked(root)?;
        let Some(desired) = state.desired_scene.as_deref() else {
            self.seen.clear();
            return Ok(Vec::new());
        };
        let registry = Registry::load(root)?;
        let scene = find_scene(root, desired)?;
        let now = host.now_ms();
        let mut due = Vec::new();
        self.seen.retain(|id, _| scene.models.contains(id));
        for id in &scene.models {
            let Some(model) = registry.models.get(id) else {
                continue;
            };
            let mode = scene.mode_of(id);
            let Some(process) = state.procs.get(id).filter(|p| p.owned_alive()) else {
                self.seen.remove(id);
                continue;
            };
            match (mode.mode, process.model_state()) {
                (Mode::Resident, ModelState::Sleeping { .. }) => {
                    due.push(IdleAction::Wake(id.clone()))
                }
                (Mode::Resident, _) => {}
                (_, ModelState::Ready) => {
                    let Some(idle_sec) = mode.idle_sec() else {
                        continue;
                    };
                    let busy =
                        host.in_flight(id) > 0 || host.engine_in_flight(model).unwrap_or(0) > 0;
                    if busy {
                        self.seen.insert(id.clone(), now);
                        continue;
                    }
                    let since = [
                        host.last_finished_ms(id),
                        process.state_since_ms,
                        self.seen.get(id).copied(),
                    ]
                    .into_iter()
                    .flatten()
                    .max();
                    let Some(since) = since else {
                        // Started by a Scene transition and never routed to: start
                        // the clock now rather than sleep it on first sight.
                        self.seen.insert(id.clone(), now);
                        continue;
                    };
                    if now.saturating_sub(since) >= idle_sec.saturating_mul(1000) {
                        due.push(IdleAction::Sleep(id.clone()));
                    }
                }
                (Mode::OnDemand, ModelState::Sleeping { .. }) => {
                    let since = process.state_since_ms.unwrap_or(now);
                    if now.saturating_sub(since) >= mode.stop_after_sec().saturating_mul(1000) {
                        due.push(IdleAction::Stop(id.clone()));
                    }
                }
                _ => {}
            }
        }
        Ok(due)
    }

    /// Decide and act. Each action is independent: one failing never stops the rest.
    pub fn tick(&mut self, lifecycle: &ModelLifecycle) -> Vec<(IdleAction, Result<()>)> {
        let due = match self.due(lifecycle.root(), lifecycle.host()) {
            Ok(due) => due,
            Err(error) => return vec![(IdleAction::Sleep(String::new()), Err(error))],
        };
        due.into_iter()
            .map(|action| {
                let result = match &action {
                    IdleAction::Sleep(id) => lifecycle.sleep(id, "idle"),
                    IdleAction::Stop(id) => lifecycle.stop(id, "asleep past stop_after_sec"),
                    IdleAction::Wake(id) => {
                        lifecycle.ensure_ready(id, Instant::now() + Duration::from_secs(600))
                    }
                };
                (action, result)
            })
            .collect()
    }
}

/// The agent's sleeper thread body.
pub fn run(lifecycle: std::sync::Arc<ModelLifecycle>) {
    let mut sleeper = IdleSleeper::default();
    let mut last_error: Option<String> = None;
    loop {
        for (action, result) in sleeper.tick(&lifecycle) {
            match result {
                Ok(()) => {}
                Err(error) => {
                    let message = format!("{action:?}: {error:#}");
                    if last_error.as_deref() != Some(message.as_str()) {
                        eprintln!("states: idle sleeper {message}");
                        last_error = Some(message);
                    }
                }
            }
        }
        std::thread::sleep(TICK);
    }
}
