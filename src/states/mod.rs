//! Per-model lifecycle states, sleep/wake, and admission for sleeping models (M3).
//!
//! ```text
//! stopped → starting → ready ⇄ draining → sleeping → waking → ready
//!                        └──── draining → stopped (engines without sleep, stop_after_sec)
//! ```
//!
//! The state lives on each process record in `.compute/state.yaml`
//! (`proc::Proc::state`); a record without one is `Ready`, which keeps every
//! state file written before M3 valid. A model with no live owned record is
//! `Stopped`.
//!
//! This module implements the lifecycle contract (M3): the
//! [`Lifecycle`] trait the on-demand gateway consumes, its implementation
//! [`ModelLifecycle`], the admission arithmetic for sleeping models, and the
//! agent's idle sleeper ([`idle`]). See `docs/sleep.md`.

pub mod idle;
mod machine;
#[cfg(test)]
mod tests;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

use crate::config::{default_sleeping_footprint_gb, Mode, Model, Registry, Scene};
use crate::governor;
use crate::proc::{Proc, State};

pub use machine::ModelLifecycle;

/// Where one model is in its lifecycle. Persisted, shown by `status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelState {
    Stopped,
    Starting,
    Ready,
    /// No new requests; waiting for in-flight to reach 0 before sleep or stop.
    Draining,
    Sleeping {
        level: u8,
    },
    Waking,
}

impl ModelState {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelState::Stopped => "stopped",
            ModelState::Starting => "starting",
            ModelState::Ready => "ready",
            ModelState::Draining => "draining",
            ModelState::Sleeping { .. } => "sleeping",
            ModelState::Waking => "waking",
        }
    }

    /// Mid-transition or asleep: the model is not expected to answer, so
    /// supervision must neither probe it as failed nor restart it.
    pub fn in_transition_or_asleep(self) -> bool {
        matches!(
            self,
            ModelState::Starting
                | ModelState::Draining
                | ModelState::Sleeping { .. }
                | ModelState::Waking
        )
    }
}

impl std::fmt::Display for ModelState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelState::Sleeping { level } => write!(f, "sleeping (level {level})"),
            other => f.write_str(other.as_str()),
        }
    }
}

/// The lifecycle contract (M3). [`ModelLifecycle`] implements it; the on-demand
/// gateway consumes it.
pub trait Lifecycle: Send + Sync {
    fn state(&self, id: &str) -> ModelState;
    /// Wake or start `id` (making room per the admission rules) and block until
    /// ready or `deadline`. Idempotent; concurrent callers share one wake.
    fn ensure_ready(&self, id: &str, deadline: std::time::Instant) -> anyhow::Result<()>;
    /// Drain then sleep (or stop, for engines without sleep). Refuses resident models.
    fn sleep(&self, id: &str, reason: &str) -> anyhow::Result<()>;
}

/// What the lifecycle needs from the box beyond the registry and state file:
/// the clock, memory, and how busy a model is. Injected so tests can pin
/// every one of them.
pub trait Host: Send + Sync {
    /// Wall clock, Unix ms. Persisted timestamps and the idle clock use it.
    fn now_ms(&self) -> u64;
    /// `(total_gb, available_gb)` of `pool`, or `None` when sensing failed.
    fn memory(&self, pool: &governor::Pool) -> Option<(f64, f64)>;
    /// Requests the gateway holds open for `id`, including those held while it wakes.
    fn in_flight(&self, id: &str) -> u32;
    /// The subset of `in_flight` held for `id`'s wake, not yet forwarded.
    fn waiting(&self, _id: &str) -> u32 {
        0
    }
    /// When the gateway last finished a request for `id`, Unix ms.
    fn last_finished_ms(&self, id: &str) -> Option<u64>;
    /// Requests the engine itself reports (running + waiting): catches clients
    /// that talk to the model's port directly. `None` = unknown.
    fn engine_in_flight(&self, model: &Model) -> Option<u32> {
        crate::health::engine_load(model).map(|load| load.in_flight())
    }
    /// How often waits re-check their condition.
    fn poll(&self) -> Duration {
        Duration::from_millis(200)
    }
}

pub fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// The host inside the agent: the gateway's own in-flight counters.
pub struct AgentHost {
    in_flight: crate::gateway::InFlight,
}

impl AgentHost {
    pub fn new(in_flight: crate::gateway::InFlight) -> Self {
        Self { in_flight }
    }
}

impl Host for AgentHost {
    fn now_ms(&self) -> u64 {
        unix_now_ms()
    }
    fn memory(&self, pool: &governor::Pool) -> Option<(f64, f64)> {
        pool.sense().ok().map(|m| (m.total_gb, m.available_gb))
    }
    fn in_flight(&self, id: &str) -> u32 {
        self.in_flight.get(id)
    }
    fn waiting(&self, id: &str) -> u32 {
        self.in_flight.waiting(id)
    }
    fn last_finished_ms(&self, id: &str) -> Option<u64> {
        self.in_flight.last_finished_ms(id)
    }
}

/// The host for a CLI command: in-flight counts come from the running agent's
/// gateway (`/lumbridge-computer/v1/inflight`); no agent means nothing in flight.
pub struct CliHost {
    gateway_addr: String,
}

impl CliHost {
    pub fn new(gateway_addr: &str) -> Self {
        Self {
            gateway_addr: gateway_addr.to_string(),
        }
    }
}

/// `in_flight` and `last_finished_ms` as the agent's gateway reports them.
#[derive(Debug, Default, Deserialize)]
pub struct GatewayCounts {
    #[serde(default)]
    pub in_flight: std::collections::BTreeMap<String, u32>,
    #[serde(default)]
    pub last_finished_ms: std::collections::BTreeMap<String, u64>,
    #[serde(default)]
    pub waiting: std::collections::BTreeMap<String, u32>,
}

pub fn read_gateway_counts(addr: &str) -> Option<GatewayCounts> {
    let body = crate::gateway::fetch_inflight(
        addr,
        Duration::from_millis(300),
        Duration::from_millis(700),
    )?;
    serde_json::from_str(&body).ok()
}

impl Host for CliHost {
    fn now_ms(&self) -> u64 {
        unix_now_ms()
    }
    fn memory(&self, pool: &governor::Pool) -> Option<(f64, f64)> {
        pool.sense().ok().map(|m| (m.total_gb, m.available_gb))
    }
    fn in_flight(&self, id: &str) -> u32 {
        read_gateway_counts(&self.gateway_addr)
            .and_then(|counts| counts.in_flight.get(id).copied())
            .unwrap_or(0)
    }
    fn waiting(&self, id: &str) -> u32 {
        read_gateway_counts(&self.gateway_addr)
            .and_then(|counts| counts.waiting.get(id).copied())
            .unwrap_or(0)
    }
    fn last_finished_ms(&self, id: &str) -> Option<u64> {
        read_gateway_counts(&self.gateway_addr)
            .and_then(|counts| counts.last_finished_ms.get(id).copied())
    }
}

// ---- admission accounting ---------------------------------------------------

/// What `id` holds while asleep, in GB.
///
/// In order: the peak footprint sampled since its *current* sleep began (only
/// while it is asleep), then the registry's `serve.sleep.footprint_gb`, then
/// `max(6, 0.3 × footprint_gb)` (the spike's measurement). Never above its full
/// footprint.
pub fn sleeping_gb(root: &Path, registry: &Registry, id: &str, process: Option<&Proc>) -> f64 {
    let Some(model) = registry.models.get(id) else {
        return 0.0;
    };
    let declared = model.serve.sleep.as_ref().map_or_else(
        || default_sleeping_footprint_gb(model.footprint_gb),
        |sleep| sleep.footprint_for(model.footprint_gb),
    );
    let measured = process
        .filter(|p| matches!(p.model_state(), ModelState::Sleeping { .. }))
        .and_then(|p| p.state_since_ms)
        .and_then(|since| measured_since(root, id, since));
    measured
        .unwrap_or(declared)
        .clamp(0.0, model.footprint_gb.max(0.0))
}

fn measured_since(root: &Path, id: &str, since_ms: u64) -> Option<f64> {
    let samples = crate::footprint::Samples::open(root).ok()?;
    let learned = samples.learned(id, since_ms as i64).ok()??;
    Some(learned.peak_demand_gb)
}

/// What `id` counts for in admission right now: its sleeping footprint while
/// asleep, its admission footprint while it holds (or is taking) its memory,
/// 0 when stopped. A model serving outside Lumbridge Computer's ownership counts in full
/// when it answers its health check, as before M3.
pub fn counted_gb(root: &Path, registry: &Registry, state: &State, id: &str) -> f64 {
    let Some(model) = registry.models.get(id) else {
        return 0.0;
    };
    match state.procs.get(id).filter(|p| p.owned_alive()) {
        Some(process) => match process.model_state() {
            ModelState::Sleeping { .. } => sleeping_gb(root, registry, id, Some(process)),
            ModelState::Stopped => 0.0,
            _ => crate::footprint::admission_gb(root, registry, id),
        },
        None if governor::is_running(model) => crate::footprint::admission_gb(root, registry, id),
        None => 0.0,
    }
}

/// Committed memory in `pool`, every model at [`counted_gb`].
pub fn committed_gb(root: &Path, registry: &Registry, state: &State, pool: &governor::Pool) -> f64 {
    registry
        .models
        .iter()
        .filter(|(_, model)| pool.holds(model))
        .map(|(id, _)| counted_gb(root, registry, state, id))
        .sum()
}

/// A Scene's footprint for its budget check (M3).
///
/// Residents count in full. Non-resident models count at their sleeping
/// footprint (an `on_demand` model without sleep at 0: idle means stopped),
/// plus room for the largest one of them to be awake. So a Scene validates
/// when everything resident fits with at least one non-resident model awake;
/// waking a second one makes room by sleeping the first. Without `modes:` this
/// is exactly the sum of footprints, as before M3.
pub fn scene_need_gb(registry: &Registry, scene: &Scene, pool: &governor::Pool) -> f64 {
    let mut resident = 0.0;
    let mut floors = 0.0;
    let mut widest: f64 = 0.0;
    for id in &scene.models {
        let Some(model) = registry.models.get(id) else {
            continue;
        };
        if !pool.holds(model) {
            continue;
        }
        let full = model.footprint_gb;
        match scene.mode_of(id).mode {
            Mode::Resident => resident += full,
            Mode::Sleepable | Mode::OnDemand => {
                let floor = if crate::engines::sleep::supports_sleep(model) {
                    model
                        .serve
                        .sleep
                        .as_ref()
                        .map_or(full, |sleep| sleep.footprint_for(full))
                } else {
                    0.0
                };
                floors += floor;
                widest = widest.max(full - floor);
            }
        }
    }
    resident + floors + widest
}

/// Scene `modes:` rules: every entry names a Scene model, `sleepable` needs an
/// engine that can sleep, and the knobs only appear where they mean something.
pub fn validate_modes(registry: &Registry, scene: &Scene) -> Result<()> {
    let name = &scene.metadata.name;
    for (id, mode) in &scene.modes {
        if !scene.models.contains(id) {
            anyhow::bail!("Scene '{name}' sets a mode for '{id}', which is not one of its models");
        }
        let Some(model) = registry.models.get(id) else {
            continue;
        };
        match mode.mode {
            Mode::Resident => {
                if mode.idle_sec.is_some() || mode.stop_after_sec.is_some() {
                    anyhow::bail!(
                        "Scene '{name}': '{id}' is resident, so idle_sec/stop_after_sec do not apply"
                    );
                }
            }
            Mode::Sleepable => {
                if !crate::engines::sleep::supports_sleep(model) {
                    anyhow::bail!(
                        "Scene '{name}': '{id}' cannot be sleepable: only vLLM models with \
                         `serve.sleep` in the registry can sleep (engine '{}'); use on_demand",
                        crate::engines::detect(model)
                    );
                }
                if mode.stop_after_sec.is_some() {
                    anyhow::bail!(
                        "Scene '{name}': stop_after_sec only applies to on_demand models, not '{id}'"
                    );
                }
            }
            Mode::OnDemand => {}
        }
    }
    Ok(())
}

/// Should supervision leave `id` alone this tick? Yes while it is asleep or
/// mid-transition, and for a non-resident model with no live process: being
/// stopped is its normal idle state, never a failure to repair.
pub fn supervision_paused(mode: Mode, process: Option<&Proc>) -> bool {
    match process {
        Some(process) if process.model_state().in_transition_or_asleep() => true,
        Some(process) => mode != Mode::Resident && process.exited(),
        None => mode != Mode::Resident,
    }
}

/// Should a Scene transition expect `id` to answer its health check? Not for
/// `on_demand` models (they start on the first request) nor for a model that
/// is asleep or mid-transition under the lifecycle.
pub fn expects_healthy(scene: &Scene, state: &State, id: &str) -> bool {
    if scene.mode_of(id).mode == Mode::OnDemand {
        return false;
    }
    !state
        .procs
        .get(id)
        .filter(|p| p.owned_alive())
        .is_some_and(|p| p.model_state().in_transition_or_asleep())
}
