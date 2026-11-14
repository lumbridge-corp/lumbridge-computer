//! Config types + loaders for the model registry and scenes.
//!
//! The registry (`registry/models.yaml`) resolves stable model *ids* to weights +
//! launch commands, and evolves as models requantize. Scenes (`scenes/*.scene.yaml`)
//! reference those ids and stay stable — the public, shareable contract.
//!
//! Several fields below are deserialized but never read by the binary
//! (`api_version`, `channel`, `entry`, `app`, `app_dir`, `tags`, `author`).
//! That is deliberate: they are the published `lumbridge/v1` manifest surface,
//! and declaring them is what makes a manifest carrying them parse rather than
//! fail. Deleting them to satisfy the lint would silently narrow the contract.
#![allow(dead_code)]

/// The one scene shipped by default with Lumbridge Computer.
pub const DEFAULT_SCENE: &str = "everything";

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Deserialize)]
pub struct Registry {
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    pub models: BTreeMap<String, Model>,
    /// Memory held back for tenants that are not models (the Factory's Server
    /// database, first). Host policy like `supervision`, so it lives in the local
    /// registry and never in a shareable Scene.
    #[serde(default)]
    pub reservations: BTreeMap<String, Reservation>,
    /// Spawn each model in its own transient systemd user scope (M2). Default on;
    /// `off` keeps the M1 behaviour (models share the agent's cgroup).
    #[serde(default)]
    pub scopes: ScopesMode,
    /// Which footprint admission budgets against. Default `declared`.
    #[serde(default)]
    pub admission: AdmissionMode,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ScopesMode {
    #[default]
    On,
    Off,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AdmissionMode {
    /// The registry's `footprint_gb`, exactly as typed.
    #[default]
    Declared,
    /// `max(footprint_gb, learned peak)`: a model that has been seen using more
    /// than it declares is budgeted at what it really used.
    Measured,
}

/// Per-model cgroup limits for its scope (all optional; see `scope::resolve`).
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    /// `MemoryHigh` (throttle + reclaim, never kill). Default footprint × 1.15.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_high_gb: Option<f64>,
    /// `MemoryMax` (OOM-kill inside the scope). Default none: never kill a brain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_max_gb: Option<f64>,
    /// `MemorySwapMax`. Default 0 for models that hold memory, none for 0 GB shims.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swap_max_gb: Option<f64>,
}

/// Unified memory promised to a non-model tenant on this node.
#[derive(Debug, Clone, Deserialize)]
pub struct Reservation {
    /// The ceiling the tenant is held to (e.g. the sum of its containers' limits).
    pub gb: f64,
    /// cgroup v2 path under /sys/fs/cgroup whose `memory.current` is the tenant's
    /// real use, e.g. `lumbridge.slice/lumbridge-server.slice`. Without it (or when
    /// it can't be read) the whole reservation is treated as unused, which is the
    /// safe direction: models get less, the tenant keeps its room.
    #[serde(default)]
    pub cgroup: Option<String>,
    #[serde(default)]
    pub description: String,
    /// Memory pool the reservation is held in; `unified` on the Spark.
    #[serde(default)]
    pub pool: crate::pool::PoolId,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Model {
    pub name: String,
    /// Worst-case unified memory held once serving (weights + KV/cache + encoder).
    pub footprint_gb: f64,
    /// Memory pool the footprint is admitted against. Optional: every model on the
    /// Spark lives in `unified`, and an unknown pool is a load error.
    #[serde(default)]
    pub pool: crate::pool::PoolId,
    /// `stable` pins the exact weights; `latest` may resolve a newer quant.
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub health: Option<String>,
    /// Optional marker that must appear in the health response. This
    /// distinguishes mutually-exclusive models that intentionally share a port.
    #[serde(default)]
    pub health_contains: Option<String>,
    /// Opt-in resident-agent recovery. Models without this block keep the
    /// original one-shot Scene lifecycle exactly as it was.
    #[serde(default)]
    pub supervision: Option<Supervision>,
    /// Watchdog shed order (2026-10-09). When MemAvailable falls under the floor the
    /// watchdog stops the alive model with the HIGHEST `shed_priority` first (default 0),
    /// newest first among equals. Models with a 0 GB footprint (shims) are shed last:
    /// stopping them frees nothing. Negative = critical (brain, TTS, ASR).
    #[serde(default)]
    pub shed_priority: Option<i32>,
    /// cgroup limits for this model's scope; omitted means the defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limits: Option<Limits>,
    pub serve: Serve,
    /// How the HTTP gateway reaches this model. Optional: without it requests go
    /// to `serve.port` with their bytes untouched.
    #[serde(default, skip_serializing_if = "ModelGateway::is_default")]
    pub gateway: ModelGateway,
}

/// Per-model gateway routing policy (host policy, so registry-only like `serve`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelGateway {
    /// Send this model's requests to `host:port` instead of `127.0.0.1:<serve.port>`,
    /// e.g. a tracing proxy in front of the engine (brain → proxy :8041 → :8001).
    #[serde(default)]
    pub upstream: Option<String>,
    /// Replace the client's `model` with this model's own name (first `served_name`,
    /// else the registry id) before forwarding, for engines that only accept that.
    #[serde(default)]
    pub rewrite_model: bool,
}

impl ModelGateway {
    /// `model show` omits an all-default block, so models that never mention
    /// `gateway:` print exactly as they did before it existed.
    pub fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

/// Recovery policy for one model in the persisted desired Scene.
///
/// This belongs to the local registry rather than a shareable Scene: restarting
/// a runtime is an operator policy tied to the vetted command and footprint on
/// this node, not something a downloaded Scene may turn on.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Supervision {
    pub restart: RestartPolicy,
    #[serde(default = "default_check_interval_sec")]
    pub check_interval_sec: u64,
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,
    #[serde(default = "default_backoff_sec")]
    pub backoff_sec: u64,
    #[serde(default = "default_max_backoff_sec")]
    pub max_backoff_sec: u64,
    /// Startup no-progress window. A starting model is given up on only when
    /// its process exits, or when nothing moved (port, log file, metrics) for
    /// this long; any progress resets the window. See `lifecycle::wait_started`.
    #[serde(default = "default_startup_timeout_sec")]
    pub startup_timeout_sec: u64,
    /// How long the readiness probe (connect plus the `health` GET) may take.
    /// Hundreds of milliseconds is too tight for an engine busy with a long
    /// prefill on a box under memory pressure; a timeout while the engine
    /// reports work in flight counts as busy, not failed.
    #[serde(default)]
    pub probe_timeout_ms: Option<u64>,
    /// Opt-in check that the engine is still making progress, on top of the
    /// readiness probe above. Without it a model whose API answers but whose
    /// engine is frozen looks healthy forever (brain, 2026-11-05: 6.5 hours).
    #[serde(default)]
    pub liveness: Option<Liveness>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RestartPolicy {
    Always,
}

/// How a supervised model proves its engine is moving, not just that its HTTP
/// server answers.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Liveness {
    pub kind: LivenessKind,
    /// Prometheus endpoint on the model's own serving port.
    #[serde(default = "default_metrics_path")]
    pub metrics_path: String,
    /// How long requests may be in flight with no token counted before the
    /// model counts as failing its health check.
    #[serde(default = "default_stall_sec")]
    pub stall_sec: u64,
    /// Optional real-request probe. The metrics rule can't see an engine that
    /// stops reporting: on 2026-11-06 vLLM's EngineCore spun at 100% CPU with
    /// `num_requests_running` frozen at 0, so it looked idle while every chat hung
    /// for two hours and `/v1/models` kept answering.
    #[serde(default)]
    pub chat_probe: Option<ChatProbe>,
}

/// A 1-token `POST /v1/chat/completions` on the model's own port, sent only when
/// the engine looks idle and its token count hasn't moved since the last check,
/// so it never competes with real traffic. A failure is a failed health check.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatProbe {
    /// Least time between two probes.
    #[serde(default = "default_probe_interval_sec")]
    pub interval_sec: u64,
    /// How long the 1-token answer may take.
    #[serde(default = "default_probe_timeout_sec")]
    pub timeout_sec: u64,
    /// Model name to send; defaults to the first `served_name`.
    #[serde(default)]
    pub model: Option<String>,
}

pub const MIN_PROBE_INTERVAL_SEC: u64 = 30;

impl ChatProbe {
    pub fn interval_sec(&self) -> u64 {
        self.interval_sec.max(MIN_PROBE_INTERVAL_SEC)
    }
    pub fn timeout_sec(&self) -> u64 {
        self.timeout_sec.clamp(5, 120)
    }
}

const fn default_probe_interval_sec() -> u64 {
    120
}

const fn default_probe_timeout_sec() -> u64 {
    30
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LivenessKind {
    /// vLLM `/metrics`: `num_requests_running + num_requests_waiting > 0` while
    /// `prompt_tokens_total + generation_tokens_total` stays flat.
    VllmMetrics,
}

/// The shortest stall window accepted. vLLM counts prompt tokens only when a
/// request emits its first token, so a long prefill is legitimately silent;
/// a typo such as `stall_sec: 5` must not turn every long prompt into a restart.
pub const MIN_STALL_SEC: u64 = 60;

impl Liveness {
    pub fn stall_sec(&self) -> u64 {
        self.stall_sec.max(MIN_STALL_SEC)
    }

    /// Only a plain absolute path on the model's own port; anything else falls
    /// back to `/metrics`, like `health` is held to localhost.
    pub fn metrics_path(&self) -> &str {
        if self.metrics_path.starts_with('/') && !self.metrics_path.contains("://") {
            &self.metrics_path
        } else {
            "/metrics"
        }
    }
}

fn default_metrics_path() -> String {
    "/metrics".to_string()
}

const fn default_stall_sec() -> u64 {
    300
}

impl Supervision {
    pub fn check_interval_sec(&self) -> u64 {
        self.check_interval_sec.max(1)
    }

    pub fn failure_threshold(&self) -> u32 {
        self.failure_threshold.max(1)
    }

    pub fn backoff_sec(&self) -> u64 {
        self.backoff_sec.max(1)
    }

    pub fn max_backoff_sec(&self) -> u64 {
        self.max_backoff_sec.max(self.backoff_sec())
    }

    pub fn startup_timeout_sec(&self) -> u64 {
        self.startup_timeout_sec.max(1)
    }

    pub fn probe_timeout(&self) -> Duration {
        Duration::from_millis(
            self.probe_timeout_ms
                .unwrap_or(DEFAULT_PROBE_TIMEOUT_MS)
                .clamp(MIN_PROBE_TIMEOUT_MS, MAX_PROBE_TIMEOUT_MS),
        )
    }
}

/// Readiness probe timeout when `supervision.probe_timeout_ms` is not set.
pub const DEFAULT_PROBE_TIMEOUT_MS: u64 = 5_000;
const MIN_PROBE_TIMEOUT_MS: u64 = 500;
const MAX_PROBE_TIMEOUT_MS: u64 = 60_000;

impl Model {
    /// The readiness probe timeout for this model, supervised or not.
    pub fn probe_timeout(&self) -> Duration {
        self.supervision
            .as_ref()
            .map(Supervision::probe_timeout)
            .unwrap_or(Duration::from_millis(DEFAULT_PROBE_TIMEOUT_MS))
    }
}

const fn default_check_interval_sec() -> u64 {
    30
}

const fn default_failure_threshold() -> u32 {
    3
}

const fn default_backoff_sec() -> u64 {
    60
}

const fn default_max_backoff_sec() -> u64 {
    900
}

const fn default_startup_timeout_sec() -> u64 {
    180
}

/// How to launch a model. Fields are permissive across runtimes (vllm / diffusers /
/// python / uvicorn); only the ones a given `kind` needs are populated.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Serve {
    pub kind: String,
    /// Optional explicit serving engine (`vllm`, `sglang`, `llama-cpp`,
    /// `nemo-speech`, `generic`). Without it the engine is detected from
    /// `kind` and the launch argv; see `engines::detect`.
    #[serde(default)]
    pub engine: Option<crate::engines::Engine>,
    /// Optional runtime executable for kind-based builders (for example a
    /// model-specific vLLM virtualenv). Defaults to the kind's command on PATH.
    #[serde(default)]
    pub executable: Option<String>,
    #[serde(default)]
    pub port: Option<u16>,
    /// Explicit launch argv. If set, it is used verbatim (argv[0] = program) and
    /// takes precedence over any `kind`-based builder. This is where launch commands
    /// live — never in a scene — so downloaded scenes can't smuggle code.
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub weights: Option<String>,
    #[serde(default)]
    pub entry: Option<String>,
    #[serde(default)]
    pub app: Option<String>,
    #[serde(default)]
    pub app_dir: Option<String>,
    #[serde(default)]
    pub served_name: Vec<String>,
    #[serde(default)]
    pub args: BTreeMap<String, serde_yaml::Value>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Opt-in engine sleep (M3). Only vLLM supports it, through the `vllm`
    /// builder: the entry adds `--enable-sleep-mode`, Lumbridge Computer's worker
    /// extension, `VLLM_SERVER_DEV_MODE=1` and a loopback bind to the launch,
    /// and lets a Scene mark the model `sleepable`. See `docs/sleep.md`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sleep: Option<SleepConfig>,
}

/// How a model sleeps and wakes. The defaults are what was measured on a GB10
/// box (`docs/sleep.md`); every one can be overridden here.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SleepConfig {
    /// vLLM sleep level. Only 2 is accepted: on GB10 level 1 backs the weights
    /// up into the same unified pool, frees only the KV cache, and on a hybrid
    /// model failed with HTTP 500 and killed the engine (spike, "Level 1").
    #[serde(default = "default_sleep_level")]
    pub level: u8,
    /// What the model holds while asleep, used for admission when no footprint
    /// sample was taken during the current sleep. Default
    /// `max(6, 0.3 × footprint_gb)`: a sleeping vLLM keeps its whole host side
    /// plus the CUDA context and graph pool (6.4 GB measured for a 4B model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint_gb: Option<f64>,
    /// Calls that wake the engine, in order, on its own loopback port.
    /// Default: `POST /wake_up?tags=weights`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake: Option<Vec<HttpStep>>,
    /// Calls after `wake` that restore what level 2 dropped. Default:
    /// `POST /collective_rpc {"method":"reload_all_fast"}` (Lumbridge Computer's worker
    /// extension: target *and* MTP drafter, ~30× faster than the stock
    /// `reload_weights` on GB10), then `POST /wake_up?tags=kv_cache`, which
    /// resumes the scheduler.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reload: Option<Vec<HttpStep>>,
    /// How long `POST /sleep` may take. Default 120 s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sleep_timeout_sec: Option<u64>,
    /// How long the whole wake (calls plus readiness) may take, and how long the
    /// gateway holds a request for this model while it is asleep. Default 60 s;
    /// a cold start keeps the gateway's own `--wake-timeout-sec`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_timeout_sec: Option<u64>,
    /// How long to wait for in-flight requests to finish before sleeping or
    /// stopping. Default 30 s; a drain that times out leaves the model ready.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drain_timeout_sec: Option<u64>,
    /// Directory holding `lumbridge_computer_sleep_ext.py`, put first on the engine's
    /// `PYTHONPATH`. Default: `assets/vllm` under the Lumbridge Computer root, else next
    /// to the binary (see `proc::sleep_extension_dir`). A relative path is
    /// relative to the Lumbridge Computer root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extension_dir: Option<String>,
    /// Keep the engine's own `--host` instead of forcing 127.0.0.1. Dev mode
    /// exposes `/collective_rpc` (any worker method) to whoever reaches the
    /// port, so this is an escape hatch that `lumbridge-computer doctor` flags.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_public_dev_mode: bool,
}

/// One loopback HTTP call to the engine. `path` is held to a plain absolute
/// path on the model's own port, like `health`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HttpStep {
    #[serde(default = "default_step_method")]
    pub method: String,
    pub path: String,
    /// JSON body, sent with `Content-Type: application/json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

impl HttpStep {
    pub fn post(path: &str, body: Option<&str>) -> HttpStep {
        HttpStep {
            method: "POST".to_string(),
            path: path.to_string(),
            body: body.map(str::to_string),
        }
    }
}

fn default_step_method() -> String {
    "POST".to_string()
}

const fn default_sleep_level() -> u8 {
    2
}

/// The only sleep level Lumbridge Computer accepts on GB10.
pub const SLEEP_LEVEL: u8 = 2;
/// Floor of the default sleeping footprint (GB).
pub const MIN_SLEEP_FOOTPRINT_GB: f64 = 6.0;
/// Share of the full footprint a sleeping model is assumed to keep.
pub const SLEEP_FOOTPRINT_SHARE: f64 = 0.3;
/// Drain timeout used by models without a `serve.sleep` block (they stop instead).
pub const DEFAULT_DRAIN_TIMEOUT_SEC: u64 = 30;
/// Default `serve.sleep.wake_timeout_sec`. The spike measured 1.4–2.0 s for an
/// 8 GB checkpoint and expects 5–15 s for the 22 GB brain.
pub const DEFAULT_SLEEP_WAKE_TIMEOUT_SEC: u64 = 60;
/// The worker extension's import path, as vLLM's `--worker-extension-cls` wants it.
pub const SLEEP_EXTENSION_CLS: &str = "lumbridge_computer_sleep_ext.LumbridgeComputerSleepExt";
/// The worker extension's file name inside `extension_dir`.
pub const SLEEP_EXTENSION_FILE: &str = "lumbridge_computer_sleep_ext.py";

/// The sleeping footprint for a model whose full footprint is `full_gb`, when
/// nothing measured or declared says otherwise.
pub fn default_sleeping_footprint_gb(full_gb: f64) -> f64 {
    MIN_SLEEP_FOOTPRINT_GB.max(SLEEP_FOOTPRINT_SHARE * full_gb.max(0.0))
}

impl SleepConfig {
    pub fn wake_steps(&self) -> Vec<HttpStep> {
        self.wake
            .clone()
            .unwrap_or_else(|| vec![HttpStep::post("/wake_up?tags=weights", None)])
    }

    pub fn reload_steps(&self) -> Vec<HttpStep> {
        self.reload.clone().unwrap_or_else(|| {
            vec![
                HttpStep::post("/collective_rpc", Some(r#"{"method":"reload_all_fast"}"#)),
                HttpStep::post("/wake_up?tags=kv_cache", None),
            ]
        })
    }

    /// Declared (or default) sleeping footprint, never above the full one.
    pub fn footprint_for(&self, full_gb: f64) -> f64 {
        self.footprint_gb
            .unwrap_or_else(|| default_sleeping_footprint_gb(full_gb))
            .clamp(0.0, full_gb.max(0.0))
    }

    pub fn sleep_timeout(&self) -> Duration {
        Duration::from_secs(self.sleep_timeout_sec.unwrap_or(120).clamp(1, 1800))
    }

    pub fn wake_timeout(&self) -> Duration {
        Duration::from_secs(
            self.wake_timeout_sec
                .unwrap_or(DEFAULT_SLEEP_WAKE_TIMEOUT_SEC)
                .clamp(1, 3600),
        )
    }

    pub fn drain_timeout(&self) -> Duration {
        Duration::from_secs(
            self.drain_timeout_sec
                .unwrap_or(DEFAULT_DRAIN_TIMEOUT_SEC)
                .min(3600),
        )
    }

    fn validate(&self, id: &str) -> Result<()> {
        if self.level != SLEEP_LEVEL {
            anyhow::bail!(
                "model '{id}': serve.sleep.level must be 2, got {}: on GB10 level 1 frees only the \
                 KV cache and returned HTTP 500 and killed the engine on a hybrid model \
                 (docs/sleep.md, \"Why only level 2\")",
                self.level
            );
        }
        if let Some(gb) = self.footprint_gb {
            if !gb.is_finite() || gb < 0.0 {
                anyhow::bail!("model '{id}': serve.sleep.footprint_gb must be >= 0");
            }
        }
        if self.extension_dir.as_deref().is_some_and(str::is_empty) {
            anyhow::bail!("model '{id}': serve.sleep.extension_dir must not be empty");
        }
        for step in self.wake_steps().iter().chain(self.reload_steps().iter()) {
            if !matches!(step.method.as_str(), "POST" | "GET") {
                anyhow::bail!(
                    "model '{id}': serve.sleep step method must be POST or GET, got '{}'",
                    step.method
                );
            }
            if !step.path.starts_with('/') || step.path.contains("://") || step.path.contains(' ') {
                anyhow::bail!(
                    "model '{id}': serve.sleep step path must be a plain absolute path, got '{}'",
                    step.path
                );
            }
        }
        Ok(())
    }
}

/// How a Scene keeps one of its models (M3). Lives in the Scene, not the
/// registry: it says when a model runs, never how, so a shared Scene can't use
/// it to smuggle a command.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Always up and supervised, never slept automatically (the M1 behaviour).
    #[default]
    Resident,
    /// Up with the Scene, slept after `idle_sec` without requests, woken on demand.
    Sleepable,
    /// Not started with the Scene: started by the first request, slept when idle,
    /// stopped after `stop_after_sec` asleep.
    OnDemand,
}

impl Mode {
    /// May a request wake or start it? (The gateway holds requests for these.)
    pub fn wakes_on_demand(self) -> bool {
        self != Mode::Resident
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Resident => "resident",
            Mode::Sleepable => "sleepable",
            Mode::OnDemand => "on_demand",
        }
    }
}

/// A Scene's `modes:` entry for one model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelMode {
    #[serde(default)]
    pub mode: Mode,
    /// Seconds without requests before a ready model is slept. `0` means never
    /// for idleness: the model is only slept to make room for another. Default 600.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_sec: Option<u64>,
    /// `on_demand` only: seconds asleep before the model is stopped. Default 1800.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_after_sec: Option<u64>,
}

pub const DEFAULT_IDLE_SEC: u64 = 600;
/// A typo like `idle_sec: 5` must not make a model flap between sleep and wake.
pub const MIN_IDLE_SEC: u64 = 30;
pub const DEFAULT_STOP_AFTER_SEC: u64 = 1800;

impl ModelMode {
    /// `None`: never slept for idleness.
    pub fn idle_sec(&self) -> Option<u64> {
        match self.idle_sec.unwrap_or(DEFAULT_IDLE_SEC) {
            0 => None,
            sec => Some(sec.max(MIN_IDLE_SEC)),
        }
    }

    pub fn stop_after_sec(&self) -> u64 {
        self.stop_after_sec.unwrap_or(DEFAULT_STOP_AFTER_SEC)
    }
}

#[derive(Debug, Deserialize)]
pub struct Scene {
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    pub metadata: SceneMeta,
    /// Stable model ids resolved via the registry.
    pub models: Vec<String>,
    /// Optional per-scene budget override (GB); defaults to the Governor's global budget.
    #[serde(default)]
    pub budget_gb: Option<f64>,
    #[serde(default)]
    pub activation: Option<Activation>,
    /// Per-model modes (M3); a model not listed is `resident`.
    #[serde(default)]
    pub modes: BTreeMap<String, ModelMode>,
}

#[derive(Debug, Deserialize)]
pub struct SceneMeta {
    pub name: String,
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub author: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Activation {
    /// `footprint-asc` (default) | `listed`.
    #[serde(default)]
    pub order: Option<String>,
    #[serde(default)]
    pub wait_healthy: Option<bool>,
}

impl Registry {
    /// Load `<root>/registry/models.yaml`, resolving `extends:` and macros.
    pub fn load(root: &Path) -> Result<Registry> {
        Registry::load_file(&root.join("registry/models.yaml"))
    }

    /// Load a registry from any path, resolving `extends:` and macros.
    pub fn load_file(p: &Path) -> Result<Registry> {
        let s =
            fs::read_to_string(p).with_context(|| format!("reading registry {}", p.display()))?;
        Registry::parse(&s).with_context(|| format!("parsing registry {}", p.display()))
    }

    /// Parse registry YAML. Every returned `Model` is fully resolved: callers
    /// never see `extends:` or an unexpanded `${MACRO}` (the cross-lane contract).
    pub fn parse(s: &str) -> Result<Registry> {
        let registry = resolve::parse_registry(s)?;
        for (id, model) in &registry.models {
            if let Some(sleep) = &model.serve.sleep {
                if crate::engines::detect(model) != crate::engines::Engine::Vllm {
                    anyhow::bail!(
                        "model '{id}': serve.sleep is only supported for vLLM, not engine '{}'",
                        crate::engines::detect(model)
                    );
                }
                sleep.validate(id)?;
                // The builder is what forces the loopback bind, the worker
                // extension and dev mode; a hand-written argv can't be checked.
                if !model.serve.command.is_empty() {
                    anyhow::bail!(
                        "model '{id}': serve.sleep needs the vllm builder (kind: vllm without \
                         `command`), so Lumbridge Computer can add the worker extension and bind the \
                         dev-mode engine to 127.0.0.1"
                    );
                }
                for key in ["worker-extension-cls", "worker_extension_cls"] {
                    if model.serve.args.contains_key(key) {
                        anyhow::bail!(
                            "model '{id}': serve.sleep sets --worker-extension-cls itself \
                             ({SLEEP_EXTENSION_CLS}); remove `{key}` from serve.args"
                        );
                    }
                }
            }
        }
        Ok(registry)
    }
}

/// What `lumbridge-computer model show` prints: the model exactly as every other module
/// sees it after `extends`/macro resolution, plus the argv it would launch with.
#[derive(Debug, Serialize)]
pub struct ModelShow<'a> {
    pub id: &'a str,
    pub model: &'a Model,
    /// `None` when the model has no `command` and its kind has no builder.
    pub argv: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub argv_error: Option<String>,
}

impl<'a> ModelShow<'a> {
    /// `mem_total_gb` is this box's pool size, as `proc::spawn` would read it,
    /// so a derived `--gpu-memory-utilization` shows exactly what would launch.
    pub fn new(id: &'a str, model: &'a Model, mem_total_gb: Option<f64>) -> Self {
        let (argv, argv_error) = match crate::proc::build_argv(model, mem_total_gb) {
            Ok(argv) => (Some(argv), None),
            Err(error) => (None, Some(format!("{error:#}"))),
        };
        ModelShow {
            id,
            model,
            argv,
            argv_error,
        }
    }
}

/// `lumbridge-computer model show <id> [--json]`.
pub fn show_model(root: &Path, id: &str, json: bool) -> Result<()> {
    let registry = Registry::load(root)?;
    let model = registry
        .models
        .get(id)
        .with_context(|| format!("no model '{id}' in {}/registry/models.yaml", root.display()))?;
    let show = ModelShow::new(id, model, crate::mem::read().ok().map(|m| m.total_gb));
    if json {
        println!("{}", serde_json::to_string_pretty(&show)?);
    } else {
        print!("{}", serde_yaml::to_string(&show)?);
    }
    Ok(())
}

/// `extends:` inheritance and `${NAME}` macros for registry models.
///
/// A model that has no `extends:` and no macro in any string ("plain") is
/// deserialized straight from the source text by serde_yaml, exactly as before
/// this feature existed, so its values (and its error messages, with line
/// numbers) are unchanged. Only models that need resolution go through
/// `serde_yaml::Value`: they are merged and expanded there, then re-emitted as
/// YAML and parsed again, which keeps serde_yaml's text coercions (an unquoted
/// `MAX_JOBS: 4` still reads as the string "4").
mod resolve {
    use super::{Model, Registry, Reservation};
    use anyhow::{anyhow, bail, Context, Result};
    use serde::de::{DeserializeSeed, Deserializer, IgnoredAny, MapAccess, Visitor};
    use serde_yaml::{Mapping, Value};
    use std::collections::{BTreeMap, BTreeSet};
    use std::fmt;

    const EXTENDS: &str = "extends";

    pub(super) fn parse_registry(s: &str) -> Result<Registry> {
        let doc: Value = serde_yaml::from_str(s)?;
        let raw: BTreeMap<String, Value> = match doc.get("models") {
            Some(models) => serde_yaml::from_value(models.clone())
                .context("`models` must map model ids to model definitions")?,
            None => BTreeMap::new(),
        };
        let plain: BTreeSet<String> = raw
            .iter()
            .filter(|(_, v)| v.get(EXTENDS).is_none() && !contains_macro(v))
            .map(|(id, _)| id.clone())
            .collect();

        let mut registry = serde_yaml::Deserializer::from_str(s)
            .deserialize_map(RegistryVisitor { plain: &plain })?;

        let mut memo = BTreeMap::new();
        for id in raw.keys().filter(|id| !plain.contains(*id)) {
            let mut merged = merged(id, &raw, &mut memo, &mut Vec::new())?;
            expand_macros(id, &mut merged)?;
            // Re-parse from text so resolved models get the same coercions as plain ones.
            let text = serde_yaml::to_string(&merged)?;
            let model: Model = serde_yaml::from_str(&text)
                .with_context(|| format!("registry model '{id}' (after extends/macros)"))?;
            registry.models.insert(id.clone(), model);
        }
        Ok(registry)
    }

    /// The model's mapping with its whole `extends` chain merged in, macros not
    /// yet expanded (an inherited `${PORT}` must take the child's port).
    fn merged(
        id: &str,
        raw: &BTreeMap<String, Value>,
        memo: &mut BTreeMap<String, Value>,
        chain: &mut Vec<String>,
    ) -> Result<Value> {
        if let Some(done) = memo.get(id) {
            return Ok(done.clone());
        }
        if chain.iter().any(|seen| seen == id) {
            chain.push(id.to_string());
            bail!("registry `extends` cycle: {}", chain.join(" -> "));
        }
        let mut own = raw
            .get(id)
            .cloned()
            .with_context(|| format!("no registry model '{id}'"))?;
        let base = match own.as_mapping_mut().and_then(|m| m.remove(EXTENDS)) {
            None => None,
            Some(Value::String(base)) => Some(base),
            Some(other) => bail!(
                "registry model '{id}': `extends` must be a model id, got {}",
                serde_yaml::to_string(&other)?.trim()
            ),
        };
        let resolved = match base {
            None => own,
            Some(base) => {
                if !raw.contains_key(&base) {
                    bail!("registry model '{id}' extends unknown model '{base}'");
                }
                chain.push(id.to_string());
                let mut parent = merged(&base, raw, memo, chain)?;
                chain.pop();
                merge(&mut parent, own);
                parent
            }
        };
        memo.insert(id.to_string(), resolved.clone());
        Ok(resolved)
    }

    /// Deep-merge `child` onto `base`: mappings merge key by key with the child
    /// winning, an explicit `null` in the child deletes the inherited key, and
    /// everything else (sequences included) is replaced wholesale.
    pub(super) fn merge(base: &mut Value, child: Value) {
        match (base, child) {
            (Value::Mapping(base), Value::Mapping(child)) => merge_mappings(base, child),
            (base, child) => *base = child,
        }
    }

    fn merge_mappings(base: &mut Mapping, child: Mapping) {
        for (key, value) in child {
            if value.is_null() {
                base.remove(&key);
                continue;
            }
            match base.get_mut(&key) {
                Some(existing) => merge(existing, value),
                None => {
                    base.insert(key, strip_nulls(value));
                }
            }
        }
    }

    /// A `null` that deletes nothing is dropped too, so `key: null` means
    /// "absent" whether or not the base had the key.
    fn strip_nulls(value: Value) -> Value {
        match value {
            Value::Mapping(map) => Value::Mapping(
                map.into_iter()
                    .filter(|(_, v)| !v.is_null())
                    .map(|(k, v)| (k, strip_nulls(v)))
                    .collect(),
            ),
            other => other,
        }
    }

    /// One token found while scanning a string for `${NAME}`.
    enum Piece<'a> {
        Text(&'a str),
        Macro(&'a str),
    }

    /// Split `s` into literal text and `${NAME}` macros, where NAME is
    /// `[A-Z_][A-Z0-9_]*`. Anything else containing `$` or `{` (shell `$HOME`,
    /// `${lower}`, JSON like `{"method":"mtp"}`) is literal. `$${NAME}` is the
    /// escape for a literal `${NAME}`.
    fn scan(s: &str) -> Vec<Piece<'_>> {
        let mut out = Vec::new();
        let bytes = s.as_bytes();
        let (mut i, mut start) = (0, 0);
        while i < bytes.len() {
            if bytes[i] == b'$' {
                let escaped = bytes.get(i + 1) == Some(&b'$');
                let at = if escaped { i + 1 } else { i };
                if let Some(len) = macro_len(&s[at..]) {
                    out.push(Piece::Text(&s[start..i]));
                    if escaped {
                        out.push(Piece::Text(&s[at..at + len]));
                    } else {
                        out.push(Piece::Macro(&s[at + 2..at + len - 1]));
                    }
                    i = at + len;
                    start = i;
                    continue;
                }
            }
            i += 1;
        }
        out.push(Piece::Text(&s[start..]));
        out
    }

    /// Length of a `${NAME}` at the start of `s`, if there is one.
    fn macro_len(s: &str) -> Option<usize> {
        let rest = s.strip_prefix("${")?;
        let end = rest.find('}')?;
        let name = &rest[..end];
        let mut chars = name.chars();
        let first = chars.next()?;
        let ok = (first.is_ascii_uppercase() || first == '_')
            && chars.all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
        ok.then_some(end + 3)
    }

    fn contains_macro(value: &Value) -> bool {
        match value {
            Value::String(s) => s.contains('$') && macro_or_escape(s),
            Value::Sequence(items) => items.iter().any(contains_macro),
            Value::Mapping(map) => map.values().any(contains_macro),
            Value::Tagged(tagged) => contains_macro(&tagged.value),
            _ => false,
        }
    }

    fn macro_or_escape(s: &str) -> bool {
        s.char_indices()
            .any(|(i, c)| c == '$' && macro_len(&s[i..]).is_some())
    }

    /// Expand `${MODEL_ID}` and `${PORT}` in every string value of a resolved model.
    fn expand_macros(id: &str, model: &mut Value) -> Result<()> {
        let port = match model.get("serve").and_then(|serve| serve.get("port")) {
            None | Some(Value::Null) => None,
            Some(Value::Number(n)) => Some(
                n.as_u64()
                    .and_then(|p| u16::try_from(p).ok())
                    .with_context(|| {
                        format!("registry model '{id}': serve.port {n} is not a port")
                    })?,
            ),
            Some(other) => bail!(
                "registry model '{id}': serve.port must be a number, got {}",
                serde_yaml::to_string(other)?.trim()
            ),
        };
        let lookup = |name: &str| -> Result<String> {
            match name {
                "MODEL_ID" => Ok(id.to_string()),
                "PORT" => port.map(|p| p.to_string()).ok_or_else(|| {
                    anyhow!("registry model '{id}' uses ${{PORT}} but has no serve.port")
                }),
                other => bail!(
                    "registry model '{id}' uses unknown macro ${{{other}}} \
                     (known: ${{MODEL_ID}}, ${{PORT}}; write $${{{other}}} for a literal)"
                ),
            }
        };
        expand_value(model, &lookup)
    }

    fn expand_value(value: &mut Value, lookup: &dyn Fn(&str) -> Result<String>) -> Result<()> {
        match value {
            Value::String(s) => {
                if s.contains('$') {
                    let mut out = String::with_capacity(s.len());
                    for piece in scan(s) {
                        match piece {
                            Piece::Text(text) => out.push_str(text),
                            Piece::Macro(name) => out.push_str(&lookup(name)?),
                        }
                    }
                    *s = out;
                }
            }
            Value::Sequence(items) => {
                for item in items {
                    expand_value(item, lookup)?;
                }
            }
            Value::Mapping(map) => {
                for (_, item) in map.iter_mut() {
                    expand_value(item, lookup)?;
                }
            }
            Value::Tagged(tagged) => expand_value(&mut tagged.value, lookup)?,
            _ => {}
        }
        Ok(())
    }

    /// Walks the top-level registry mapping with the real text deserializer, so
    /// plain models and reservations parse exactly as `serde_yaml::from_str`
    /// always parsed them. Models needing resolution are skipped here.
    struct RegistryVisitor<'a> {
        plain: &'a BTreeSet<String>,
    }

    impl<'de> Visitor<'de> for RegistryVisitor<'_> {
        type Value = Registry;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a model registry mapping")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Registry, A::Error> {
            use serde::de::Error;
            let mut api_version = None;
            let mut models = None;
            let mut reservations = None;
            let mut scopes = None;
            let mut admission = None;
            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "apiVersion" => api_version = Some(map.next_value::<String>()?),
                    "models" => models = Some(map.next_value_seed(PlainModels(self.plain))?),
                    "reservations" => {
                        reservations = Some(map.next_value::<BTreeMap<String, Reservation>>()?);
                    }
                    "scopes" => scopes = Some(map.next_value::<super::ScopesMode>()?),
                    "admission" => admission = Some(map.next_value::<super::AdmissionMode>()?),
                    _ => {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
            }
            Ok(Registry {
                api_version: api_version.ok_or_else(|| A::Error::missing_field("apiVersion"))?,
                models: models.ok_or_else(|| A::Error::missing_field("models"))?,
                reservations: reservations.unwrap_or_default(),
                scopes: scopes.unwrap_or_default(),
                admission: admission.unwrap_or_default(),
            })
        }
    }

    struct PlainModels<'a>(&'a BTreeSet<String>);

    impl<'de> DeserializeSeed<'de> for PlainModels<'_> {
        type Value = BTreeMap<String, Model>;

        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
            d.deserialize_map(self)
        }
    }

    impl<'de> Visitor<'de> for PlainModels<'_> {
        type Value = BTreeMap<String, Model>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a mapping of model ids to models")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some(id) = map.next_key::<String>()? {
                if self.0.contains(&id) {
                    let model = map.next_value::<Model>()?;
                    out.insert(id, model);
                } else {
                    map.next_value::<IgnoredAny>()?;
                }
            }
            Ok(out)
        }
    }
}

impl Scene {
    /// How this Scene keeps `id`: its `modes:` entry, else `resident`.
    pub fn mode_of(&self, id: &str) -> ModelMode {
        self.modes.get(id).copied().unwrap_or_default()
    }

    /// Two models of one scene can never serve on the same port: the second
    /// would fail to bind (or, worse, health would pass against the first).
    /// Same-port models stay legal across scenes, where they swap. Ids the
    /// registry lacks are skipped here; the unknown-id check reports them.
    pub fn check_ports(&self, registry: &Registry) -> Result<()> {
        let mut owners: BTreeMap<u16, &str> = BTreeMap::new();
        let mut clashes = Vec::new();
        for id in &self.models {
            let Some(port) = registry.models.get(id).and_then(|m| m.serve.port) else {
                continue;
            };
            match owners.get(&port) {
                Some(owner) if *owner != id.as_str() => {
                    clashes.push(format!("{owner} and {id} on :{port}"));
                }
                Some(_) => {}
                None => {
                    owners.insert(port, id);
                }
            }
        }
        if !clashes.is_empty() {
            anyhow::bail!(
                "Scene '{}' has models sharing a serve.port: {}",
                self.metadata.name,
                clashes.join("; ")
            );
        }
        Ok(())
    }
}

/// Parse one scene manifest file.
pub fn load_scene_file(path: &Path) -> Result<Scene> {
    let s = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_yaml::from_str(&s).with_context(|| format!("parsing {}", path.display()))
}

/// Load every `*.scene.yaml` under `<root>/scenes`, sorted by name.
pub fn load_scenes(root: &Path) -> Result<Vec<Scene>> {
    let dir = root.join("scenes");
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let is_scene = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.ends_with(".scene.yaml"))
            .unwrap_or(false);
        if is_scene {
            out.push(load_scene_file(&path)?);
        }
    }
    out.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));
    Ok(out)
}

pub fn find_scene(root: &Path, name: &str) -> Result<Scene> {
    load_scenes(root)?
        .into_iter()
        .find(|s| s.metadata.name == name)
        .with_context(|| format!("no scene named '{name}' in {}/scenes", root.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_default_scene_has_exactly_its_required_models() {
        use std::collections::BTreeSet;
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let scenes = load_scenes(root).unwrap();
        assert_eq!(scenes.len(), 1, "only the default scene should ship");
        let scene = &scenes[0];
        assert_eq!(scene.metadata.name, DEFAULT_SCENE);
        let registry = Registry::load(root).unwrap();
        let ids: BTreeSet<_> = scene.models.iter().collect();
        assert_eq!(ids.len(), scene.models.len(), "duplicate scene models");
        assert_eq!(ids, registry.models.keys().collect());
        let mut ports = BTreeSet::new();
        for model in registry.models.values() {
            assert!(model.footprint_gb.is_finite() && model.footprint_gb > 0.0);
            assert!(ports.insert(model.serve.port.expect("service needs a port")));
            assert!(model.health.is_some());
            assert!(model.health_contains.is_some());
            assert!(model.supervision.is_some());
        }
        let total: f64 = registry.models.values().map(|m| m.footprint_gb).sum();
        assert!(scene.budget_gb.unwrap() >= total + crate::pool::SAFETY_MARGIN_GB);
    }

    #[test]
    fn supervision_is_opt_in_and_defaults_are_bounded() {
        let plain: Model = serde_yaml::from_str(
            "name: plain\nfootprint_gb: 1\nserve:\n  kind: exec\n  port: 9000\n",
        )
        .unwrap();
        assert!(plain.supervision.is_none());

        let watched: Model = serde_yaml::from_str(
            "name: watched\nfootprint_gb: 1\nsupervision:\n  restart: always\nserve:\n  kind: exec\n  port: 9001\n",
        )
        .unwrap();
        let policy = watched.supervision.unwrap();
        assert_eq!(policy.restart, RestartPolicy::Always);
        assert_eq!(policy.check_interval_sec(), 30);
        assert_eq!(policy.failure_threshold(), 3);
        assert_eq!(policy.backoff_sec(), 60);
        assert_eq!(policy.max_backoff_sec(), 900);
        assert_eq!(policy.startup_timeout_sec(), 180);
        assert_eq!(policy.probe_timeout(), Duration::from_secs(5));
        assert!(policy.liveness.is_none(), "liveness must be opt-in");
    }

    #[test]
    fn liveness_defaults_are_bounded_and_unknown_kinds_are_refused() {
        let watched: Model = serde_yaml::from_str(
            "name: watched\nfootprint_gb: 1\nsupervision:\n  restart: always\n  liveness:\n    kind: vllm-metrics\nserve:\n  kind: vllm\n  port: 9001\n",
        )
        .unwrap();
        let liveness = watched.supervision.unwrap().liveness.unwrap();
        assert_eq!(liveness.kind, LivenessKind::VllmMetrics);
        assert_eq!(liveness.stall_sec(), 300);
        assert_eq!(liveness.metrics_path(), "/metrics");

        let hasty: Liveness = serde_yaml::from_str(
            "kind: vllm-metrics\nstall_sec: 5\nmetrics_path: http://elsewhere:80/metrics\n",
        )
        .unwrap();
        assert_eq!(hasty.stall_sec(), MIN_STALL_SEC);
        assert_eq!(hasty.metrics_path(), "/metrics");

        assert!(serde_yaml::from_str::<Liveness>("kind: completion\n").is_err());
        assert!(serde_yaml::from_str::<Liveness>("kind: vllm-metrics\nstal_sec: 300\n").is_err());
    }

    #[test]
    fn gateway_policy_is_optional_and_strict() {
        let plain: Model = serde_yaml::from_str(
            "name: plain\nfootprint_gb: 1\nserve:\n  kind: exec\n  port: 9000\n",
        )
        .unwrap();
        assert!(plain.gateway.upstream.is_none());
        assert!(!plain.gateway.rewrite_model);

        let traced: Model = serde_yaml::from_str(
            "name: brain\nfootprint_gb: 1\ngateway:\n  upstream: 127.0.0.1:8041\n  rewrite_model: true\nserve:\n  kind: vllm\n  port: 8001\n",
        )
        .unwrap();
        assert_eq!(traced.gateway.upstream.as_deref(), Some("127.0.0.1:8041"));
        assert!(traced.gateway.rewrite_model);

        assert!(serde_yaml::from_str::<ModelGateway>("upstrem: 127.0.0.1:8041\n").is_err());
    }

    #[test]
    fn zero_intervals_cannot_create_a_hot_restart_loop() {
        let watched: Model = serde_yaml::from_str(
            "name: watched\nfootprint_gb: 1\nsupervision:\n  restart: always\n  check_interval_sec: 0\n  failure_threshold: 0\n  backoff_sec: 0\n  max_backoff_sec: 0\n  startup_timeout_sec: 0\nserve:\n  kind: exec\n  port: 9001\n",
        )
        .unwrap();
        let policy = watched.supervision.unwrap();
        assert_eq!(policy.check_interval_sec(), 1);
        assert_eq!(policy.failure_threshold(), 1);
        assert_eq!(policy.backoff_sec(), 1);
        assert_eq!(policy.max_backoff_sec(), 1);
        assert_eq!(policy.startup_timeout_sec(), 1);
    }

    fn box_fixture() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/box")
    }

    /// The golden file is the resolved form of a registry that uses neither
    /// `extends` nor macros: the loader must keep producing exactly the same
    /// `Model` values, field for field.
    #[test]
    fn the_fixture_registry_resolves_exactly_to_its_golden() {
        let root = box_fixture();
        let registry = Registry::load(&root).unwrap();
        assert!(registry.models.len() >= 20, "fixture looks truncated");
        let golden = fs::read_to_string(root.join("models.resolved.debug.txt")).unwrap();
        // `pool` is not in the golden. No fixture model names one, so each must carry
        // the default, and with that one line removed nothing else may differ.
        let resolved = format!("{:#?}\n", registry.models);
        let pool_line = "        pool: Unified,\n";
        assert_eq!(resolved.matches(pool_line).count(), registry.models.len());
        let resolved = resolved.replace(pool_line, "");
        // So does `serve.sleep`; no fixture model sleeps.
        let sleep_line = "            sleep: None,\n";
        assert_eq!(resolved.matches(sleep_line).count(), registry.models.len());
        let resolved = resolved.replace(sleep_line, "");
        assert!(
            resolved == golden,
            "resolved fixture registry drifted from the golden"
        );
    }

    /// `extends` takes the re-parse path, so this proves that path reads every
    /// real model exactly as the plain path does.
    #[test]
    fn a_bare_extends_of_any_fixture_model_resolves_identically_to_its_base() {
        let root = box_fixture();
        let mut text = fs::read_to_string(root.join("registry/models.yaml")).unwrap();
        let ids: Vec<String> = Registry::load(&root).unwrap().models.into_keys().collect();
        text.push('\n');
        for id in &ids {
            text.push_str(&format!("  copy-of-{id}:\n    extends: {id}\n"));
        }
        let registry = Registry::parse(&text).unwrap();
        for id in &ids {
            assert_eq!(
                format!("{:?}", registry.models[id]),
                format!("{:?}", registry.models[&format!("copy-of-{id}")]),
                "copy of '{id}' differs"
            );
        }
    }

    #[test]
    fn two_models_of_one_scene_on_the_same_port_are_refused_by_name() {
        let r = registry(&format!(
            "{BASE}  twin:\n    extends: base\n  other:\n    extends: base\n    serve:\n      port: 8002\n"
        ))
        .unwrap();
        let scene: Scene = serde_yaml::from_str(
            "apiVersion: lumbridge/v1\nmetadata:\n  name: clash\nmodels: [base, other, twin]\n",
        )
        .unwrap();
        let msg = format!("{:#}", scene.check_ports(&r).unwrap_err());
        assert!(msg.contains("base and twin on :8001"), "{msg}");
        assert!(!msg.contains("other"), "{msg}");
        let activation = crate::lifecycle::validate_scene(&r, &scene).unwrap_err();
        assert!(format!("{activation:#}").contains("sharing a serve.port"));
        let swap: Scene = serde_yaml::from_str(
            "apiVersion: lumbridge/v1\nmetadata:\n  name: ok\nmodels: [base, other]\n",
        )
        .unwrap();
        swap.check_ports(&r).unwrap();
    }

    #[test]
    fn every_fixture_scene_passes_the_port_check() {
        let root = box_fixture();
        let registry = Registry::load(&root).unwrap();
        let scenes = load_scenes(&root).unwrap();
        assert!(scenes.len() >= 15, "fixture scenes look truncated");
        for scene in &scenes {
            scene.check_ports(&registry).unwrap();
        }
    }

    fn two_brains() -> (Registry, Scene) {
        let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
        (
            Registry::load_file(&examples.join("two-brains.registry.yaml")).unwrap(),
            load_scene_file(&examples.join("two-brains.scene.yaml")).unwrap(),
        )
    }

    #[test]
    fn the_two_brains_example_resolves_brain_alt_as_a_delta_on_the_main_brain() {
        let (r, _) = two_brains();
        let (stock, alt) = (&r.models["brain"], &r.models["brain-alt"]);
        let alt_weights = "/home/sam/models/brain-alt-35b-a3b-nvfp4";
        assert_eq!(alt.serve.port, Some(8002));
        assert_eq!(
            alt.health.as_deref(),
            Some("http://localhost:8002/v1/models")
        );
        assert_eq!(
            stock.health.as_deref(),
            Some("http://localhost:8001/v1/models")
        );
        assert_eq!(alt.serve.weights.as_deref(), Some(alt_weights));
        assert_eq!(
            alt.health_contains.as_deref(),
            Some(format!("\"root\":\"{alt_weights}\"").as_str())
        );
        assert_eq!(alt.serve.served_name, ["brain-alt"]);
        assert_eq!(
            format!("{:?}", alt.serve.args),
            format!("{:?}", stock.serve.args)
        );
        assert_eq!(alt.serve.env, stock.serve.env);
        assert_eq!(alt.serve.executable, stock.serve.executable);
        assert_eq!(alt.footprint_gb, stock.footprint_gb);
        let probe = alt.supervision.as_ref().unwrap().liveness.as_ref().unwrap();
        assert!(
            probe.chat_probe.is_some(),
            "liveness and its chat probe are inherited"
        );
    }

    /// The example's shared entries must stay the fixture's, not drift from them.
    #[test]
    fn the_two_brains_examples_shared_models_match_the_fixture_exactly() {
        let (example, _) = two_brains();
        let fixture = Registry::load(&box_fixture()).unwrap();
        for id in ["brain", "embed", "rerank"] {
            assert_eq!(
                format!("{:?}", example.models[id]),
                format!("{:?}", fixture.models[id]),
                "example '{id}' drifted from the fixture"
            );
        }
    }

    #[test]
    fn the_two_brains_scene_validates_within_its_100_gb_budget() {
        let (r, scene) = two_brains();
        crate::lifecycle::validate_scene(&r, &scene).unwrap();
        let total: f64 = scene
            .models
            .iter()
            .map(|id| r.models[id].footprint_gb)
            .sum();
        assert_eq!(total, 88.0);
        assert!(total + crate::pool::SAFETY_MARGIN_GB <= scene.budget_gb.unwrap());
        assert_eq!(scene.budget_gb, Some(100.0));
        let mut ports: Vec<u16> = scene
            .models
            .iter()
            .filter_map(|id| r.models[id].serve.port)
            .collect();
        ports.sort_unstable();
        assert_eq!(ports, [8001, 8002, 8012, 8015]);
    }

    fn registry(models: &str) -> Result<Registry> {
        Registry::parse(&format!("apiVersion: lumbridge/v1\nmodels:\n{models}"))
    }

    const BASE: &str = r#"
  base:
    name: Base
    footprint_gb: 40
    health: http://localhost:${PORT}/v1/models
    health_contains: "${MODEL_ID}"
    supervision:
      restart: always
      check_interval_sec: 15
      liveness:
        kind: vllm-metrics
    serve:
      kind: vllm
      port: 8001
      weights: /models/base
      served_name: [brain, "${MODEL_ID}"]
      args:
        max-model-len: 131072
        enforce-eager: true
        speculative-config: '{"method":"mtp","num_speculative_tokens":3}'
      env:
        MAX_JOBS: 4
        CUTE_DSL_ARCH: sm_121a
"#;

    #[test]
    fn a_child_inherits_everything_and_its_own_keys_win_in_nested_maps() {
        let r = registry(&format!(
            "{BASE}  child:\n    extends: base\n    footprint_gb: 41\n    serve:\n      port: 8002\n      args:\n        max-model-len: 65536\n      env:\n        EXTRA: x\n"
        ))
        .unwrap();
        let child = &r.models["child"];
        assert_eq!(child.name, "Base");
        assert_eq!(child.footprint_gb, 41.0);
        assert_eq!(child.serve.port, Some(8002));
        assert_eq!(child.serve.weights.as_deref(), Some("/models/base"));
        assert_eq!(
            child.serve.args["max-model-len"],
            serde_yaml::Value::from(65536)
        );
        assert_eq!(
            child.serve.args["enforce-eager"],
            serde_yaml::Value::from(true)
        );
        assert_eq!(
            child.serve.env["MAX_JOBS"], "4",
            "text coercion survives the merge"
        );
        assert_eq!(child.serve.env["CUTE_DSL_ARCH"], "sm_121a");
        assert_eq!(child.serve.env["EXTRA"], "x");
        let supervision = child.supervision.as_ref().unwrap();
        assert_eq!(supervision.check_interval_sec, 15);
        assert!(supervision.liveness.is_some());
    }

    #[test]
    fn sequences_are_replaced_rather_than_concatenated() {
        let r = registry(&format!(
            "{BASE}  child:\n    extends: base\n    serve:\n      served_name: [brain-alt]\n"
        ))
        .unwrap();
        assert_eq!(r.models["child"].serve.served_name, vec!["brain-alt"]);
    }

    #[test]
    fn model_show_prints_the_resolved_model_and_its_argv() {
        let r = registry(&format!(
            "{BASE}  child:\n    extends: base\n    serve:\n      port: 8002\n"
        ))
        .unwrap();
        let show = ModelShow::new("child", &r.models["child"], None);
        let yaml = serde_yaml::to_string(&show).unwrap();
        assert!(
            yaml.contains("health: http://localhost:8002/v1/models"),
            "{yaml}"
        );
        assert!(!yaml.contains("extends"), "{yaml}");
        let json: serde_json::Value = serde_json::to_value(&show).unwrap();
        let argv: Vec<&str> = json["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap())
            .collect();
        assert_eq!(&argv[..3], ["vllm", "serve", "/models/base"]);
        assert!(argv.windows(2).any(|w| w == ["--port", "8002"]));
        assert!(argv
            .windows(3)
            .any(|w| w == ["--served-model-name", "brain", "child"]));
        assert_eq!(json["model"]["supervision"]["restart"], "always");
    }

    #[test]
    fn an_explicit_null_deletes_the_inherited_key() {
        let r = registry(&format!(
            "{BASE}  child:\n    extends: base\n    health_contains: null\n    supervision:\n      liveness: null\n    serve:\n      args:\n        enforce-eager: null\n        not-inherited: null\n"
        ))
        .unwrap();
        let child = &r.models["child"];
        assert!(child.health_contains.is_none());
        assert!(child.supervision.as_ref().unwrap().liveness.is_none());
        assert!(!child.serve.args.contains_key("enforce-eager"));
        assert!(!child.serve.args.contains_key("not-inherited"));
        assert!(child.serve.args.contains_key("max-model-len"));
    }

    #[test]
    fn chains_resolve_with_the_nearest_definition_winning() {
        let r = registry(&format!(
            "{BASE}  mid:\n    extends: base\n    name: Mid\n    footprint_gb: 20\n  leaf:\n    extends: mid\n    footprint_gb: 10\n"
        ))
        .unwrap();
        let leaf = &r.models["leaf"];
        assert_eq!((leaf.name.as_str(), leaf.footprint_gb), ("Mid", 10.0));
        assert_eq!(leaf.serve.weights.as_deref(), Some("/models/base"));
        assert_eq!(r.models["mid"].footprint_gb, 20.0);
    }

    #[test]
    fn an_extends_cycle_is_refused_and_names_every_model_in_it() {
        let err = registry(&format!(
            "{BASE}  a:\n    extends: b\n  b:\n    extends: c\n  c:\n    extends: a\n"
        ))
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("cycle"), "{msg}");
        assert!(msg.contains("a -> b -> c -> a"), "{msg}");
        let own = registry("  me:\n    extends: me\n").unwrap_err();
        assert!(format!("{own:#}").contains("me -> me"));
    }

    #[test]
    fn an_unknown_base_is_refused_and_names_both_models() {
        let err = registry(&format!("{BASE}  child:\n    extends: nowhere\n")).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("'child'") && msg.contains("'nowhere'"),
            "{msg}"
        );
    }

    #[test]
    fn port_and_model_id_macros_expand_per_model_after_the_merge() {
        let r = registry(&format!(
            "{BASE}  child:\n    extends: base\n    serve:\n      port: 8002\n"
        ))
        .unwrap();
        let (base, child) = (&r.models["base"], &r.models["child"]);
        assert_eq!(
            base.health.as_deref(),
            Some("http://localhost:8001/v1/models")
        );
        assert_eq!(
            child.health.as_deref(),
            Some("http://localhost:8002/v1/models")
        );
        assert_eq!(base.health_contains.as_deref(), Some("base"));
        assert_eq!(child.health_contains.as_deref(), Some("child"));
        assert_eq!(child.serve.served_name, vec!["brain", "child"]);
        assert_eq!(
            child.serve.args["speculative-config"],
            serde_yaml::Value::from(r#"{"method":"mtp","num_speculative_tokens":3}"#),
            "JSON braces are not macros"
        );
    }

    #[test]
    fn only_uppercase_braced_names_are_macros_and_double_dollar_escapes_one() {
        let r = registry(
            "  m:\n    name: m\n    footprint_gb: 1\n    serve:\n      kind: exec\n      port: 9000\n      command: [sh, -c, 'echo $HOME ${lower} $${PORT} ${PORT} $$ ~/x']\n",
        )
        .unwrap();
        assert_eq!(
            r.models["m"].serve.command[2],
            "echo $HOME ${lower} ${PORT} 9000 $$ ~/x"
        );
    }

    #[test]
    fn an_unknown_macro_is_a_load_error_not_left_in_place() {
        let err = registry(
            "  m:\n    name: m\n    footprint_gb: 1\n    health: http://localhost:${PROT}/\n    serve:\n      kind: exec\n      port: 9000\n",
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("${PROT}") && msg.contains("'m'"), "{msg}");
    }

    #[test]
    fn the_port_macro_without_a_port_is_a_load_error() {
        let err = registry(
            "  m:\n    name: m\n    footprint_gb: 1\n    health: http://localhost:${PORT}/\n    serve:\n      kind: exec\n",
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("no serve.port"));
    }

    #[test]
    fn a_resolved_model_that_is_still_incomplete_names_the_model() {
        let err = registry(&format!(
            "{BASE}  child:\n    extends: base\n    serve: null\n"
        ))
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("'child'") && msg.contains("serve"), "{msg}");
    }

    #[test]
    fn probe_timeout_is_configurable_but_never_hundreds_of_milliseconds_or_minutes() {
        let parse = |yaml: &str| serde_yaml::from_str::<Supervision>(yaml).unwrap();
        assert_eq!(
            parse("restart: always\nprobe_timeout_ms: 8000\n").probe_timeout(),
            Duration::from_secs(8)
        );
        assert_eq!(
            parse("restart: always\nprobe_timeout_ms: 300\n").probe_timeout(),
            Duration::from_millis(MIN_PROBE_TIMEOUT_MS)
        );
        assert_eq!(
            parse("restart: always\nprobe_timeout_ms: 600000\n").probe_timeout(),
            Duration::from_millis(MAX_PROBE_TIMEOUT_MS)
        );
        let plain: Model = serde_yaml::from_str(
            "name: plain\nfootprint_gb: 1\nserve:\n  kind: exec\n  port: 9000\n",
        )
        .unwrap();
        assert_eq!(plain.probe_timeout(), Duration::from_secs(5));
    }
}
