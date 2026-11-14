//! The one place that understands serving engines.
//!
//! Lumbridge Computer runs models on several engines (vLLM, SGLang, llama.cpp, the NeMo
//! speech server). Each names its load metrics differently, and names drift
//! across versions. Supervision, telemetry and (later) scheduling all ask the
//! same question, "how busy is this model right now?", so the answer lives
//! here: [`detect`] says which engine a registry model runs, [`parse_load`]
//! turns that engine's Prometheus text into an [`EngineLoad`], and [`scrape`]
//! does both against the model's own local port.
//!
//! The rule everywhere: **unknown means `None`, never idle.** A missing gauge,
//! a renamed metric or an unreachable endpoint must never read as "0 running",
//! because idle is what lets a supervisor send probes, a scheduler admit
//! work, and the watchdog pick a victim.

pub mod http;
pub mod prom;
pub mod sleep;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::config::Model;
use prom::Sample;

/// Upper bound on one scrape. Telemetry samples every few seconds and liveness
/// runs on a one-second loop; a stuck engine must not stall either.
pub const DEFAULT_SCRAPE_TIMEOUT: Duration = Duration::from_secs(2);

/// The serving engine behind a registry model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Engine {
    Vllm,
    Sglang,
    LlamaCpp,
    NemoSpeech,
    /// Not recognised from the registry. Parsing then sniffs the metric names,
    /// so a container whose argv hides the engine still reports load.
    Generic,
}

impl Engine {
    pub fn as_str(self) -> &'static str {
        match self {
            Engine::Vllm => "vllm",
            Engine::Sglang => "sglang",
            Engine::LlamaCpp => "llama-cpp",
            Engine::NemoSpeech => "nemo-speech",
            Engine::Generic => "generic",
        }
    }

    /// `Generic` resolved by the metric names actually exposed; a concrete
    /// engine is returned unchanged.
    pub fn resolve(self, samples: &[Sample]) -> Engine {
        match self {
            Engine::Generic => sniff(samples).unwrap_or(Engine::Generic),
            known => known,
        }
    }
}

impl std::fmt::Display for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How busy an engine is, normalised across engines and versions.
///
/// `running`/`waiting` are summed over every label set (vLLM data-parallel
/// `engine="0"`, `engine="1"`; several `model_name`s; vllm-omni stages).
/// `kv_usage` is a 0..1 fraction and takes the *fullest* cache across label
/// sets, since summing ratios is meaningless and the fullest one is what
/// limits admission. Token totals are lifetime counters (they reset when the
/// engine restarts) and are `None` when the engine does not export them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EngineLoad {
    pub running: u32,
    pub waiting: u32,
    pub kv_usage: Option<f64>,
    pub prompt_tokens_total: Option<f64>,
    pub generation_tokens_total: Option<f64>,
}

/// One reading of an engine's work counters, for the supervision liveness rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngineProgress {
    pub running: f64,
    pub waiting: f64,
    /// Requests the engine holds: running plus waiting.
    pub in_flight: f64,
    /// Prompt plus generation tokens counted since the engine started.
    pub tokens: f64,
}

/// Metric names for one engine family.
struct Names {
    running: &'static str,
    waiting: &'static str,
    /// First one present wins: newer name first, then older spellings.
    kv_usage: &'static [&'static str],
    prompt_tokens: Option<&'static str>,
    generation_tokens: Option<&'static str>,
}

const VLLM: Names = Names {
    running: "vllm:num_requests_running",
    waiting: "vllm:num_requests_waiting",
    // `gpu_cache_usage_perc` was renamed in vLLM v1 (0.9-0.10). Both are
    // 0..1 fractions despite the "perc" suffix.
    kv_usage: &["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    prompt_tokens: Some("vllm:prompt_tokens_total"),
    generation_tokens: Some("vllm:generation_tokens_total"),
};

const SGLANG: Names = Names {
    running: "sglang:num_running_reqs",
    waiting: "sglang:num_queue_reqs",
    kv_usage: &["sglang:token_usage"],
    prompt_tokens: Some("sglang:prompt_tokens_total"),
    generation_tokens: Some("sglang:generation_tokens_total"),
};

/// llama.cpp `llama-server --metrics`.
const LLAMA_CPP: Names = Names {
    running: "llamacpp:requests_processing",
    waiting: "llamacpp:requests_deferred",
    kv_usage: &["llamacpp:kv_cache_usage_ratio"],
    prompt_tokens: Some("llamacpp:prompt_tokens_total"),
    generation_tokens: Some("llamacpp:tokens_predicted_total"),
};

/// The NeMo speech server's `speech_*` exporter. Its only live gauge today is
/// `speech_stream_active` (open websockets); HTTP transcriptions in progress
/// are invisible, so without an in-flight gauge the load is unknown.
const NEMO_SPEECH: Names = Names {
    running: "speech_requests_in_flight",
    waiting: "speech_requests_waiting",
    kv_usage: &[],
    prompt_tokens: None,
    generation_tokens: None,
};

fn names(engine: Engine) -> Option<&'static Names> {
    match engine {
        Engine::Vllm => Some(&VLLM),
        Engine::Sglang => Some(&SGLANG),
        Engine::LlamaCpp => Some(&LLAMA_CPP),
        Engine::NemoSpeech => Some(&NEMO_SPEECH),
        Engine::Generic => None,
    }
}

/// Which engine is behind a registry model.
///
/// In order: an explicit `serve.engine`, then `serve.kind`, then the launch
/// argv (`vllm serve`, `sglang.launch_server`/`sglang serve`, `llama-server`,
/// `nemo-speech`, or a `docker run` image named after vLLM/SGLang).
pub fn detect(model: &Model) -> Engine {
    if let Some(engine) = model.serve.engine {
        return engine;
    }
    match model.serve.kind.as_str() {
        "vllm" => return Engine::Vllm,
        "sglang" => return Engine::Sglang,
        "llama-cpp" | "llamacpp" | "llama-server" => return Engine::LlamaCpp,
        "nemo-speech" => return Engine::NemoSpeech,
        _ => {}
    }
    let argv: Vec<&str> = model
        .serve
        .executable
        .iter()
        .chain(model.serve.command.iter())
        .map(String::as_str)
        .collect();
    detect_argv(&argv)
}

fn basename(arg: &str) -> &str {
    arg.rsplit('/').next().unwrap_or(arg)
}

fn detect_argv(argv: &[&str]) -> Engine {
    let launcher = |a: &str, b: Option<&&str>| {
        let next = b.copied().unwrap_or_default();
        match (basename(a), next) {
            ("vllm", "serve") => Some(Engine::Vllm),
            (_, _) if a.starts_with("vllm.entrypoints") => Some(Engine::Vllm),
            ("sglang", "serve") => Some(Engine::Sglang),
            (_, _) if a.starts_with("sglang.launch_server") => Some(Engine::Sglang),
            ("llama-server", _) => Some(Engine::LlamaCpp),
            ("nemo-speech", _) => Some(Engine::NemoSpeech),
            _ => None,
        }
    };
    for (index, arg) in argv.iter().enumerate() {
        if let Some(engine) = launcher(arg, argv.get(index + 1)) {
            return engine;
        }
    }
    // `docker run ... example/vllm-dock:tag` runs the image's entrypoint, so
    // the engine is only visible in the image name. Paths (`-v /x/vllm:/y`) and
    // flags are skipped: an SGLang container may well mount a vLLM cache.
    if argv
        .first()
        .is_some_and(|program| matches!(basename(program), "docker" | "podman"))
    {
        for arg in argv {
            if arg.starts_with(['-', '/', '~', '.']) || !arg.contains('/') {
                continue;
            }
            let repository = arg.split(['@', ':']).next().unwrap_or(arg);
            let image = basename(repository);
            if image.contains("sglang") {
                return Engine::Sglang;
            }
            if image.contains("vllm") {
                return Engine::Vllm;
            }
        }
    }
    Engine::Generic
}

/// Guess the engine from the metric names it exports.
pub fn sniff(samples: &[Sample]) -> Option<Engine> {
    [
        Engine::Vllm,
        Engine::Sglang,
        Engine::LlamaCpp,
        Engine::NemoSpeech,
    ]
    .into_iter()
    .find(|engine| {
        names(*engine).is_some_and(|names| {
            samples
                .iter()
                .any(|s| s.name == names.running && s.value.is_finite())
        })
    })
}

/// Raw sums in f64, before rounding to the public contract. Liveness compares
/// these exact values, so they stay unrounded.
struct RawLoad {
    running: f64,
    waiting: Option<f64>,
    kv_usage: Option<f64>,
    prompt_tokens: Option<f64>,
    generation_tokens: Option<f64>,
}

fn raw_load(engine: Engine, samples: &[Sample]) -> Option<RawLoad> {
    let engine = engine.resolve(samples);
    let names = names(engine)?;
    // SGLang can break its gauges down by request priority as well as
    // reporting the total; counting both would double the load.
    let keep = |sample: &Sample| {
        engine != Engine::Sglang || sample.labels.get("priority").is_none_or(String::is_empty)
    };
    let mut running = prom::sum_where(samples, names.running, keep)?;
    if engine == Engine::NemoSpeech {
        running += prom::sum(samples, "speech_stream_active").unwrap_or(0.0);
    }
    Some(RawLoad {
        running,
        waiting: prom::sum_where(samples, names.waiting, keep),
        kv_usage: names
            .kv_usage
            .iter()
            .find_map(|name| prom::max_where(samples, name, keep)),
        prompt_tokens: names.prompt_tokens.and_then(|n| prom::sum(samples, n)),
        generation_tokens: names.generation_tokens.and_then(|n| prom::sum(samples, n)),
    })
}

fn to_count(value: f64) -> u32 {
    // `as` saturates and maps NaN to 0; NaN was already filtered out.
    value.max(0.0).round() as u32
}

/// Normalised load from already-parsed samples. `None` when the engine's
/// running gauge is absent: we cannot tell idle from unknown. A missing
/// waiting gauge alongside a present running gauge reads as 0 waiting, the
/// same rule liveness has always used, because every engine we know exports
/// the pair together and an engine without a queue has nothing waiting.
pub fn load_from_samples(engine: Engine, samples: &[Sample]) -> Option<EngineLoad> {
    let raw = raw_load(engine, samples)?;
    Some(EngineLoad {
        running: to_count(raw.running),
        waiting: to_count(raw.waiting.unwrap_or(0.0)),
        kv_usage: raw.kv_usage.map(|v| v.clamp(0.0, 1.0)),
        prompt_tokens_total: raw.prompt_tokens,
        generation_tokens_total: raw.generation_tokens,
    })
}

/// Normalise one engine's Prometheus text. See [`load_from_samples`].
// The health module's contract: liveness
// switches to it at merge, so the binary does not call it yet.
#[allow(dead_code)]
pub fn parse_load(engine: Engine, prometheus_text: &str) -> Option<EngineLoad> {
    load_from_samples(engine, &prom::parse(prometheus_text))
}

/// The liveness reading: requests held and tokens counted. `None` unless both
/// the running gauge and the generation counter are present.
pub fn progress(engine: Engine, prometheus_text: &str) -> Option<EngineProgress> {
    let raw = raw_load(engine, &prom::parse(prometheus_text))?;
    let waiting = raw.waiting.unwrap_or(0.0);
    Some(EngineProgress {
        running: raw.running,
        waiting,
        in_flight: raw.running + waiting,
        tokens: raw.generation_tokens? + raw.prompt_tokens.unwrap_or(0.0),
    })
}

/// vLLM v1 bumps `prompt_tokens_total` only when a request emits its first
/// token, so a long prefill moves neither counter until it finishes. The stall
/// window, not this parser, is what keeps that case from looking wedged.
pub fn vllm_progress(prometheus_text: &str) -> Option<EngineProgress> {
    progress(Engine::Vllm, prometheus_text)
}

/// Prometheus path on the model's own port: the liveness block's
/// `metrics_path` when set (already held to a plain local path), else `/metrics`.
fn metrics_path(model: &Model) -> &str {
    model
        .supervision
        .as_ref()
        .and_then(|s| s.liveness.as_ref())
        .map(|l| l.metrics_path())
        .unwrap_or("/metrics")
}

/// `http://127.0.0.1:<port>/metrics` for a model; always loopback.
pub fn metrics_url(model: &Model) -> Result<String> {
    let port = model
        .serve
        .port
        .with_context(|| format!("model '{}' has no serve.port to scrape", model.name))?;
    Ok(format!("http://127.0.0.1:{port}{}", metrics_path(model)))
}

/// Fetch a model's raw `/metrics` text (telemetry also needs the counters).
pub fn fetch_metrics(model: &Model, timeout: Duration) -> Result<String> {
    fetch_url(&metrics_url(model)?, timeout)
}

/// GET a metrics URL with one budget: `timeout` (capped at
/// [`DEFAULT_SCRAPE_TIMEOUT`]) for the read, and at most half a second of it
/// to connect, since a local port either accepts at once or is closed.
pub fn fetch_url(url: &str, timeout: Duration) -> Result<String> {
    let timeout = timeout.clamp(Duration::from_millis(50), DEFAULT_SCRAPE_TIMEOUT);
    http::fetch_http_within(url, timeout.min(Duration::from_millis(500)), timeout)
}

/// Scrape a model's load from its own port with the default timeout.
///
/// `Err` when the endpoint cannot be read (no port, refused, non-200,
/// timeout); `Ok(None)` when it answered but carries no load metrics this
/// engine is known to export. Both mean "unknown", never idle.
// The health module's contract: liveness
// switches to it at merge, so the binary does not call it yet.
#[allow(dead_code)]
pub fn scrape(model: &Model) -> Result<Option<EngineLoad>> {
    scrape_within(model, DEFAULT_SCRAPE_TIMEOUT)
}

/// [`scrape`] with a caller-chosen timeout, capped at [`DEFAULT_SCRAPE_TIMEOUT`].
// The health module's contract: liveness
// switches to it at merge, so the binary does not call it yet.
#[allow(dead_code)]
pub fn scrape_within(model: &Model, timeout: Duration) -> Result<Option<EngineLoad>> {
    let text = fetch_metrics(model, timeout)?;
    Ok(parse_load(detect(model), &text))
}

/// The concurrency ceiling a model was launched with, from its argv or `args`:
/// SGLang `--max-running-requests`, vLLM `--max-num-seqs`, llama.cpp
/// `-np`/`--parallel`. 0 when not configured.
pub fn configured_concurrency(model: &Model) -> u32 {
    let command = &model.serve.command;
    for flag in [
        "--max-running-requests",
        "--max-num-seqs",
        "--parallel",
        "-np",
    ] {
        if let Some(value) = command
            .iter()
            .position(|arg| arg == flag)
            .and_then(|index| command.get(index + 1))
            .and_then(|value| value.parse().ok())
        {
            return value;
        }
    }
    for key in ["max-running-requests", "max-num-seqs", "parallel"] {
        if let Some(value) = model
            .serve
            .args
            .get(key)
            .and_then(serde_yaml::Value::as_u64)
        {
            return value.min(u32::MAX as u64) as u32;
        }
    }
    0
}

#[cfg(test)]
mod tests;
