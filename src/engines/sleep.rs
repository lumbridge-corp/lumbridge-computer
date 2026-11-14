//! Engine sleep adapter (M3): put a model's engine to sleep and wake it, on its
//! own loopback port.
//!
//! Only vLLM sleeps, only at level 2, and only when the registry opts in with
//! `serve.sleep`: that adds `--enable-sleep-mode`, Lumbridge Computer's worker extension
//! (`assets/vllm/lumbridge_computer_sleep_ext.py`), `VLLM_SERVER_DEV_MODE=1` and a
//! 127.0.0.1 bind to the launch (see `proc::build_argv` and `proc::launch_env`),
//! which exposes `POST /sleep`, `POST /wake_up`, `POST /collective_rpc` and
//! `GET /is_sleeping`. Every other engine (NeMo, llama.cpp, shims) can only be
//! stopped and cold-started.
//!
//! The wake procedure is data (`serve.sleep.wake` then `serve.sleep.reload`,
//! `config::SleepConfig`); its default is the three measured steps in
//! `docs/sleep.md`. Whatever the steps, a
//! `/collective_rpc` reply is checked, and a model with a speculative (MTP)
//! config must report its drafter reloaded, or the wake is a failure.

use anyhow::{bail, Context, Result};
use std::time::{Duration, Instant};

use super::Engine;
use crate::config::{HttpStep, Model};

/// A loopback port either accepts at once or is closed.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(500);
/// `GET /is_sleeping` answers from the API server without touching the GPU.
const QUERY_TIMEOUT: Duration = Duration::from_secs(5);

/// True when `model` can sleep: a vLLM engine launched by the `vllm` builder
/// whose registry entry has `serve.sleep` (the registry refuses `serve.sleep`
/// with a hand-written `command`, which Lumbridge Computer can't add the extension to).
pub fn supports_sleep(model: &Model) -> bool {
    model.serve.sleep.is_some()
        && super::detect(model) == Engine::Vllm
        && model.serve.command.is_empty()
}

/// True when the model runs a speculative decoder (MTP or a draft model), whose
/// weights a level-2 wake must reload too.
pub fn has_speculative_config(model: &Model) -> bool {
    ["speculative-config", "speculative_config"]
        .iter()
        .filter_map(|key| model.serve.args.get(*key))
        .any(|value| match value {
            serde_yaml::Value::Null | serde_yaml::Value::Bool(false) => false,
            serde_yaml::Value::String(s) => !s.trim().is_empty(),
            _ => true,
        })
}

fn base_url(model: &Model) -> Result<String> {
    let port = model
        .serve
        .port
        .with_context(|| format!("model '{}' has no serve.port", model.name))?;
    Ok(format!("http://127.0.0.1:{port}"))
}

/// One call; `Err` for transport errors, `Ok((status, body))` otherwise.
fn send(model: &Model, step: &HttpStep, timeout: Duration) -> Result<(u16, String)> {
    let url = format!("{}{}", base_url(model)?, step.path);
    super::http::request_within(
        &step.method,
        &url,
        step.body.as_deref(),
        CONNECT_TIMEOUT,
        timeout,
    )
    .with_context(|| format!("{} {}", step.method, step.path))
}

fn call(model: &Model, step: &HttpStep, timeout: Duration) -> Result<String> {
    let (status, body) = send(model, step, timeout)?;
    if !(200..300).contains(&status) {
        bail!(
            "{} {} returned HTTP {status}: {}",
            step.method,
            step.path,
            body.chars().take(200).collect::<String>()
        );
    }
    Ok(body)
}

/// `POST /sleep?level=N&mode=wait`, then confirm with `GET /is_sleeping`. vLLM
/// answers the sleep call only once the memory has been released.
///
/// `mode=wait` (vLLM ≥ 0.27) lets in-flight requests finish instead of the
/// default `abort`, which truncates them with an HTTP 200. Lumbridge Computer drains
/// first anyway, so `mode=wait` only covers a request that slipped in after
/// the drain (a direct client). An engine that rejects the parameter (400 or
/// 422) is asked again without it: the drain already did its job.
pub fn sleep(model: &Model, level: u8) -> Result<()> {
    let config = model
        .serve
        .sleep
        .as_ref()
        .with_context(|| format!("model '{}' has no serve.sleep", model.name))?;
    if !supports_sleep(model) {
        bail!("model '{}' cannot sleep", model.name);
    }
    let waiting = HttpStep::post(&format!("/sleep?level={level}&mode=wait"), None);
    let (status, _) = send(model, &waiting, config.sleep_timeout())?;
    if matches!(status, 400 | 422) {
        call(
            model,
            &HttpStep::post(&format!("/sleep?level={level}"), None),
            config.sleep_timeout(),
        )?;
    } else if !(200..300).contains(&status) {
        bail!("POST {} returned HTTP {status}", waiting.path);
    }
    if !is_sleeping(model)? {
        bail!(
            "model '{}' accepted /sleep but does not report sleeping",
            model.name
        );
    }
    Ok(())
}

/// What a wake's `/collective_rpc` reload reported (`results[0]` of the reply).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Reload {
    #[serde(default)]
    pub target_secs: Option<f64>,
    #[serde(default)]
    pub draft_secs: Option<f64>,
    /// `true` once the drafter's weights were reloaded. The spike's version of
    /// the extension sent the tensor count here, so a positive number counts.
    #[serde(default, deserialize_with = "flag_or_count")]
    pub draft_loaded: Option<bool>,
}

fn flag_or_count<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<bool>, D::Error> {
    use serde::Deserialize;
    Ok(match serde_json::Value::deserialize(d)? {
        serde_json::Value::Bool(flag) => Some(flag),
        serde_json::Value::Number(n) => Some(n.as_f64().is_some_and(|n| n > 0.0)),
        _ => None,
    })
}

/// `results[0]` of a `/collective_rpc` reply, if it is an object.
fn parse_rpc(body: &str) -> Result<Option<Reload>> {
    if body.trim().is_empty() {
        return Ok(None);
    }
    let reply: serde_json::Value =
        serde_json::from_str(body.trim()).context("parsing the /collective_rpc reply")?;
    let first = reply.get("results").and_then(|results| results.get(0));
    match first {
        Some(value) if value.is_object() => serde_json::from_value(value.clone())
            .map(Some)
            .context("parsing results[0] of the /collective_rpc reply"),
        _ => Ok(None),
    }
}

/// Run the configured wake steps then the reload steps, in order, each bounded
/// by what is left until `deadline`. Readiness is the caller's to check.
///
/// Fails when a step fails or the deadline passes, and when the model has a
/// speculative config but no `/collective_rpc` step reported
/// `draft_loaded: true`: the stock `reload_weights` silently leaves the MTP
/// drafter empty (acceptance 0, decode 3–4× slower). Any failure leaves the
/// engine in an unknown state; the caller must stop it, never serve from it.
pub fn wake(model: &Model, deadline: Instant) -> Result<Option<Reload>> {
    let config = model
        .serve
        .sleep
        .as_ref()
        .with_context(|| format!("model '{}' has no serve.sleep", model.name))?;
    let mut reload = None;
    for step in config
        .wake_steps()
        .iter()
        .chain(config.reload_steps().iter())
    {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            bail!(
                "waking '{}' ran out of its {}s wake timeout",
                model.name,
                config.wake_timeout().as_secs()
            );
        }
        let body = call(model, step, left)?;
        if step.path.split('?').next() == Some("/collective_rpc") {
            if let Some(report) = parse_rpc(&body)? {
                reload = Some(report);
            }
        }
    }
    if has_speculative_config(model) && reload.as_ref().and_then(|r| r.draft_loaded) != Some(true) {
        bail!(
            "'{}' has a speculative config but the reload did not report its drafter \
             loaded (draft_loaded: {}); serving would run without MTP",
            model.name,
            reload
                .as_ref()
                .and_then(|r| r.draft_loaded)
                .map_or("missing".to_string(), |flag| flag.to_string())
        );
    }
    Ok(reload)
}

/// `GET /is_sleeping` → `{"is_sleeping": bool}`.
pub fn is_sleeping(model: &Model) -> Result<bool> {
    let body = call(
        model,
        &HttpStep {
            method: "GET".to_string(),
            path: "/is_sleeping".to_string(),
            body: None,
        },
        QUERY_TIMEOUT,
    )?;
    #[derive(serde::Deserialize)]
    struct Reply {
        is_sleeping: bool,
    }
    let reply: Reply = serde_json::from_str(body.trim())
        .with_context(|| format!("parsing /is_sleeping from '{}'", model.name))?;
    Ok(reply.is_sleeping)
}

/// A fake vLLM for tests: `/sleep`, `/wake_up`, `/collective_rpc`,
/// `/is_sleeping`, `/v1/models` and `/metrics` on 127.0.0.1:0.
#[cfg(test)]
pub mod testing {
    use crate::config::Model;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    #[derive(Debug)]
    pub struct FakeState {
        pub sleeping: bool,
        pub level: Option<u8>,
        /// Every control call, as `METHOD /path?query`.
        pub calls: Vec<String>,
        /// Requests the engine itself reports running (direct clients).
        pub running: u32,
        /// While true `/v1/models` answers without the marker (not ready).
        pub unready: bool,
        /// Extra delay on `/wake_up`, to make concurrent wakes overlap.
        pub wake_delay: Duration,
        /// Make `/sleep` fail with a 500.
        pub fail_sleep: bool,
        /// Answer `/sleep?…&mode=…` with a 422, like a vLLM without `mode`.
        pub reject_sleep_mode: bool,
        /// Body of a 200 `/collective_rpc` reply.
        pub rpc_reply: String,
        /// Make `/collective_rpc` fail with a 500.
        pub fail_rpc: bool,
        /// `POST /v1/chat/completions` calls seen (liveness probes).
        pub chats: u32,
        /// Health-style reads seen: `/health`, `/v1/models`, `/metrics` and chats.
        pub probes: u32,
    }

    /// What the real extension answers for a model without a drafter.
    pub const RELOAD_NO_DRAFT: &str =
        r#"{"results":[{"target_secs":1.2,"has_draft":false,"draft_loaded":false}]}"#;
    /// ... and for a model whose MTP drafter reloaded.
    pub const RELOAD_WITH_DRAFT: &str = r#"{"results":[{"target_secs":1.2,"has_draft":true,"draft_secs":0.03,"draft_tensors":15,"draft_loaded":true}]}"#;

    impl Default for FakeState {
        fn default() -> Self {
            FakeState {
                sleeping: false,
                level: None,
                calls: Vec::new(),
                running: 0,
                unready: false,
                wake_delay: Duration::ZERO,
                fail_sleep: false,
                reject_sleep_mode: false,
                rpc_reply: RELOAD_NO_DRAFT.to_string(),
                fail_rpc: false,
                chats: 0,
                probes: 0,
            }
        }
    }

    pub struct FakeVllm {
        pub port: u16,
        pub state: Arc<Mutex<FakeState>>,
    }

    impl FakeVllm {
        pub fn start() -> FakeVllm {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let state = Arc::new(Mutex::new(FakeState::default()));
            let shared = Arc::clone(&state);
            thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let state = Arc::clone(&shared);
                    thread::spawn(move || serve(stream, &state));
                }
            });
            FakeVllm { port, state }
        }

        pub fn set(&self, change: impl FnOnce(&mut FakeState)) {
            change(&mut self.state.lock().unwrap());
        }

        pub fn sleeping(&self) -> bool {
            self.state.lock().unwrap().sleeping
        }

        pub fn calls(&self) -> Vec<String> {
            self.state.lock().unwrap().calls.clone()
        }

        pub fn count(&self, prefix: &str) -> usize {
            self.calls()
                .iter()
                .filter(|c| c.starts_with(prefix))
                .count()
        }

        /// Registry YAML for a sleepable vLLM model on this port, indented
        /// under `models:`.
        pub fn registry_entry(&self, id: &str, footprint_gb: f64, extra: &str) -> String {
            let port = self.port;
            format!(
                "  {id}:\n    name: {id}\n    footprint_gb: {footprint_gb}\n    health: http://localhost:{port}/v1/models\n    health_contains: fake-vllm\n{extra}    serve:\n      kind: vllm\n      port: {port}\n      sleep: {{level: 2, footprint_gb: 2}}\n"
            )
        }

        pub fn model(&self) -> Model {
            let yaml = self.registry_entry("fake", 10.0, "");
            let body: String = yaml
                .lines()
                .skip(1)
                .map(|line| format!("{}\n", line.strip_prefix("    ").unwrap_or(line)))
                .collect();
            serde_yaml::from_str(&body).unwrap()
        }
    }

    fn serve(mut stream: TcpStream, state: &Mutex<FakeState>) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
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
        let mut words = text.split_whitespace();
        let method = words.next().unwrap_or("GET").to_string();
        let target = words.next().unwrap_or("/").to_string();
        let path = target.split('?').next().unwrap_or("/").to_string();
        if matches!(
            path.as_str(),
            "/health" | "/v1/models" | "/metrics" | "/v1/chat/completions"
        ) {
            state.lock().unwrap().probes += 1;
        }
        let (status, body) = match path.as_str() {
            "/sleep" | "/wake_up" | "/collective_rpc" => {
                let delay = {
                    let mut s = state.lock().unwrap();
                    s.calls.push(format!("{method} {target}"));
                    if path == "/wake_up" {
                        s.wake_delay
                    } else {
                        Duration::ZERO
                    }
                };
                thread::sleep(delay);
                let mut s = state.lock().unwrap();
                let query = target.split_once('?').map_or("", |(_, q)| q).to_string();
                match path.as_str() {
                    "/sleep" if s.fail_sleep => ("500 Internal Server Error", "{}".to_string()),
                    "/sleep" if s.reject_sleep_mode && query.contains("mode=") => {
                        ("422 Unprocessable Entity", "{}".to_string())
                    }
                    "/sleep" => {
                        s.sleeping = true;
                        s.level = query
                            .split('&')
                            .find_map(|kv| kv.strip_prefix("level="))
                            .and_then(|l| l.parse().ok());
                        ("200 OK", String::new())
                    }
                    // Like vLLM: still sleeping (scheduler paused) until every
                    // tag is awake, so only kv_cache or an untagged wake ends it.
                    "/wake_up" if query == "tags=weights" => ("200 OK", String::new()),
                    "/wake_up" => {
                        s.sleeping = false;
                        ("200 OK", String::new())
                    }
                    _ if s.fail_rpc => ("500 Internal Server Error", "{}".to_string()),
                    _ => ("200 OK", s.rpc_reply.clone()),
                }
            }
            // vLLM's /health cannot see sleep: 200 asleep or awake.
            "/health" => ("200 OK", String::new()),
            "/v1/chat/completions" => {
                let mut s = state.lock().unwrap();
                s.chats += 1;
                (
                    "200 OK",
                    "{\"choices\":[{\"message\":{\"content\":\"ok\"}}]}".to_string(),
                )
            }
            "/is_sleeping" => {
                let s = state.lock().unwrap();
                ("200 OK", format!("{{\"is_sleeping\":{}}}", s.sleeping))
            }
            "/v1/models" => {
                let s = state.lock().unwrap();
                if s.unready {
                    ("503 Service Unavailable", "{}".to_string())
                } else {
                    ("200 OK", "{\"data\":[{\"id\":\"fake-vllm\"}]}".to_string())
                }
            }
            "/metrics" => {
                let s = state.lock().unwrap();
                (
                    "200 OK",
                    format!(
                        "vllm:num_requests_running{{engine=\"0\"}} {}\nvllm:num_requests_waiting{{engine=\"0\"}} 0\nvllm:prompt_tokens_total{{engine=\"0\"}} 1\nvllm:generation_tokens_total{{engine=\"0\"}} 1\n",
                        s.running
                    ),
                )
            }
            _ => ("404 Not Found", String::new()),
        };
        let _ = write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::testing::FakeVllm;
    use super::*;

    fn soon() -> Instant {
        Instant::now() + Duration::from_secs(10)
    }

    fn with_mtp(mut model: Model) -> Model {
        model.serve.args.insert(
            "speculative-config".into(),
            serde_yaml::Value::String(r#"{"method":"mtp","num_speculative_tokens":3}"#.into()),
        );
        model
    }

    #[test]
    fn sleep_asks_vllm_to_finish_in_flight_work_and_confirms_with_is_sleeping() {
        let engine = FakeVllm::start();
        let model = engine.model();
        assert!(supports_sleep(&model));
        assert!(!is_sleeping(&model).unwrap());
        sleep(&model, 2).unwrap();
        assert!(is_sleeping(&model).unwrap());
        assert_eq!(engine.calls(), ["POST /sleep?level=2&mode=wait"]);
        assert_eq!(engine.state.lock().unwrap().level, Some(2));
    }

    #[test]
    fn an_engine_without_sleep_modes_is_asked_again_without_one() {
        let engine = FakeVllm::start();
        engine.set(|s| s.reject_sleep_mode = true);
        sleep(&engine.model(), 2).unwrap();
        assert_eq!(
            engine.calls(),
            ["POST /sleep?level=2&mode=wait", "POST /sleep?level=2"]
        );
        assert!(engine.sleeping());
    }

    #[test]
    fn the_default_wake_is_weights_then_the_fast_reload_then_the_kv_cache() {
        let engine = FakeVllm::start();
        let model = engine.model();
        sleep(&model, 2).unwrap();
        let reload = wake(&model, soon()).unwrap().unwrap();
        assert_eq!(reload.target_secs, Some(1.2));
        assert!(!engine.sleeping());
        assert_eq!(
            engine.calls(),
            [
                "POST /sleep?level=2&mode=wait",
                "POST /wake_up?tags=weights",
                "POST /collective_rpc",
                "POST /wake_up?tags=kv_cache",
            ]
        );
        let steps = model.serve.sleep.as_ref().unwrap().reload_steps();
        assert_eq!(
            steps[0].body.as_deref(),
            Some(r#"{"method":"reload_all_fast"}"#)
        );
    }

    #[test]
    fn a_model_with_mtp_must_report_its_drafter_reloaded() {
        let engine = FakeVllm::start();
        let model = with_mtp(engine.model());
        assert!(has_speculative_config(&model));
        assert!(!has_speculative_config(&engine.model()));

        // The extension says the drafter loaded: a good wake.
        engine.set(|s| s.rpc_reply = testing::RELOAD_WITH_DRAFT.into());
        let reload = wake(&model, soon()).unwrap().unwrap();
        assert_eq!(reload.draft_loaded, Some(true));

        // draft_loaded false, missing, or no reply at all (the stock
        // reload_weights): a failed wake, whatever HTTP said.
        for reply in [
            testing::RELOAD_NO_DRAFT,
            r#"{"results":[{"target_secs":42.1}]}"#,
            r#"{"results":[null]}"#,
            "",
        ] {
            engine.set(|s| s.rpc_reply = reply.into());
            let error = format!("{:#}", wake(&model, soon()).unwrap_err());
            assert!(error.contains("drafter"), "{reply}: {error}");
        }

        // The spike's extension sent a tensor count: positive counts as loaded.
        engine.set(|s| s.rpc_reply = r#"{"results":[{"draft_loaded":785}]}"#.into());
        wake(&model, soon()).unwrap();
        engine.set(|s| s.rpc_reply = r#"{"results":[{"draft_loaded":0}]}"#.into());
        assert!(wake(&model, soon()).is_err());
    }

    #[test]
    fn a_failed_reload_fails_the_wake_and_skips_the_kv_cache() {
        let engine = FakeVllm::start();
        let model = engine.model();
        sleep(&model, 2).unwrap();
        engine.set(|s| s.fail_rpc = true);
        let error = format!("{:#}", wake(&model, soon()).unwrap_err());
        assert!(error.contains("/collective_rpc"), "{error}");
        assert!(
            !engine.calls().iter().any(|c| c.contains("kv_cache")),
            "the scheduler must stay paused on torn weights"
        );
    }

    #[test]
    fn a_wake_is_bounded_by_its_deadline() {
        let engine = FakeVllm::start();
        engine.set(|s| s.wake_delay = Duration::from_millis(300));
        let model = engine.model();
        // The first call is cut off at the deadline (its read timeout is what is
        // left), and nothing after it runs.
        assert!(wake(&model, Instant::now() + Duration::from_millis(100)).is_err());
        assert_eq!(engine.count("POST /collective_rpc"), 0);
    }

    #[test]
    fn the_wake_procedure_is_configurable() {
        let engine = FakeVllm::start();
        let mut model = engine.model();
        let config = model.serve.sleep.as_mut().unwrap();
        config.wake = Some(vec![HttpStep::post("/wake_up", None)]);
        config.reload = Some(Vec::new());
        assert_eq!(wake(&model, soon()).unwrap(), None);
        assert_eq!(engine.calls(), ["POST /wake_up"]);
    }

    #[test]
    fn a_failed_sleep_is_an_error() {
        let engine = FakeVllm::start();
        engine.set(|s| s.fail_sleep = true);
        let error = sleep(&engine.model(), 2).unwrap_err();
        assert!(format!("{error:#}").contains("500"), "{error:#}");
    }

    #[test]
    fn only_a_builder_launched_vllm_with_a_sleep_block_can_sleep() {
        let engine = FakeVllm::start();
        let mut model = engine.model();
        assert!(supports_sleep(&model));
        model.serve.command = vec![
            "vllm".into(),
            "serve".into(),
            "/w".into(),
            "--enable-sleep-mode".into(),
        ];
        assert!(
            !supports_sleep(&model),
            "a hand-written argv can't carry the extension or the loopback bind"
        );
        model.serve.command.clear();
        model.serve.sleep = None;
        assert!(!supports_sleep(&model));
    }

    #[test]
    fn engine_control_never_leaves_loopback() {
        let error = super::super::http::request_within(
            "POST",
            "http://10.0.0.1:8001/sleep?level=2",
            None,
            CONNECT_TIMEOUT,
            QUERY_TIMEOUT,
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("loopback only"));
    }
}
