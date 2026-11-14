//! The Governor — the kernel of Lumbridge Computer.
//!
//! On a unified-memory box, over-commit doesn't fail gracefully: the whole machine
//! thrashes and wedges (SSH/ping included) before the OOM killer acts. The Governor
//! makes that impossible via (1) admission control against a hard budget and
//! (2) a watchdog on `MemAvailable` that kills the newest model before thrash.
//!
//! This module currently provides the *sensing* + *admission* half. The watchdog and
//! process spawn/kill land with the `up`/`activate` commands.

use crate::config::{Liveness, LivenessKind, Model, Registry};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

#[cfg(test)]
use crate::pool::DEFAULT_BUDGET_GB;
/// Admission, headroom, reservations and the floor are memory-pool questions;
/// they live in `pool`. The unified pool's compiled numbers keep their old names.
pub use crate::pool::{Pool, PoolId, WATCHDOG_FLOOR_GB};

/// Status must use the same configured Scene ceiling as lifecycle admission.
pub fn scene_budget_gb(
    pool: &Pool,
    root: &std::path::Path,
    scene: Option<&str>,
) -> anyhow::Result<f64> {
    match scene {
        Some(name) => Ok(pool.scene_budget_gb(crate::config::find_scene(root, name)?.budget_gb)),
        None => Ok(pool.scene_budget_gb(None)),
    }
}

/// A model is considered "running" if its serving port accepts a connection.
pub fn port_open(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(150)).is_ok()
}

/// The outcome of one readiness probe. Supervision needs to know *how* a probe
/// failed: a refused connection or the wrong model on the port is a failure no
/// matter what, while a timeout may only mean the engine is busy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readiness {
    /// The port answered and, if a marker is configured, the response held it.
    Ready,
    /// Nothing listening (or no serving port at all).
    Refused,
    /// Connected, or tried to, but no complete answer within the timeout.
    Timeout,
    /// Answered without the model's `health_contains` marker.
    Mismatch,
}

impl Readiness {
    pub fn describe(self) -> &'static str {
        match self {
            Readiness::Ready => "ready",
            Readiness::Refused => "connection refused",
            Readiness::Timeout => "readiness probe timed out",
            Readiness::Mismatch => "health marker missing from the response",
        }
    }
}

pub fn is_running(m: &Model) -> bool {
    readiness(m, m.probe_timeout()) == Readiness::Ready
}

/// Probe the model's port and exact health marker within `timeout`.
pub fn readiness(m: &Model, timeout: Duration) -> Readiness {
    let Some(port) = m.serve.port else {
        return Readiness::Refused;
    };
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    // A listening localhost socket accepts from its backlog even while the
    // server is busy, so a long connect means the backlog is full: slow, not gone.
    let stream = match TcpStream::connect_timeout(&addr, timeout.min(Duration::from_secs(2))) {
        Ok(stream) => stream,
        Err(error) if error.kind() == std::io::ErrorKind::TimedOut => return Readiness::Timeout,
        Err(_) => return Readiness::Refused,
    };
    let Some(marker) = &m.health_contains else {
        return Readiness::Ready;
    };
    match health_response(m, stream, timeout) {
        Ok(response) if response.contains(marker.as_str()) => Readiness::Ready,
        Ok(_) => Readiness::Mismatch,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
            ) =>
        {
            Readiness::Timeout
        }
        Err(_) => Readiness::Refused,
    }
}

/// Fetch the local HTTP health endpoint without adding an HTTP client runtime.
/// Registry health URLs are deliberately localhost-only.
fn health_response(m: &Model, mut stream: TcpStream, timeout: Duration) -> std::io::Result<String> {
    let path = m
        .health
        .as_deref()
        .and_then(|url| url.split_once("localhost"))
        .map(|(_, tail)| tail.trim_start_matches(|c: char| c.is_ascii_digit() || c == ':'))
        .filter(|path| path.starts_with('/'))
        .unwrap_or("/");
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    stream.write_all(format!("GET {path} HTTP/1.0\r\nHost: localhost\r\n\r\n").as_bytes())?;
    let mut response = Vec::new();
    stream.take(1 << 20).read_to_end(&mut response)?;
    Ok(String::from_utf8_lossy(&response).into_owned())
}

/// Moved to `engines`, the one place that understands engine metrics.
pub use crate::engines::EngineProgress;

/// Read the model's own `/metrics`. `None` when it can't be read, doesn't
/// carry the counters, or the model has no vLLM endpoint (neither `kind: vllm`
/// nor vLLM liveness); callers treat that as "unknown", not as wedged or idle.
pub fn engine_progress(m: &Model) -> Option<EngineProgress> {
    let liveness: Option<&Liveness> = m
        .supervision
        .as_ref()
        .and_then(|policy| policy.liveness.as_ref());
    let path = match liveness {
        Some(liveness) => match liveness.kind {
            LivenessKind::VllmMetrics => liveness.metrics_path(),
        },
        None if m.serve.kind == "vllm" => "/metrics",
        None => return None,
    };
    let port = m.serve.port?;
    // A busy engine is slow to answer /metrics too; give it the readiness budget.
    let timeout = m.probe_timeout();
    let url = format!("http://127.0.0.1:{port}{path}");
    let body =
        crate::engines::http::fetch_http_within(&url, timeout.min(Duration::from_secs(2)), timeout)
            .ok()?;
    vllm_progress(&body)
}

/// Thin wrapper kept so liveness is unchanged until lane A switches to
/// `engines::scrape`; the parser itself lives in `engines`.
pub fn vllm_progress(metrics: &str) -> Option<EngineProgress> {
    crate::engines::vllm_progress(metrics)
}

/// Send one 1-token chat to the model's own port. `true` only for an HTTP 200
/// whose body holds `"choices"`. Direct to the backend, never via a gateway.
pub fn chat_probe(m: &Model, probe: &crate::config::ChatProbe) -> bool {
    let Some(port) = m.serve.port else {
        return false;
    };
    let Some(name) = probe
        .model
        .clone()
        .or_else(|| m.serve.served_name.first().cloned())
    else {
        return false;
    };
    let body = serde_json::json!({
        "model": name,
        "messages": [{"role": "user", "content": "Reply with OK"}],
        "max_tokens": 1,
    })
    .to_string();
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_secs(2)) else {
        return false;
    };
    let timeout = Some(Duration::from_secs(probe.timeout_sec()));
    if stream.set_read_timeout(timeout).is_err() || stream.set_write_timeout(timeout).is_err() {
        return false;
    }
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nUser-Agent: lumbridge-computer-liveness\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = Vec::new();
    let _ = (&mut stream).take(1 << 20).read_to_end(&mut response);
    let text = String::from_utf8_lossy(&response);
    let status_ok = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        == Some("200");
    status_ok && text.contains("\"choices\"")
}

pub fn running_ids(reg: &Registry) -> Vec<String> {
    reg.models
        .iter()
        .filter(|(_, m)| is_running(m))
        .map(|(id, _)| id.clone())
        .collect()
}

/// Block until the exact registered model is healthy, or `timeout` elapses.
/// Models without a port are considered instantly ready. Startup paths use
/// `lifecycle::wait_started`, which also watches the process and progress.
#[cfg(test)]
pub fn wait_healthy(m: &Model, timeout: Duration) -> bool {
    let Some(_) = m.serve.port else {
        return true;
    };
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if is_running(m) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(400));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_uses_the_active_scene_budget_instead_of_the_default() {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("compute-budget-{suffix}"));
        std::fs::create_dir_all(root.join("scenes")).unwrap();
        std::fs::write(root.join("scenes/everything.scene.yaml"),
            "apiVersion: lumbridge/v1\nmetadata:\n  name: everything\n  version: 1\nmodels: []\nbudget_gb: 111\n").unwrap();
        assert_eq!(
            scene_budget_gb(&Pool::unified(), &root, Some("everything")).unwrap(),
            111.0
        );
        assert_eq!(
            scene_budget_gb(&Pool::unified(), &root, None).unwrap(),
            DEFAULT_BUDGET_GB
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn vllm_progress_reads_in_flight_and_both_token_counters() {
        // Shape of a vLLM 0.27.1 brain's /metrics, trimmed.
        let metrics =
            "# HELP vllm:num_requests_running Number of requests in model execution batches.\n\
# TYPE vllm:num_requests_running gauge\n\
vllm:num_requests_running{engine=\"0\",model_name=\"brain\"} 2.0\n\
vllm:num_requests_waiting{engine=\"0\",model_name=\"brain\"} 1.0\n\
vllm:num_requests_waiting_by_reason{engine=\"0\",model_name=\"brain\",reason=\"capacity\"} 1.0\n\
vllm:prompt_tokens_total{engine=\"0\",model_name=\"brain\"} 165238.0\n\
vllm:prompt_tokens_created{engine=\"0\",model_name=\"brain\"} 1.79e+09\n\
vllm:prompt_tokens_cached_total{engine=\"0\",model_name=\"brain\"} 9999.0\n\
vllm:generation_tokens_total{engine=\"0\",model_name=\"brain\"} 7476.0\n";
        assert_eq!(
            vllm_progress(metrics),
            Some(EngineProgress {
                running: 2.0,
                waiting: 1.0,
                in_flight: 3.0,
                tokens: 165238.0 + 7476.0,
            })
        );
        // Not a vLLM endpoint (or a renamed metric): unknown, never "idle".
        assert_eq!(vllm_progress("up 1\n"), None);
        assert_eq!(
            vllm_progress("vllm:num_requests_running{engine=\"0\"} 1.0\n"),
            None
        );
    }
}
