//! M3 hold-and-wake tests: a fake `Lifecycle` (state, release timing, failure)
//! in front of the fake upstreams from `http_tests`.

use super::*;
use crate::gateway::routes::SceneMode;
use crate::gateway::INFLIGHT_PATH;
use crate::states::{Lifecycle, ModelState};
use std::collections::HashMap as Map;
use std::sync::atomic::AtomicU32;
use std::sync::{Condvar, Mutex};

/// A lifecycle whose wake finishes when the test says so (`release`), or never
/// (until the deadline). Counts `ensure_ready` calls.
struct FakeLifecycle {
    state: Mutex<ModelState>,
    outcome: Mutex<Option<Result<(), String>>>,
    released: Condvar,
    calls: AtomicU32,
    /// Time left until each `ensure_ready` call's deadline, as it was called.
    budgets: Mutex<Vec<Duration>>,
}

impl FakeLifecycle {
    fn new(state: ModelState) -> Arc<FakeLifecycle> {
        Arc::new(FakeLifecycle {
            state: Mutex::new(state),
            outcome: Mutex::new(None),
            released: Condvar::new(),
            calls: AtomicU32::new(0),
            budgets: Mutex::new(Vec::new()),
        })
    }

    fn release(&self, outcome: Result<(), &str>) {
        *self.outcome.lock().unwrap() = Some(outcome.map_err(str::to_string));
        self.released.notify_all();
    }

    /// Release after `delay`, from another thread (the "wake takes N ms" knob).
    fn release_after(self: &Arc<Self>, delay: Duration, outcome: Result<(), &'static str>) {
        let me = self.clone();
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            me.release(outcome);
        });
    }

    fn calls(&self) -> u32 {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Lifecycle for FakeLifecycle {
    fn state(&self, _id: &str) -> ModelState {
        *self.state.lock().unwrap()
    }

    fn ensure_ready(&self, _id: &str, deadline: Instant) -> anyhow::Result<()> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.budgets
            .lock()
            .unwrap()
            .push(deadline.saturating_duration_since(Instant::now()));
        let mut outcome = self.outcome.lock().unwrap();
        while outcome.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                anyhow::bail!("deadline passed while waking");
            }
            outcome = self.released.wait_timeout(outcome, left).unwrap().0;
        }
        match outcome.clone().unwrap_or(Ok(())) {
            Ok(()) => {
                *self.state.lock().unwrap() = ModelState::Ready;
                Ok(())
            }
            Err(reason) => anyhow::bail!(reason),
        }
    }

    fn sleep(&self, _id: &str, _reason: &str) -> anyhow::Result<()> {
        *self.state.lock().unwrap() = ModelState::Sleeping { level: 2 };
        Ok(())
    }
}

struct Wake {
    wake_timeout: Duration,
    max_held: u32,
}

impl Default for Wake {
    fn default() -> Wake {
        Wake {
            wake_timeout: Duration::from_secs(10),
            max_held: 32,
        }
    }
}

async fn start_wake_gateway(
    routes: Routes,
    lifecycle: Option<Arc<dyn Lifecycle>>,
    wake: Wake,
) -> Gateway {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let in_flight = InFlight::default();
    let shared = Shared::new(
        routes,
        in_flight.clone(),
        Config {
            default_upstream: None,
            max_body_bytes: MIB,
            lifecycle,
            wake_timeout: wake.wake_timeout,
            max_held_per_model: wake.max_held,
            keepalive_every: Duration::from_millis(40),
        },
    );
    tokio::spawn(serve(listener, shared));
    Gateway { addr, in_flight }
}

/// `brain` with `mode`, on `port`; `up` is what the port probe saw.
fn routes_for(port: u16, mode: SceneMode, up: bool) -> Routes {
    let reg = registry(&[
        ("brain", port, &["osprey"], ""),
        ("embed", 1, &["bge"], ""),
        ("brain-alt", 2, &["osprey-alt"], ""),
    ]);
    let modes = Map::from([("brain".to_string(), mode)]);
    let mut routes =
        Routes::build_with_modes(&reg, &["brain".into(), "embed".into()], &modes, |m| {
            up && m.serve.port == Some(port)
        });
    routes.scene = Some("lite".into());
    routes
}

async fn chat(gw: &Gateway, model: &str) -> (StatusCode, HeaderMap, serde_json::Value) {
    send(
        gw.addr,
        Method::POST,
        "/v1/chat/completions",
        "application/json",
        format!(r#"{{"model":"{model}"}}"#),
    )
    .await
}

/// Collect a streaming body until it ends, as text.
async fn read_all(body: &mut Incoming) -> String {
    let mut out = String::new();
    while let Some(chunk) = next_chunk(body).await {
        out.push_str(std::str::from_utf8(&chunk).unwrap());
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_non_streaming_request_waits_through_a_wake_and_succeeds() {
    let upstream = echo_upstream("A").await;
    let lifecycle = FakeLifecycle::new(ModelState::Stopped);
    let gw = start_wake_gateway(
        routes_for(upstream.port(), SceneMode::OnDemand, false),
        Some(lifecycle.clone()),
        Wake::default(),
    )
    .await;
    lifecycle.release_after(Duration::from_millis(150), Ok(()));
    let started = Instant::now();
    let (status, headers, reply) = chat(&gw, "osprey").await;
    assert_eq!(status, StatusCode::OK);
    assert!(started.elapsed() >= Duration::from_millis(150), "it waited");
    assert_eq!(reply["upstream"], "A");
    assert_eq!(headers[MODEL_HEADER], "brain");
    assert_eq!(lifecycle.calls(), 1);

    // Ready now: the next request goes straight through, no wake.
    let (status, _, _) = chat(&gw, "brain").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(lifecycle.calls(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_streaming_request_gets_keepalive_comments_then_real_chunks_in_one_response() {
    let (upstream, mut streams) = sse_upstream().await;
    // Sleeping vLLM: the port is open, but the lifecycle says it isn't ready.
    let lifecycle = FakeLifecycle::new(ModelState::Sleeping { level: 2 });
    let gw = start_wake_gateway(
        routes_for(upstream.port(), SceneMode::Sleepable, true),
        Some(lifecycle.clone()),
        Wake::default(),
    )
    .await;

    let (mut body, headers, _task) = start_stream(&gw).await;
    assert_eq!(headers[header::CONTENT_TYPE], "text/event-stream");
    assert_eq!(headers["x-accel-buffering"], "no");
    assert_eq!(headers[MODEL_HEADER], "brain");
    assert_eq!(
        next_chunk(&mut body).await.unwrap(),
        ": lumbridge-computer waking brain 0s\n\n",
        "headers and a first comment arrive before the model is ready"
    );
    let second = next_chunk(&mut body).await.unwrap();
    assert!(
        second.starts_with(b": lumbridge-computer waking brain "),
        "{second:?}"
    );
    assert_eq!(gw.in_flight.waiting_snapshot()["brain"], 1);

    lifecycle.release(Ok(()));
    let chunks = streams.recv().await.unwrap();
    assert_eq!(
        gw.in_flight.waiting_snapshot()["brain"],
        0,
        "no longer waiting"
    );
    assert_eq!(gw.in_flight.get("brain"), 1, "still in flight");
    chunks
        .send(Bytes::from_static(b"data: one\n\n"))
        .await
        .unwrap();
    chunks
        .send(Bytes::from_static(b"data: [DONE]\n\n"))
        .await
        .unwrap();
    drop(chunks);
    let rest = read_all(&mut body).await;
    // Keep-alives may still have been queued before the wake finished; after
    // them come the upstream's events, in order, and then a clean end.
    let data_at = rest.find("data: one").expect("upstream events arrive");
    assert!(rest[..data_at]
        .split("\n\n")
        .all(|part| part.is_empty() || part.starts_with(": lumbridge-computer waking brain ")));
    assert_eq!(&rest[data_at..], "data: one\n\ndata: [DONE]\n\n");
    let in_flight = gw.in_flight.clone();
    eventually("in-flight to drop", || in_flight.get("brain") == 0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wake_failure_mid_stream_produces_an_error_event_and_ends_the_stream() {
    let lifecycle = FakeLifecycle::new(ModelState::Stopped);
    let gw = start_wake_gateway(
        routes_for(9, SceneMode::OnDemand, false),
        Some(lifecycle.clone()),
        Wake::default(),
    )
    .await;
    let (mut body, headers, _task) = start_stream(&gw).await;
    assert_eq!(headers[header::CONTENT_TYPE], "text/event-stream");
    next_chunk(&mut body).await.unwrap();
    lifecycle.release(Err("not enough memory to admit brain"));
    let rest = read_all(&mut body).await;
    let event = rest
        .split("\n\n")
        .find_map(|part| part.strip_prefix("data: "))
        .expect("an error event");
    let event: serde_json::Value = serde_json::from_str(event).unwrap();
    assert_eq!(event["error"]["code"], "model_unavailable");
    assert_eq!(event["error"]["type"], "server_error");
    assert!(event["error"]["message"]
        .as_str()
        .unwrap()
        .contains("not enough memory"));
    assert!(
        rest.ends_with("\n\n"),
        "the event is complete, then the stream ends"
    );
    let in_flight = gw.in_flight.clone();
    eventually("in-flight to drop", || in_flight.get("brain") == 0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_non_sse_upstream_answer_after_an_early_stream_becomes_an_error_event() {
    let upstream = fake_upstream(|_request| async {
        Response::builder()
            .status(400)
            .header("content-type", "application/json")
            .body(
                Full::new(Bytes::from_static(
                    br#"{"object":"error","message":"max_tokens too large","code":400}"#,
                ))
                .boxed_unsync(),
            )
            .unwrap()
    })
    .await;
    let lifecycle = FakeLifecycle::new(ModelState::Stopped);
    lifecycle.release(Ok(()));
    let gw = start_wake_gateway(
        routes_for(upstream.port(), SceneMode::OnDemand, false),
        Some(lifecycle),
        Wake::default(),
    )
    .await;
    let (mut body, _, _task) = start_stream(&gw).await;
    let rest = read_all(&mut body).await;
    let event = rest
        .split("\n\n")
        .find_map(|part| part.strip_prefix("data: "))
        .expect("an error event");
    let event: serde_json::Value = serde_json::from_str(event).unwrap();
    assert_eq!(event["error"]["message"], "max_tokens too large");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn held_requests_beyond_the_cap_get_503_and_count_as_in_flight() {
    let upstream = echo_upstream("A").await;
    let lifecycle = FakeLifecycle::new(ModelState::Stopped);
    let gw = start_wake_gateway(
        routes_for(upstream.port(), SceneMode::OnDemand, false),
        Some(lifecycle.clone()),
        Wake {
            max_held: 2,
            ..Wake::default()
        },
    )
    .await;
    let addr = gw.addr;
    let held: Vec<_> = (0..2)
        .map(|_| {
            tokio::spawn(async move {
                send(
                    addr,
                    Method::POST,
                    "/v1/chat/completions",
                    "application/json",
                    r#"{"model":"brain"}"#,
                )
                .await
                .0
            })
        })
        .collect();
    let in_flight = gw.in_flight.clone();
    eventually("two held requests", || {
        in_flight.waiting_snapshot().get("brain") == Some(&2)
    })
    .await;
    // Held requests are in flight, so the idle sleeper leaves the model alone.
    assert_eq!(gw.in_flight.get("brain"), 2);
    assert_eq!(gw.in_flight.idle_for("brain"), Some(Duration::ZERO));
    let (status, _, reply) = send(gw.addr, Method::GET, INFLIGHT_PATH, "", "").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["in_flight"]["brain"], 2);
    assert_eq!(reply["waiting"]["brain"], 2);

    let (status, headers, reply) = chat(&gw, "brain").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(headers[header::RETRY_AFTER], RETRY_AFTER_SEC);
    assert_eq!(reply["error"]["code"], "wake_queue_full");
    assert_eq!(
        gw.in_flight.waiting_snapshot()["brain"],
        2,
        "the refusal isn't counted"
    );

    lifecycle.release(Ok(()));
    for request in held {
        assert_eq!(request.await.unwrap(), StatusCode::OK);
    }
    assert_eq!(
        lifecycle.calls(),
        2,
        "each held request asked; the lifecycle shares the wake"
    );
    eventually("in-flight to drop", || in_flight.get("brain") == 0).await;
    assert_eq!(gw.in_flight.waiting_snapshot()["brain"], 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wake_past_the_deadline_is_a_504_and_a_failed_wake_a_503() {
    let upstream = echo_upstream("A").await;
    let lifecycle = FakeLifecycle::new(ModelState::Stopped);
    let gw = start_wake_gateway(
        routes_for(upstream.port(), SceneMode::OnDemand, false),
        Some(lifecycle.clone()),
        Wake {
            wake_timeout: Duration::from_millis(200),
            ..Wake::default()
        },
    )
    .await;
    let (status, _, reply) = chat(&gw, "brain").await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(reply["error"]["code"], "wake_timeout");
    assert_eq!(reply["error"]["model_id"], "brain");
    assert_eq!(gw.in_flight.get("brain"), 0);

    lifecycle.release(Err("start failed: exit status 1"));
    let (status, headers, reply) = chat(&gw, "brain").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(headers[header::RETRY_AFTER], RETRY_AFTER_SEC);
    assert_eq!(reply["error"]["code"], "model_unavailable");
    assert_eq!(reply["error"]["type"], "server_error");
    assert!(reply["error"]["message"]
        .as_str()
        .unwrap()
        .contains("exit status 1"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sleeping_model_is_held_for_its_wake_deadline_and_a_stopped_one_for_the_cold_one() {
    let upstream = echo_upstream("A").await;
    // `serve.sleep.wake_timeout_sec: 3` against a 600 s gateway cold deadline.
    let reg: Registry = serde_yaml::from_str(&format!(
        "apiVersion: lumbridge/v1\nmodels:\n  brain:\n    name: brain\n    footprint_gb: 1\n    serve:\n      kind: vllm\n      port: {}\n      served_name: [osprey]\n      sleep: {{level: 2, wake_timeout_sec: 3}}\n",
        upstream.port()
    ))
    .unwrap();
    let modes = Map::from([("brain".to_string(), SceneMode::Sleepable)]);
    let routes = Routes::build_with_modes(&reg, &["brain".into()], &modes, |_| true);
    let lifecycle = FakeLifecycle::new(ModelState::Sleeping { level: 2 });
    lifecycle.release(Ok(()));
    let gw = start_wake_gateway(
        routes,
        Some(lifecycle.clone()),
        Wake {
            wake_timeout: Duration::from_secs(600),
            ..Wake::default()
        },
    )
    .await;

    let (status, _, _) = chat(&gw, "brain").await;
    assert_eq!(status, StatusCode::OK);
    *lifecycle.state.lock().unwrap() = ModelState::Waking;
    let (status, _, _) = chat(&gw, "brain").await;
    assert_eq!(status, StatusCode::OK);
    *lifecycle.state.lock().unwrap() = ModelState::Stopped;
    let (status, _, _) = chat(&gw, "brain").await;
    assert_eq!(status, StatusCode::OK);

    let budgets = lifecycle.budgets.lock().unwrap().clone();
    assert_eq!(budgets.len(), 3);
    for wake in &budgets[..2] {
        assert!(
            *wake > Duration::from_secs(2) && *wake <= Duration::from_secs(3),
            "asleep or waking: the wake deadline, got {wake:?}"
        );
    }
    assert!(
        budgets[2] > Duration::from_secs(590) && budgets[2] <= Duration::from_secs(600),
        "stopped: the cold-start deadline, got {:?}",
        budgets[2]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resident_down_is_still_503_and_off_scene_still_404_with_a_lifecycle() {
    let lifecycle = FakeLifecycle::new(ModelState::Stopped);
    let gw = start_wake_gateway(
        routes_for(9, SceneMode::Resident, false),
        Some(lifecycle.clone()),
        Wake::default(),
    )
    .await;
    let (status, headers, reply) = chat(&gw, "brain").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(headers[header::RETRY_AFTER], RETRY_AFTER_SEC);
    assert_eq!(reply["error"]["code"], "model_not_running");
    let (status, _, reply) = chat(&gw, "bge").await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "embed is resident too"
    );
    assert_eq!(reply["error"]["code"], "model_not_running");

    let (status, _, reply) = chat(&gw, "osprey-alt").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(reply["error"]["code"], "model_not_in_scene");
    assert_eq!(lifecycle.calls(), 0, "nothing was woken");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_a_lifecycle_a_dormant_on_demand_model_is_just_not_running() {
    let gw = start_wake_gateway(
        routes_for(9, SceneMode::OnDemand, false),
        None,
        Wake::default(),
    )
    .await;
    let (status, headers, reply) = chat(&gw, "brain").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(headers[header::RETRY_AFTER], RETRY_AFTER_SEC);
    assert_eq!(reply["error"]["code"], "model_not_running");
    let (_, _, reply) = send(gw.addr, Method::GET, INFLIGHT_PATH, "", "").await;
    assert!(reply.get("waiting").is_none(), "the endpoint is unchanged");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_leaving_during_a_streaming_wake_releases_its_counts() {
    let lifecycle = FakeLifecycle::new(ModelState::Stopped);
    let gw = start_wake_gateway(
        routes_for(9, SceneMode::OnDemand, false),
        Some(lifecycle.clone()),
        Wake::default(),
    )
    .await;
    let (mut body, _, task) = start_stream(&gw).await;
    next_chunk(&mut body).await.unwrap();
    assert_eq!(gw.in_flight.get("brain"), 1);
    task.abort();
    drop(body);
    let in_flight = gw.in_flight.clone();
    eventually("counts to drop", || {
        in_flight.get("brain") == 0 && in_flight.waiting_snapshot()["brain"] == 0
    })
    .await;
    lifecycle.release(Ok(()));
}
