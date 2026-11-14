//! The OpenAI-aware gateway (`--gateway-mode http`).
//!
//! One hyper server on its own tokio runtime, inside the agent's existing
//! gateway thread, so the rest of the agent stays synchronous. Per request it
//! reads the `model` (JSON body, multipart field or `?model=`), looks it up in the
//! route table, and forwards the original request to that model's upstream. The
//! response is never buffered: each upstream frame is handed to the client as it
//! arrives, and dropping the client drops the upstream request with it.
//!
//! With a [`Lifecycle`] (M3), a request for a `sleepable`/`on_demand` model that
//! isn't `Ready` is held while the model wakes; see [`hold`].

use anyhow::{Context, Result};
use bytes::Bytes;
use http_body_util::combinators::UnsyncBoxBody;
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::{Body, Frame, Incoming, SizeHint};
use hyper::header::{self, HeaderMap, HeaderName, HeaderValue};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode, Uri};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context as TaskContext, Poll};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use super::extract::{self, Peek};
use super::routes::{self, Lookup, Routes, Target};
use super::{InFlight, InFlightGuard, Options, WaitingGuard};
use crate::states::{Lifecycle, ModelState};

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type ReqBody = UnsyncBoxBody<Bytes, BoxError>;

/// How often the route table is rebuilt from the registry and state file.
const REFRESH_EVERY: Duration = Duration::from_secs(2);
/// Seconds a client should wait before retrying a model that isn't running.
const RETRY_AFTER_SEC: &str = "10";
const MODEL_HEADER: HeaderName = HeaderName::from_static("x-lumbridge-computer-model");
/// SSE comment keep-alive cadence while a streaming request waits for a wake.
pub const KEEPALIVE_EVERY: Duration = Duration::from_secs(5);
/// OpenAI endpoints that answer `"stream": true` with `text/event-stream`. Only
/// these get early SSE headers during a wake: anywhere else we can't promise the
/// upstream's answer will be an event stream.
const SSE_PATHS: &[&str] = &["/v1/chat/completions", "/v1/completions", "/v1/responses"];
/// Largest non-SSE upstream answer turned into a mid-stream error event.
const MAX_ERROR_BODY: usize = 64 * 1024;

pub struct Config {
    /// Where requests that carry no `model` and don't need one are sent.
    pub default_upstream: Option<String>,
    pub max_body_bytes: usize,
    /// Wakes sleeping/stopped models. `None`: the pre-M3 gateway, unchanged.
    pub lifecycle: Option<Arc<dyn Lifecycle>>,
    pub wake_timeout: Duration,
    pub max_held_per_model: u32,
    pub keepalive_every: Duration,
}

pub struct Shared {
    routes: RwLock<Arc<Routes>>,
    in_flight: InFlight,
    client: Client<HttpConnector, ReqBody>,
    config: Config,
}

impl Shared {
    pub fn new(routes: Routes, in_flight: InFlight, config: Config) -> Arc<Shared> {
        let mut connector = HttpConnector::new();
        connector.set_nodelay(true);
        connector.set_connect_timeout(Some(Duration::from_secs(5)));
        // No request or read timeout on purpose: a 128K-token prefill is silent
        // for minutes and a stream may run for many more.
        let client = Client::builder(TokioExecutor::new())
            .pool_idle_timeout(Duration::from_secs(30))
            .build(connector);
        Arc::new(Shared {
            routes: RwLock::new(Arc::new(routes)),
            in_flight,
            client,
            config,
        })
    }

    fn routes(&self) -> Arc<Routes> {
        self.routes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn set_routes(&self, routes: Routes) {
        *self.routes.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(routes);
    }
}

pub fn run(
    options: &Options,
    in_flight: InFlight,
    lifecycle: Option<Arc<dyn Lifecycle>>,
) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("lumbridge-gw") // Linux keeps 15 bytes of a thread name
        .enable_all()
        .build()
        .context("starting the gateway runtime")?;
    let root = options.root.clone();
    let listen = options.listen.clone();
    let config = Config {
        default_upstream: Some(options.upstream.clone()).filter(|u| !u.is_empty()),
        max_body_bytes: options.max_body_bytes,
        lifecycle,
        wake_timeout: options.wake_timeout,
        max_held_per_model: options.max_held_per_model,
        keepalive_every: KEEPALIVE_EVERY,
    };
    let on_demand = config.lifecycle.is_some();
    runtime.block_on(async move {
        let listener = TcpListener::bind(&listen)
            .await
            .with_context(|| format!("binding Lumbridge Computer gateway at {listen}"))?;
        // Build the first table before accepting, so the first request routes.
        let first = load_routes(root.clone(), on_demand)
            .await
            .unwrap_or_else(|error| {
                eprintln!("gateway: route table unavailable: {error:#}");
                Routes::default()
            });
        println!(
            "Lumbridge Computer gateway (http) {listen}: {}; default upstream {}",
            first.summary(),
            config.default_upstream.as_deref().unwrap_or("none")
        );
        let shared = Shared::new(first, in_flight, config);
        tokio::spawn(refresh_routes(root, shared.clone()));
        serve(listener, shared).await
    })
}

async fn load_routes(root: PathBuf, on_demand: bool) -> Result<Routes> {
    // File reads and port probes are blocking; keep them off the proxy workers.
    tokio::task::spawn_blocking(move || routes::load(&root, on_demand))
        .await
        .context("route table task panicked")?
}

async fn refresh_routes(root: PathBuf, shared: Arc<Shared>) {
    let mut last_summary = shared.routes().summary();
    let mut last_error = None;
    let on_demand = shared.config.lifecycle.is_some();
    loop {
        tokio::time::sleep(REFRESH_EVERY).await;
        match load_routes(root.clone(), on_demand).await {
            Ok(routes) => {
                last_error = None;
                let summary = routes.summary();
                if summary != last_summary {
                    println!("gateway routes: {summary}");
                    last_summary = summary;
                }
                shared.set_routes(routes);
            }
            // A half-written registry must not empty the table: keep serving the
            // last good one and say so once.
            Err(error) => {
                let message = format!("{error:#}");
                if last_error.as_ref() != Some(&message) {
                    eprintln!("gateway: keeping the previous route table: {message}");
                    last_error = Some(message);
                }
            }
        }
    }
}

pub async fn serve(listener: TcpListener, shared: Arc<Shared>) -> Result<()> {
    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(accepted) => accepted,
            Err(error) => {
                // EMFILE and friends: back off instead of spinning.
                eprintln!("gateway accept failed: {error}");
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
        };
        stream.set_nodelay(true).ok();
        let shared = shared.clone();
        tokio::spawn(async move {
            let service = service_fn(move |request| {
                let shared = shared.clone();
                async move { Ok::<_, Infallible>(handle(&shared, peer, request).await) }
            });
            // A connection error only ever affects this one client; hyper has
            // already torn the connection (and any upstream request) down.
            hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service)
                .await
                .ok();
        });
    }
}

async fn handle(
    shared: &Arc<Shared>,
    peer: SocketAddr,
    request: Request<Incoming>,
) -> Response<Tracked> {
    let mut trace = Trace::new(request.method(), request.uri());
    if extract::is_engine_control(request.uri().path()) {
        return trace.local(engine_control_forbidden(request.uri().path()));
    }
    match (request.method(), request.uri().path()) {
        (&Method::GET, super::INFLIGHT_PATH) => {
            return trace.local(match shared.config.lifecycle {
                None => inflight_response(&shared.in_flight, peer),
                Some(_) => inflight_with_waiting(&shared.in_flight, peer),
            })
        }
        (&Method::GET, "/health") => {
            return trace.local(json_response(
                StatusCode::OK,
                serde_json::json!({"status": "ok"}),
            ))
        }
        (&Method::GET, "/v1/models") => return trace.local(models_response(&shared.routes())),
        _ => {}
    }

    let (mut parts, body) = request.into_parts();
    let (body, peek) = match read_model(shared, &parts, body).await {
        Ok(read) => read,
        Err(response) => return trace.local(*response),
    };
    // Taken after the body is read: an upload can outlast a table refresh.
    let routes = shared.routes();
    trace.streamed = peek.stream;
    let query_model = parts
        .uri
        .query()
        .and_then(|q| extract::query_param(q, "model"))
        .filter(|m| !m.is_empty());
    let from_query = peek.model.is_none() && query_model.is_some();
    let model = peek.model.or(query_model);

    // Set when the named model is routable but not ready: hold and wake it.
    let mut wake: Option<Arc<dyn Lifecycle>> = None;
    let (addr, target) = match &model {
        Some(name) => match routes.lookup(name) {
            Lookup::Running(target) => {
                // A sleeping vLLM still accepts connections, so an open port is
                // not readiness for a model that sleeps: ask the lifecycle.
                if let Some(lifecycle) = &shared.config.lifecycle {
                    if target.mode.wakes_on_demand()
                        && lifecycle.state(&target.id) != ModelState::Ready
                    {
                        wake = Some(lifecycle.clone());
                    }
                }
                (target.addr.clone(), Some(target.clone()))
            }
            Lookup::Dormant(target) => match &shared.config.lifecycle {
                Some(lifecycle) => {
                    // The table is up to 2 s stale: a model that just started is
                    // dormant here but `Ready` to the lifecycle.
                    if lifecycle.state(&target.id) != ModelState::Ready {
                        wake = Some(lifecycle.clone());
                    }
                    (target.addr.clone(), Some(target.clone()))
                }
                None => {
                    trace.model = target.id.clone();
                    return trace.local(not_running(name, &target.id));
                }
            },
            Lookup::NotRunning { id } => {
                trace.model = id.to_string();
                return trace.local(not_running(name, id));
            }
            Lookup::NotInScene { id } => {
                trace.model = id.to_string();
                return trace.local(not_in_scene(name, id, routes.scene.as_deref()));
            }
            Lookup::Unknown => {
                trace.model = name.clone();
                return trace.local(unknown_model(name, &routes));
            }
        },
        // No model: forward to the default upstream, like a single vLLM serving
        // its default model would, so clients that omit `model` survive cutover.
        None => match &shared.config.default_upstream {
            Some(upstream) => {
                trace.model = "<default>".to_string();
                (upstream.clone(), routes.by_addr(upstream).cloned())
            }
            None if peek.invalid_json => {
                return trace.local(error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "invalid_json",
                    "the request body is not a JSON object".to_string(),
                    serde_json::Value::Null,
                ))
            }
            None => {
                return trace.local(error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "missing_model",
                    "no `model` in the request and no default upstream".to_string(),
                    serde_json::Value::Null,
                ))
            }
        },
    };

    let mut body = body;
    if let Some(target) = &target {
        if model.is_some() {
            trace.model = target.id.clone();
        }
        trace.guard = Some(shared.in_flight.begin(&target.id));
        // Only a request that named a model has one to rewrite; model-less
        // requests to the default upstream are counted but left alone.
        let rewrite_to = target.rewrite_to.as_deref();
        if let (Some(name), Some(rewrite_to)) = (&model, rewrite_to) {
            if name != rewrite_to {
                body = rewrite(&mut parts, body, from_query, rewrite_to);
            }
        }
        if let Some(lifecycle) = wake {
            let early_sse = peek.stream
                && parts.method == Method::POST
                && SSE_PATHS.contains(&parts.uri.path());
            let request = Held {
                parts,
                body,
                addr,
                target: target.clone(),
            };
            return hold(shared, trace, lifecycle, request, early_sse).await;
        }
    }

    forward(shared, trace, parts, body, &addr, target.as_deref()).await
}

/// A request parked until its model is ready.
struct Held {
    parts: hyper::http::request::Parts,
    body: ReadBody,
    addr: String,
    target: Arc<Target>,
}

/// Why a held request's model didn't become ready.
enum WakeError {
    /// The deadline passed (ours or the lifecycle's own): 504.
    Timeout(String),
    /// The lifecycle refused or failed (can't admit, start failed): 503.
    Failed(String),
}

/// Hold a request while `lifecycle` wakes or starts its model, then forward it.
///
/// - The request counts in `InFlight` the whole time (its guard is already in
///   `trace`), so the idle sleeper never sleeps a model someone waits for, and
///   in the separate `waiting` count until the model is ready.
/// - Beyond `max_held_per_model` waiting requests: 503 + `Retry-After`.
/// - Non-streaming: wait silently; 504 `wake_timeout` at the deadline, 503
///   `model_unavailable` if the wake fails, otherwise the upstream's response.
/// - Streaming to an SSE endpoint (`early_sse`): 200 + `text/event-stream` now,
///   `: lumbridge-computer waking <model> <n>s` comments every `keepalive_every`, then the
///   upstream's events in the same body. A failed wake (or an upstream that
///   answers with something other than an event stream) becomes one
///   `data: {"error":{...}}` event, after which the stream ends.
async fn hold(
    shared: &Arc<Shared>,
    mut trace: Trace,
    lifecycle: Arc<dyn Lifecycle>,
    request: Held,
    early_sse: bool,
) -> Response<Tracked> {
    let id = request.target.id.clone();
    let (waiting, held) = shared.in_flight.begin_waiting(&id);
    if held > shared.config.max_held_per_model {
        drop(waiting);
        return trace.local(wake_queue_full(&id, shared.config.max_held_per_model));
    }
    let deadline = Instant::now() + hold_budget(&shared.config, &lifecycle, &request.target);

    if !early_sse {
        let woke = wait_ready(lifecycle, &id, deadline).await;
        drop(waiting);
        return match woke {
            Ok(()) => {
                let Held {
                    parts,
                    body,
                    addr,
                    target,
                } = request;
                forward(shared, trace, parts, body, &addr, Some(&target)).await
            }
            Err(WakeError::Timeout(reason)) => trace.local(wake_timeout(&id, &reason)),
            Err(WakeError::Failed(reason)) => trace.local(model_unavailable(&id, &reason)),
        };
    }

    let (tx, rx) = mpsc::channel(16);
    tokio::spawn(stream_after_wake(
        shared.clone(),
        tx,
        lifecycle,
        request,
        waiting,
        deadline,
    ));
    let mut response = Response::new(ChannelFrames(rx).boxed_unsync());
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(
        HeaderName::from_static("x-accel-buffering"),
        HeaderValue::from_static("no"),
    );
    if let Ok(value) = HeaderValue::from_str(&id) {
        headers.insert(MODEL_HEADER, value);
    }
    trace.streamed = true;
    trace.local(response)
}

/// How long to hold a request for `target`, by what its model is doing now: a
/// wake from sleep gets the model's `serve.sleep.wake_timeout_sec` (seconds,
/// per the spike), anything else (stopped, starting, draining) the gateway's
/// cold-start deadline (`--wake-timeout-sec`, minutes for a 35B brain).
fn hold_budget(config: &Config, lifecycle: &Arc<dyn Lifecycle>, target: &Target) -> Duration {
    match lifecycle.state(&target.id) {
        ModelState::Sleeping { .. } | ModelState::Waking => {
            target.sleep_wake_timeout.unwrap_or(config.wake_timeout)
        }
        _ => config.wake_timeout,
    }
}

/// Call the blocking `ensure_ready` off the runtime, bounded by `deadline`.
/// If the client leaves, this future is dropped; the blocking call carries on
/// (it is shared and idempotent), and the next request finds the model waking.
async fn wait_ready(
    lifecycle: Arc<dyn Lifecycle>,
    id: &str,
    deadline: Instant,
) -> Result<(), WakeError> {
    let started = Instant::now();
    let task = tokio::task::spawn_blocking({
        let id = id.to_string();
        move || lifecycle.ensure_ready(&id, deadline)
    });
    let outcome = match tokio::time::timeout_at(deadline.into(), task).await {
        Err(_) => Err(WakeError::Timeout(format!(
            "not ready within {}s",
            deadline.saturating_duration_since(started).as_secs()
        ))),
        Ok(Err(join)) => Err(WakeError::Failed(format!("wake task failed: {join}"))),
        Ok(Ok(Err(error))) if Instant::now() >= deadline => {
            Err(WakeError::Timeout(format!("{error:#}")))
        }
        Ok(Ok(Err(error))) => Err(WakeError::Failed(format!("{error:#}"))),
        Ok(Ok(Ok(()))) => Ok(()),
    };
    let secs = started.elapsed().as_secs_f64();
    match &outcome {
        Ok(()) => eprintln!("gateway wake {id}: ready after {secs:.1}s"),
        Err(WakeError::Timeout(reason) | WakeError::Failed(reason)) => {
            eprintln!("gateway wake {id}: failed after {secs:.1}s: {reason}")
        }
    }
    outcome
}

type FrameTx = mpsc::Sender<Result<Frame<Bytes>, BoxError>>;

/// The streaming half of [`hold`]: runs as its own task feeding the response
/// body that was already handed to the client. Every send fails once the client
/// is gone, and `tx.closed()` notices a silent departure, so the task (and with
/// it the wake wait and any upstream request) stops as soon as the client does.
async fn stream_after_wake(
    shared: Arc<Shared>,
    tx: FrameTx,
    lifecycle: Arc<dyn Lifecycle>,
    request: Held,
    waiting: WaitingGuard,
    deadline: Instant,
) {
    let Held {
        parts,
        body,
        addr,
        target,
    } = request;
    let id = target.id.clone();
    let started = Instant::now();
    let comment = |secs: u64| -> Result<Frame<Bytes>, BoxError> {
        Ok(Frame::data(Bytes::from(format!(
            ": lumbridge-computer waking {id} {secs}s\n\n"
        ))))
    };
    // One comment straight away, so the client (and any proxy) sees bytes.
    if tx.send(comment(0)).await.is_err() {
        return;
    }
    let every = shared.config.keepalive_every;
    let mut ticker = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
    let wake = wait_ready(lifecycle, &id, deadline);
    tokio::pin!(wake);
    let woke = loop {
        tokio::select! {
            woke = &mut wake => break woke,
            _ = ticker.tick() => {
                if tx.send(comment(started.elapsed().as_secs())).await.is_err() {
                    return;
                }
            }
            _ = tx.closed() => return,
        }
    };
    drop(waiting);
    if let Err(error) = woke {
        let (code, reason) = match error {
            WakeError::Timeout(reason) => ("wake_timeout", reason),
            WakeError::Failed(reason) => ("model_unavailable", reason),
        };
        let message = format!("The model `{id}` could not be woken: {reason}");
        tx.send(error_event(code, &message, &id)).await.ok();
        return;
    }

    let response = tokio::select! {
        response = send_upstream(&shared, parts, body, &addr) => response,
        _ = tx.closed() => return,
    };
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            let message = format!("upstream {addr} failed: {error}");
            tx.send(error_event("upstream_unavailable", &message, &id))
                .await
                .ok();
            return;
        }
    };
    let is_sse = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));
    let status = response.status();
    let mut body = response.into_body();
    if !(status.is_success() && is_sse) {
        // Our 200 is already on the wire: report the upstream's answer (usually
        // a 400 validation error) as an error event instead.
        let bytes = Limited::new(body, MAX_ERROR_BODY)
            .collect()
            .await
            .map(|c| c.to_bytes())
            .unwrap_or_default();
        tx.send(Ok(Frame::data(upstream_error_event(status, &bytes))))
            .await
            .ok();
        return;
    }
    loop {
        tokio::select! {
            frame = body.frame() => match frame {
                Some(Ok(frame)) => {
                    if frame.is_data() && tx.send(Ok(frame)).await.is_err() {
                        return;
                    }
                }
                // Abort the client stream, as `forward` does, so a truncated
                // answer is never mistaken for a complete one.
                Some(Err(error)) => {
                    tx.send(Err(BoxError::from(error))).await.ok();
                    return;
                }
                None => return,
            },
            _ = tx.closed() => return,
        }
    }
}

/// `data: {"error":{...}}`, the OpenAI shape for an error inside a stream.
fn error_event(code: &str, message: &str, id: &str) -> Result<Frame<Bytes>, BoxError> {
    let event = serde_json::json!({"error": {
        "message": message,
        "type": "server_error",
        "param": "model",
        "code": code,
        "model_id": id,
    }});
    Ok(Frame::data(Bytes::from(format!("data: {event}\n\n"))))
}

/// An upstream's non-SSE answer as one SSE error event. OpenAI-shaped bodies
/// pass through; vLLM's flat `{"object":"error","message":..}` is wrapped.
fn upstream_error_event(status: StatusCode, body: &[u8]) -> Bytes {
    let event = match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(value) if value.get("error").is_some() => value,
        Ok(value) if value.get("message").is_some() => serde_json::json!({ "error": value }),
        _ => serde_json::json!({"error": {
            "message": format!(
                "upstream answered {status}: {}",
                String::from_utf8_lossy(body)
            ),
            "type": "server_error",
            "code": "upstream_error",
        }}),
    };
    Bytes::from(format!("data: {event}\n\n"))
}

/// A response body fed by a task: lets [`hold`] answer with headers before the
/// model (and so the upstream response) exists.
struct ChannelFrames(mpsc::Receiver<Result<Frame<Bytes>, BoxError>>);

impl Body for ChannelFrames {
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
        self.0.poll_recv(cx)
    }
}

/// The request body, buffered only when `model` has to be read out of it.
enum ReadBody {
    Buffered(Bytes),
    Streaming(Incoming),
}

async fn read_model(
    shared: &Shared,
    parts: &hyper::http::request::Parts,
    body: Incoming,
) -> Result<(ReadBody, Peek), Box<Response<ReqBody>>> {
    let content_type = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let is_json = content_type
        .split(';')
        .next()
        .is_some_and(|mime| mime.trim().to_ascii_lowercase().ends_with("json"));
    let boundary = extract::multipart_boundary(content_type);
    let has_body = matches!(parts.method, Method::POST | Method::PUT | Method::PATCH);
    if !has_body || (!is_json && boundary.is_none()) {
        return Ok((ReadBody::Streaming(body), Peek::default()));
    }

    let cap = shared.config.max_body_bytes;
    let declared = parts
        .headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if declared.is_some_and(|len| len > cap as u64) {
        return Err(Box::new(too_large(cap)));
    }
    let bytes = match Limited::new(body, cap).collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) if error.is::<http_body_util::LengthLimitError>() => {
            return Err(Box::new(too_large(cap)))
        }
        Err(error) => {
            return Err(Box::new(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "body_read_failed",
                format!("reading the request body failed: {error}"),
                serde_json::Value::Null,
            )))
        }
    };
    let peek = match boundary {
        Some(boundary) => extract::multipart(&bytes, boundary),
        None => extract::json(&bytes),
    };
    Ok((ReadBody::Buffered(bytes), peek))
}

fn rewrite(
    parts: &mut hyper::http::request::Parts,
    body: ReadBody,
    from_query: bool,
    model: &str,
) -> ReadBody {
    if from_query {
        let path = parts.uri.path();
        let query = parts.uri.query().unwrap_or("");
        let rewritten = format!("{path}?{}", extract::rewrite_query(query, "model", model));
        if let Ok(uri) = rewritten.parse::<Uri>() {
            parts.uri = uri;
        }
        return body;
    }
    let ReadBody::Buffered(bytes) = body else {
        return body;
    };
    let content_type = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let rewritten = match extract::multipart_boundary(content_type) {
        Some(boundary) => extract::multipart_field(&bytes, boundary, "model").map(|range| {
            let mut out = Vec::with_capacity(bytes.len() + model.len());
            out.extend_from_slice(&bytes[..range.start]);
            out.extend_from_slice(model.as_bytes());
            out.extend_from_slice(&bytes[range.end..]);
            out
        }),
        None => extract::rewrite_json(&bytes, model),
    };
    match rewritten {
        Some(out) => {
            parts
                .headers
                .insert(header::CONTENT_LENGTH, HeaderValue::from(out.len()));
            ReadBody::Buffered(Bytes::from(out))
        }
        None => ReadBody::Buffered(bytes),
    }
}

/// Send the client's request to `addr` as a well-behaved HTTP/1.1 proxy would.
async fn send_upstream(
    shared: &Shared,
    mut parts: hyper::http::request::Parts,
    body: ReadBody,
    addr: &str,
) -> Result<Response<Incoming>, String> {
    let path_and_query = parts
        .uri
        .path_and_query()
        .map_or("/", |pq| pq.as_str())
        .to_string();
    parts.uri = format!("http://{addr}{path_and_query}")
        .parse()
        .map_err(|error: hyper::http::uri::InvalidUri| error.to_string())?;
    parts.version = hyper::Version::HTTP_11;
    strip_hop_by_hop(&mut parts.headers);
    if let Ok(host) = HeaderValue::from_str(addr) {
        parts.headers.insert(header::HOST, host);
    }
    let body: ReqBody = match body {
        ReadBody::Buffered(bytes) => Full::new(bytes).map_err(|e| match e {}).boxed_unsync(),
        ReadBody::Streaming(incoming) => incoming.map_err(BoxError::from).boxed_unsync(),
    };

    shared
        .client
        .request(Request::from_parts(parts, body))
        .await
        .map_err(|error| format!("{error:#}"))
}

async fn forward(
    shared: &Shared,
    mut trace: Trace,
    parts: hyper::http::request::Parts,
    body: ReadBody,
    addr: &str,
    target: Option<&Target>,
) -> Response<Tracked> {
    let response = match send_upstream(shared, parts, body, addr).await {
        Ok(response) => response,
        Err(error) => {
            trace.outcome = "upstream-error";
            return trace.local(bad_gateway(addr, &error));
        }
    };
    let (mut parts, body) = response.into_parts();
    strip_hop_by_hop(&mut parts.headers);
    let event_stream = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));
    if event_stream {
        // Tell any proxy in front of us (nginx, Tailscale serve) not to buffer.
        parts.headers.insert(
            HeaderName::from_static("x-accel-buffering"),
            HeaderValue::from_static("no"),
        );
        parts
            .headers
            .entry(header::CACHE_CONTROL)
            .or_insert(HeaderValue::from_static("no-cache"));
    }
    if let Some(value) = target.and_then(|t| HeaderValue::from_str(&t.id).ok()) {
        parts.headers.insert(MODEL_HEADER, value);
    }
    trace.status = Some(parts.status.as_u16());
    trace.streamed |= event_stream || !parts.headers.contains_key(header::CONTENT_LENGTH);
    Response::from_parts(
        parts,
        Tracked::new(body.map_err(BoxError::from).boxed_unsync(), trace),
    )
}

/// RFC 9110 §7.6.1 connection-specific headers: they describe one hop, and
/// hyper sets its own framing on each side.
fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let named: Vec<HeaderName> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in named {
        headers.remove(name);
    }
    for name in [
        header::CONNECTION,
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
        header::PROXY_AUTHENTICATE,
        header::PROXY_AUTHORIZATION,
    ] {
        headers.remove(name);
    }
    headers.remove("keep-alive");
    headers.remove("proxy-connection");
}

/// One request's bookkeeping. Holds the in-flight guard and writes the log line
/// when dropped, so both happen exactly once however the request ends: body
/// complete, upstream error, or the client vanishing mid-stream.
struct Trace {
    started: Instant,
    method: Method,
    path: String,
    model: String,
    status: Option<u16>,
    bytes: u64,
    streamed: bool,
    outcome: &'static str,
    guard: Option<InFlightGuard>,
}

impl Trace {
    fn new(method: &Method, uri: &Uri) -> Trace {
        Trace {
            started: Instant::now(),
            method: method.clone(),
            path: uri.path().to_string(),
            model: "-".to_string(),
            status: None,
            bytes: 0,
            streamed: false,
            // Overwritten on every normal path; left as-is only when hyper drops
            // the request before a response exists (the client went away).
            outcome: "client-gone",
            guard: None,
        }
    }

    fn local(mut self, response: Response<ReqBody>) -> Response<Tracked> {
        self.status = Some(response.status().as_u16());
        if self.outcome == "client-gone" {
            self.outcome = "ok";
        }
        let (parts, body) = response.into_parts();
        Response::from_parts(parts, Tracked::new(body, self))
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        let status = self
            .status
            .map_or_else(|| "-".to_string(), |s| s.to_string());
        eprintln!(
            "gateway {} {} model={} status={status} bytes={} ms={} streamed={} {}",
            self.method,
            self.path,
            self.model,
            self.bytes,
            self.started.elapsed().as_millis(),
            self.streamed,
            self.outcome
        );
    }
}

/// The response body handed to hyper: upstream frames pass straight through,
/// counted, and the [`Trace`] is released the moment the body ends or is dropped.
pub struct Tracked {
    inner: ReqBody,
    trace: Option<Trace>,
}

impl Tracked {
    fn new(inner: ReqBody, mut trace: Trace) -> Tracked {
        if trace.outcome == "client-gone" {
            // From here on, a drop before end-of-stream is a client abort.
            trace.outcome = "client-abort";
        }
        Tracked {
            inner,
            trace: Some(trace),
        }
    }

    fn finish(&mut self, outcome: &'static str) {
        if let Some(mut trace) = self.trace.take() {
            if trace.outcome != "upstream-error" {
                trace.outcome = outcome;
            }
        }
    }
}

impl Body for Tracked {
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
        let polled = Pin::new(&mut self.inner).poll_frame(cx);
        match &polled {
            Poll::Ready(Some(Ok(frame))) => {
                let len = frame.data_ref().map_or(0, |data| data.len() as u64);
                if let Some(trace) = self.trace.as_mut() {
                    trace.bytes += len;
                }
                if self.inner.is_end_stream() {
                    self.finish("ok");
                }
            }
            Poll::Ready(Some(Err(_))) => self.finish("upstream-error"),
            Poll::Ready(None) => self.finish("ok"),
            Poll::Pending => {}
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        // hyper may skip polling a body that already reports end-of-stream.
        let outcome = if self.inner.is_end_stream() {
            "ok"
        } else {
            "client-abort"
        };
        self.finish(outcome);
    }
}

fn full(bytes: impl Into<Bytes>) -> ReqBody {
    Full::new(bytes.into())
        .map_err(|e| match e {})
        .boxed_unsync()
}

fn json_response(status: StatusCode, value: serde_json::Value) -> Response<ReqBody> {
    let mut response = Response::new(full(value.to_string()));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// OpenAI's error envelope, so SDKs surface the message instead of a parse error.
fn error_response(
    status: StatusCode,
    kind: &str,
    code: &str,
    message: String,
    extra: serde_json::Value,
) -> Response<ReqBody> {
    let mut error = serde_json::json!({
        "message": message,
        "type": kind,
        "param": "model",
        "code": code,
    });
    if let (Some(error), serde_json::Value::Object(extra)) = (error.as_object_mut(), extra) {
        error.extend(extra);
    }
    json_response(status, serde_json::json!({ "error": error }))
}

fn available(routes: &Routes) -> Vec<&str> {
    routes
        .listed()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect()
}

fn unknown_model(name: &str, routes: &Routes) -> Response<ReqBody> {
    let names = available(routes);
    error_response(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "model_not_found",
        format!(
            "The model `{name}` does not exist on this Lumbridge Computer. Available: {}",
            if names.is_empty() {
                "none running".to_string()
            } else {
                names.join(", ")
            }
        ),
        serde_json::json!({ "available_models": names }),
    )
}

fn not_running(name: &str, id: &str) -> Response<ReqBody> {
    let mut response = error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_error",
        "model_not_running",
        format!("The model `{name}` (registry id `{id}`) is not running in the active Scene"),
        serde_json::json!({ "model_id": id }),
    );
    response.headers_mut().insert(
        header::RETRY_AFTER,
        HeaderValue::from_static(RETRY_AFTER_SEC),
    );
    response
}

fn wake_queue_full(id: &str, cap: u32) -> Response<ReqBody> {
    let mut response = error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_error",
        "wake_queue_full",
        format!("The model `{id}` is waking and already has {cap} requests waiting for it"),
        serde_json::json!({ "model_id": id }),
    );
    response.headers_mut().insert(
        header::RETRY_AFTER,
        HeaderValue::from_static(RETRY_AFTER_SEC),
    );
    response
}

fn model_unavailable(id: &str, reason: &str) -> Response<ReqBody> {
    let mut response = error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_error",
        "model_unavailable",
        format!("The model `{id}` could not be woken: {reason}"),
        serde_json::json!({ "model_id": id }),
    );
    response.headers_mut().insert(
        header::RETRY_AFTER,
        HeaderValue::from_static(RETRY_AFTER_SEC),
    );
    response
}

fn wake_timeout(id: &str, reason: &str) -> Response<ReqBody> {
    error_response(
        StatusCode::GATEWAY_TIMEOUT,
        "server_error",
        "wake_timeout",
        format!("The model `{id}` did not become ready in time: {reason}"),
        serde_json::json!({ "model_id": id }),
    )
}

fn not_in_scene(name: &str, id: &str, scene: Option<&str>) -> Response<ReqBody> {
    let scene_text = scene.map_or_else(
        || "no Scene is active".to_string(),
        |scene| format!("it is not in the active Scene `{scene}`"),
    );
    error_response(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "model_not_in_scene",
        format!("The model `{name}` (registry id `{id}`) is installed but {scene_text}"),
        serde_json::json!({ "model_id": id, "scene": scene }),
    )
}

fn engine_control_forbidden(path: &str) -> Response<ReqBody> {
    error_response(
        StatusCode::FORBIDDEN,
        "invalid_request_error",
        "engine_control_forbidden",
        format!(
            "`{path}` is an engine-control route (vLLM dev mode); it is not served through \
             the Lumbridge Computer gateway"
        ),
        serde_json::Value::Null,
    )
}

fn too_large(cap: usize) -> Response<ReqBody> {
    error_response(
        StatusCode::PAYLOAD_TOO_LARGE,
        "invalid_request_error",
        "body_too_large",
        format!("request body exceeds the gateway's {cap}-byte limit"),
        serde_json::Value::Null,
    )
}

fn bad_gateway(addr: &str, error: &str) -> Response<ReqBody> {
    error_response(
        StatusCode::BAD_GATEWAY,
        "server_error",
        "upstream_unavailable",
        format!("upstream {addr} failed: {error}"),
        serde_json::Value::Null,
    )
}

/// In-flight counts per model id, for `lumbridge-computer status`. The gateway may listen on
/// 0.0.0.0 for the tailnet, but which models are busy right now is the operator's
/// business, so only callers on this machine get an answer.
fn inflight_response(in_flight: &InFlight, peer: SocketAddr) -> Response<ReqBody> {
    if !is_loopback(peer) {
        return error_response(
            StatusCode::FORBIDDEN,
            "permission_error",
            "loopback_only",
            format!("{} answers loopback callers only", super::INFLIGHT_PATH),
            serde_json::Value::Null,
        );
    }
    json_response(
        StatusCode::OK,
        serde_json::json!({
            "in_flight": in_flight.snapshot(),
            "last_finished_ms": in_flight.last_finished_snapshot(),
        }),
    )
}

/// `inflight_response` plus the requests held while their model wakes (M3,
/// additive: `in_flight` keeps its meaning and already includes them).
fn inflight_with_waiting(in_flight: &InFlight, peer: SocketAddr) -> Response<ReqBody> {
    let response = inflight_response(in_flight, peer);
    if response.status() != StatusCode::OK {
        return response;
    }
    json_response(
        StatusCode::OK,
        serde_json::json!({
            "in_flight": in_flight.snapshot(),
            "waiting": in_flight.waiting_snapshot(),
            "last_finished_ms": in_flight.last_finished_snapshot(),
        }),
    )
}

fn is_loopback(peer: SocketAddr) -> bool {
    match peer {
        SocketAddr::V4(v4) => v4.ip().is_loopback(),
        // A dual-stack listener reports 127.0.0.1 as ::ffff:127.0.0.1.
        SocketAddr::V6(v6) => {
            v6.ip().is_loopback() || v6.ip().to_ipv4_mapped().is_some_and(|ip| ip.is_loopback())
        }
    }
}

fn models_response(routes: &Routes) -> Response<ReqBody> {
    let data: Vec<serde_json::Value> = routes
        .listed()
        .iter()
        .map(|(name, id)| {
            serde_json::json!({
                "id": name,
                "object": "model",
                "created": routes.created,
                "owned_by": "lumbridge-computer",
                "root": id,
            })
        })
        .collect();
    json_response(
        StatusCode::OK,
        serde_json::json!({ "object": "list", "data": data }),
    )
}

#[cfg(test)]
#[path = "http_tests.rs"]
mod tests;
