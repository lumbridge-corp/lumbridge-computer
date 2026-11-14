//! HTTP gateway tests against fake upstreams on 127.0.0.1:0 (no network).

use super::*;
use crate::config::Registry;
use hyper::client::conn::http1 as client_conn;
use std::future::Future;
use std::net::SocketAddr;
use tokio::sync::mpsc;

/// A response or request body fed from a channel, so a test controls exactly
/// when each chunk exists and can observe the reader going away.
struct ChannelBody(mpsc::Receiver<Bytes>);

impl Body for ChannelBody {
    type Data = Bytes;
    type Error = Infallible;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        self.0
            .poll_recv(cx)
            .map(|chunk| chunk.map(|c| Ok(Frame::data(c))))
    }
}

type FakeBody = UnsyncBoxBody<Bytes, Infallible>;

/// Serve `handler` on a fresh loopback port.
async fn fake_upstream<F, Fut>(handler: F) -> SocketAddr
where
    F: Fn(Request<Incoming>) -> Fut + Send + Sync + Clone + 'static,
    Fut: Future<Output = Response<FakeBody>> + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.unwrap();
            let handler = handler.clone();
            tokio::spawn(async move {
                let service = service_fn(move |request| {
                    let handler = handler.clone();
                    async move { Ok::<_, Infallible>(handler(request).await) }
                });
                hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .await
                    .ok();
            });
        }
    });
    addr
}

/// An upstream that answers with its name, the path it saw and the exact body.
async fn echo_upstream(name: &'static str) -> SocketAddr {
    fake_upstream(move |request: Request<Incoming>| async move {
        let path = request
            .uri()
            .path_and_query()
            .map(|pq| pq.to_string())
            .unwrap_or_default();
        let body = request.into_body().collect().await.unwrap().to_bytes();
        let reply = serde_json::json!({
            "upstream": name,
            "path": path,
            "body": String::from_utf8_lossy(&body),
        });
        Response::new(Full::new(Bytes::from(reply.to_string())).boxed_unsync())
    })
    .await
}

fn registry(models: &[(&str, u16, &[&str], &str)]) -> Registry {
    let mut yaml = String::from("apiVersion: lumbridge/v1\nmodels:\n");
    for (id, port, names, extra) in models {
        yaml.push_str(&format!(
            "  {id}:\n    name: {id}\n    footprint_gb: 1\n{extra}    serve:\n      kind: exec\n      port: {port}\n      served_name: [{}]\n",
            names.join(", ")
        ));
    }
    serde_yaml::from_str(&yaml).unwrap()
}

struct Gateway {
    addr: SocketAddr,
    in_flight: InFlight,
}

async fn start_gateway(routes: Routes, default_upstream: Option<String>, cap: usize) -> Gateway {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let in_flight = InFlight::default();
    let shared = Shared::new(
        routes,
        in_flight.clone(),
        Config {
            default_upstream,
            max_body_bytes: cap,
            lifecycle: None,
            wake_timeout: Duration::from_secs(600),
            max_held_per_model: 32,
            keepalive_every: KEEPALIVE_EVERY,
        },
    );
    tokio::spawn(serve(listener, shared));
    Gateway { addr, in_flight }
}

/// One client connection; the returned task owns the socket, so aborting it is
/// a client vanishing.
async fn connect<B>(addr: SocketAddr) -> (client_conn::SendRequest<B>, tokio::task::JoinHandle<()>)
where
    B: Body + Send + 'static,
    B::Data: Send,
    B::Error: Into<BoxError>,
{
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (sender, conn) = client_conn::handshake(TokioIo::new(stream)).await.unwrap();
    let task = tokio::spawn(async move {
        conn.await.ok();
    });
    (sender, task)
}

async fn send(
    addr: SocketAddr,
    method: Method,
    uri: &str,
    content_type: &str,
    body: impl Into<Bytes>,
) -> (StatusCode, HeaderMap, serde_json::Value) {
    let (mut sender, _task) = connect::<Full<Bytes>>(addr).await;
    let mut request = Request::builder()
        .method(method)
        .uri(uri)
        .header("host", "lumbridge-computer");
    if !content_type.is_empty() {
        request = request.header("content-type", content_type);
    }
    let response = sender
        .send_request(request.body(Full::new(body.into())).unwrap())
        .await
        .unwrap();
    let (parts, body) = response.into_parts();
    let bytes = body.collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (parts.status, parts.headers, json)
}

async fn eventually(what: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

const MIB: usize = 1 << 20;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requests_route_by_model_across_two_upstreams_with_bytes_untouched() {
    let a = echo_upstream("A").await;
    let b = echo_upstream("B").await;
    let reg = registry(&[
        ("brain", a.port(), &["brain", "osprey"], ""),
        ("brain-alt", b.port(), &["osprey-alt"], ""),
    ]);
    let routes = Routes::build(&reg, &["brain".into(), "brain-alt".into()], |_| true);
    let gw = start_gateway(routes, None, MIB).await;

    let body = r#"{ "model" : "osprey-alt", "messages": [], "stream": false }"#;
    let (status, headers, reply) = send(
        gw.addr,
        Method::POST,
        "/v1/chat/completions?x=1",
        "application/json",
        body,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["upstream"], "B");
    assert_eq!(reply["path"], "/v1/chat/completions?x=1");
    assert_eq!(
        reply["body"], body,
        "the client's bytes are forwarded verbatim"
    );
    assert_eq!(headers[MODEL_HEADER], "brain-alt");

    let (_, headers, reply) = send(
        gw.addr,
        Method::POST,
        "/v1/chat/completions",
        "application/json; charset=utf-8",
        r#"{"model":"osprey"}"#,
    )
    .await;
    assert_eq!(reply["upstream"], "A");
    assert_eq!(headers[MODEL_HEADER], "brain");

    let (_, _, reply) = send(gw.addr, Method::GET, "/v1/whatever?model=brain-alt", "", "").await;
    assert_eq!(reply["upstream"], "B", "GET routes by ?model=");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn engine_control_routes_are_refused_with_or_without_a_model() {
    // A sleepable vLLM runs in dev mode on 127.0.0.1; the gateway listens on
    // the tailnet, so it must never become the way to /collective_rpc.
    let a = echo_upstream("A").await;
    let reg = registry(&[("brain", a.port(), &["osprey"], "")]);
    let routes = Routes::build(&reg, &["brain".into()], |_| true);
    let gw = start_gateway(routes, Some(a.to_string()), MIB).await;
    for (uri, body) in [
        (
            "/collective_rpc",
            r#"{"model":"brain","method":"reload_all_fast"}"#,
        ),
        ("/collective_rpc", r#"{"method":"execute_model"}"#),
        ("/sleep?level=1", ""),
        ("/wake_up?model=brain", ""),
        ("/%63ollective_rpc", "{}"),
    ] {
        let (status, _, reply) = send(
            gw.addr,
            Method::POST,
            uri,
            "application/json",
            body.to_string(),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{uri}");
        assert_eq!(reply["error"]["code"], "engine_control_forbidden", "{uri}");
        assert!(reply.get("upstream").is_none(), "{uri} reached the engine");
    }
    assert_eq!(gw.in_flight.get("brain"), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_rewrite_sends_the_upstream_its_own_name() {
    let a = echo_upstream("A").await;
    let reg = registry(&[(
        "brain-alt",
        a.port(),
        &["osprey-alt", "alt"],
        "    gateway:\n      rewrite_model: true\n",
    )]);
    let routes = Routes::build(&reg, &["brain-alt".into()], |_| true);
    let gw = start_gateway(routes, None, MIB).await;

    let (_, _, reply) = send(
        gw.addr,
        Method::POST,
        "/v1/completions",
        "application/json",
        r#"{"model":"alt","prompt":"hi"}"#,
    )
    .await;
    let forwarded: serde_json::Value =
        serde_json::from_str(reply["body"].as_str().unwrap()).unwrap();
    assert_eq!(
        forwarded,
        serde_json::json!({"model": "osprey-alt", "prompt": "hi"})
    );

    let (_, _, reply) = send(gw.addr, Method::GET, "/x?model=alt&y=2", "", "").await;
    assert_eq!(reply["path"], "/x?model=osprey-alt&y=2");
}

fn multipart_body(boundary: &str, model: &str) -> Vec<u8> {
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
    )
    .into_bytes();
    body.extend((0..=255u8).cycle().take(50_000));
    body.extend_from_slice(
        format!(
            "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n--{boundary}--\r\n"
        )
        .as_bytes(),
    );
    body
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multipart_transcriptions_route_by_their_model_field() {
    let asr = echo_upstream("asr").await;
    let tts = echo_upstream("tts").await;
    let reg = registry(&[
        ("speech-asr", asr.port(), &["whisper-1"], ""),
        ("qwen-tts", tts.port(), &["tts-1"], ""),
    ]);
    let routes = Routes::build(&reg, &["speech-asr".into(), "qwen-tts".into()], |_| true);
    let gw = start_gateway(routes, None, MIB).await;

    let boundary = "XyZ123";
    let body = multipart_body(boundary, "whisper-1");
    let (status, headers, reply) = send(
        gw.addr,
        Method::POST,
        "/v1/audio/transcriptions",
        &format!("multipart/form-data; boundary={boundary}"),
        body.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["upstream"], "asr");
    assert_eq!(headers[MODEL_HEADER], "speech-asr");
    assert_eq!(
        reply["body"].as_str().unwrap(),
        String::from_utf8_lossy(&body)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn error_responses_are_openai_shaped() {
    let a = echo_upstream("A").await;
    let reg = registry(&[
        ("brain", a.port(), &["osprey"], ""),
        ("embed", 1, &["bge"], ""),
        ("brain-alt", 2, &["osprey-alt"], ""),
    ]);
    let mut routes = Routes::build(&reg, &["brain".into(), "embed".into()], |m| {
        m.serve.port != Some(1)
    });
    routes.scene = Some("dual".into());
    let gw = start_gateway(routes, None, MIB).await;
    let chat = "/v1/chat/completions";

    let (status, _, reply) = send(
        gw.addr,
        Method::POST,
        chat,
        "application/json",
        r#"{"model":"gpt-9"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(reply["error"]["code"], "model_not_found");
    assert_eq!(reply["error"]["type"], "invalid_request_error");
    assert_eq!(
        reply["error"]["available_models"],
        serde_json::json!(["brain", "osprey"])
    );
    assert!(reply["error"]["message"]
        .as_str()
        .unwrap()
        .contains("osprey"));

    let (status, headers, reply) = send(
        gw.addr,
        Method::POST,
        chat,
        "application/json",
        r#"{"model":"bge"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(headers[header::RETRY_AFTER], RETRY_AFTER_SEC);
    assert_eq!(reply["error"]["code"], "model_not_running");
    assert_eq!(reply["error"]["model_id"], "embed");

    // Installed but outside the Scene: it won't come back by retrying.
    let (status, headers, reply) = send(
        gw.addr,
        Method::POST,
        chat,
        "application/json",
        r#"{"model":"osprey-alt"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(!headers.contains_key(header::RETRY_AFTER));
    assert_eq!(reply["error"]["code"], "model_not_in_scene");
    assert_eq!(reply["error"]["model_id"], "brain-alt");
    assert_eq!(reply["error"]["scene"], "dual");
    assert!(reply["error"]["message"]
        .as_str()
        .unwrap()
        .contains("active Scene `dual`"));

    // With no default upstream there is nowhere to send these.
    let (status, _, reply) = send(gw.addr, Method::POST, chat, "application/json", "{oops").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(reply["error"]["code"], "invalid_json");

    let (status, _, reply) = send(
        gw.addr,
        Method::POST,
        chat,
        "application/json",
        r#"{"messages":[]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(reply["error"]["code"], "missing_model");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requests_without_a_model_go_to_the_default_upstream() {
    let a = echo_upstream("A").await;
    let fallback = echo_upstream("default").await;
    let reg = registry(&[("brain", a.port(), &[], "")]);
    let routes = Routes::build(&reg, &["brain".into()], |_| true);
    let gw = start_gateway(routes, Some(fallback.to_string()), MIB).await;

    let (status, _, reply) = send(gw.addr, Method::GET, "/metrics", "", "").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["upstream"], "default");
    let (_, headers, reply) =
        send(gw.addr, Method::POST, "/tokenize", "application/json", "{}").await;
    assert_eq!(reply["upstream"], "default");
    assert!(!headers.contains_key(MODEL_HEADER));

    // Inference without `model` behaves like a lone vLLM: its default model.
    let (status, _, reply) = send(
        gw.addr,
        Method::POST,
        "/v1/chat/completions",
        "application/json",
        r#"{"messages":[]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["upstream"], "default");
    let (status, _, reply) = send(
        gw.addr,
        Method::POST,
        "/v1/chat/completions",
        "application/json",
        "{oops",
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "unparseable JSON is the upstream's to reject"
    );
    assert_eq!(reply["body"], "{oops");

    // When the default upstream is a running model's address (e.g. the brain
    // behind a proxy on :8041), model-less requests are attributed to that model.
    let gw = start_gateway(
        Routes::build(&reg, &["brain".into()], |_| true),
        Some(a.to_string()),
        MIB,
    )
    .await;
    let (_, headers, reply) = send(gw.addr, Method::GET, "/metrics", "", "").await;
    assert_eq!(reply["upstream"], "A");
    assert_eq!(headers[MODEL_HEADER], "brain");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v1_models_lists_every_running_name_and_health_answers() {
    let reg = registry(&[
        ("brain", 8001, &["brain", "osprey"], ""),
        ("brain-alt", 8002, &["osprey-alt"], ""),
        ("embed", 8012, &["bge"], ""),
    ]);
    let routes = Routes::build(
        &reg,
        &["brain".into(), "brain-alt".into(), "embed".into()],
        |m| m.serve.port != Some(8012),
    );
    let gw = start_gateway(routes, None, MIB).await;

    let (status, headers, reply) = send(gw.addr, Method::GET, "/v1/models", "", "").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "application/json");
    assert_eq!(reply["object"], "list");
    let rows: Vec<(String, String, String)> = reply["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            assert_eq!(m["object"], "model");
            assert!(m["created"].as_u64().unwrap() > 0);
            (
                m["id"].as_str().unwrap().into(),
                m["root"].as_str().unwrap().into(),
                m["owned_by"].as_str().unwrap().into(),
            )
        })
        .collect();
    let row = |id: &str, root: &str| {
        (
            id.to_string(),
            root.to_string(),
            "lumbridge-computer".to_string(),
        )
    };
    assert_eq!(
        rows,
        vec![
            row("brain", "brain"),
            row("brain-alt", "brain-alt"),
            row("osprey", "brain"),
            row("osprey-alt", "brain-alt"),
        ]
    );

    let (status, _, reply) = send(gw.addr, Method::GET, "/health", "", "").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["status"], "ok");
}

/// An SSE upstream whose chunks the test releases one at a time. Returns the
/// upstream address and a receiver yielding each request's chunk sender.
async fn sse_upstream() -> (SocketAddr, mpsc::UnboundedReceiver<mpsc::Sender<Bytes>>) {
    let (streams_tx, streams_rx) = mpsc::unbounded_channel();
    let addr = fake_upstream(move |_request| {
        let streams_tx = streams_tx.clone();
        async move {
            let (tx, rx) = mpsc::channel(1);
            streams_tx.send(tx).unwrap();
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(ChannelBody(rx).boxed_unsync())
                .unwrap()
        }
    })
    .await;
    (addr, streams_rx)
}

async fn start_stream(gw: &Gateway) -> (Incoming, HeaderMap, tokio::task::JoinHandle<()>) {
    let (mut sender, task) = connect::<Full<Bytes>>(gw.addr).await;
    let request = Request::post("/v1/chat/completions")
        .header("host", "lumbridge-computer")
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from_static(
            br#"{"model":"brain","stream":true}"#,
        )))
        .unwrap();
    let response = sender.send_request(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let (parts, body) = response.into_parts();
    (body, parts.headers, task)
}

async fn next_chunk(body: &mut Incoming) -> Option<Bytes> {
    tokio::time::timeout(Duration::from_secs(5), body.frame())
        .await
        .expect("chunk did not arrive")
        .map(|frame| frame.unwrap().into_data().unwrap())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sse_chunks_arrive_before_the_upstream_finishes_and_in_flight_returns_to_zero() {
    let (upstream, mut streams) = sse_upstream().await;
    let reg = registry(&[("brain", upstream.port(), &[], "")]);
    let gw = start_gateway(Routes::build(&reg, &["brain".into()], |_| true), None, MIB).await;

    let (mut body, headers, _task) = start_stream(&gw).await;
    assert_eq!(headers["x-accel-buffering"], "no");
    assert_eq!(headers[MODEL_HEADER], "brain");
    let chunks = streams.recv().await.unwrap();
    assert_eq!(gw.in_flight.get("brain"), 1);

    chunks
        .send(Bytes::from_static(b"data: one\n\n"))
        .await
        .unwrap();
    // The upstream has not finished (it still holds `chunks`), yet the client
    // already has the first event.
    assert_eq!(next_chunk(&mut body).await.unwrap(), "data: one\n\n");
    assert_eq!(gw.in_flight.get("brain"), 1, "still streaming");

    chunks
        .send(Bytes::from_static(b"data: [DONE]\n\n"))
        .await
        .unwrap();
    assert_eq!(next_chunk(&mut body).await.unwrap(), "data: [DONE]\n\n");
    drop(chunks);
    assert!(next_chunk(&mut body).await.is_none());
    let in_flight = gw.in_flight.clone();
    eventually("in-flight to drop after completion", || {
        in_flight.get("brain") == 0
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_abort_cancels_the_upstream_stream_and_releases_in_flight() {
    let (upstream, mut streams) = sse_upstream().await;
    let reg = registry(&[("brain", upstream.port(), &[], "")]);
    let gw = start_gateway(Routes::build(&reg, &["brain".into()], |_| true), None, MIB).await;

    let (mut body, _, task) = start_stream(&gw).await;
    let chunks = streams.recv().await.unwrap();
    chunks
        .send(Bytes::from_static(b"data: one\n\n"))
        .await
        .unwrap();
    next_chunk(&mut body).await.unwrap();
    assert_eq!(gw.in_flight.get("brain"), 1);

    // The client vanishes mid-stream.
    task.abort();
    drop(body);
    tokio::time::timeout(Duration::from_secs(5), chunks.closed())
        .await
        .expect("upstream request was not cancelled");
    let in_flight = gw.in_flight.clone();
    eventually("in-flight to drop after abort", || {
        in_flight.get("brain") == 0
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bodies_over_the_cap_are_refused_with_413() {
    let a = echo_upstream("A").await;
    let reg = registry(&[("brain", a.port(), &[], "")]);
    let gw = start_gateway(Routes::build(&reg, &["brain".into()], |_| true), None, 1024).await;

    let big = format!(r#"{{"model":"brain","pad":"{}"}}"#, "x".repeat(2048));
    let (status, _, reply) = send(
        gw.addr,
        Method::POST,
        "/v1/chat/completions",
        "application/json",
        big,
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(reply["error"]["code"], "body_too_large");

    // Chunked (no Content-Length): the cap is enforced while reading.
    let (mut sender, _task) = connect::<ChannelBody>(gw.addr).await;
    let (tx, rx) = mpsc::channel(4);
    let request = Request::post("/v1/chat/completions")
        .header("host", "lumbridge-computer")
        .header("content-type", "application/json")
        .body(ChannelBody(rx))
        .unwrap();
    tokio::spawn(async move {
        tx.send(Bytes::from_static(br#"{"model":"brain","pad":""#))
            .await
            .ok();
        tx.send(Bytes::from(vec![b'x'; 2048])).await.ok();
        tx.send(Bytes::from_static(br#""}"#)).await.ok();
    });
    let response = sender.send_request(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

    let (status, _, _) = send(
        gw.addr,
        Method::POST,
        "/v1/chat/completions",
        "application/json",
        r#"{"model":"brain"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "bodies under the cap still pass");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unreachable_upstream_is_a_502_not_a_hang() {
    // Bind then drop, so nothing listens on the port.
    let dead = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let reg = registry(&[("brain", dead.port(), &[], "")]);
    let gw = start_gateway(Routes::build(&reg, &["brain".into()], |_| true), None, MIB).await;
    let (status, _, reply) = send(
        gw.addr,
        Method::POST,
        "/v1/chat/completions",
        "application/json",
        r#"{"model":"brain"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(reply["error"]["code"], "upstream_unavailable");
    assert_eq!(gw.in_flight.get("brain"), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_abort_during_a_silent_prefill_cancels_the_upstream_request() {
    // The upstream accepts the request and says nothing (a long prefill). Its
    // handler holds `alive`; the handler being dropped closes the channel.
    let (alive_tx, mut alive_rx) = mpsc::unbounded_channel::<mpsc::Sender<()>>();
    let upstream = fake_upstream(move |_request| {
        let alive_tx = alive_tx.clone();
        async move {
            let (tx, _rx) = mpsc::channel::<()>(1);
            alive_tx.send(tx.clone()).unwrap();
            std::future::pending::<()>().await;
            drop(tx);
            unreachable!()
        }
    })
    .await;
    let reg = registry(&[("brain", upstream.port(), &[], "")]);
    let gw = start_gateway(Routes::build(&reg, &["brain".into()], |_| true), None, MIB).await;

    let (mut sender, task) = connect::<Full<Bytes>>(gw.addr).await;
    let request = Request::post("/v1/chat/completions")
        .header("host", "lumbridge-computer")
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from_static(br#"{"model":"brain"}"#)))
        .unwrap();
    let pending = tokio::spawn(async move { sender.send_request(request).await });
    let alive = alive_rx.recv().await.unwrap();
    assert_eq!(gw.in_flight.get("brain"), 1);

    task.abort();
    pending.abort();
    tokio::time::timeout(Duration::from_secs(5), alive.closed())
        .await
        .expect("upstream request outlived the client");
    let in_flight = gw.in_flight.clone();
    eventually("in-flight to drop", || in_flight.get("brain") == 0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inflight_endpoint_reports_a_live_stream_to_loopback_callers() {
    let (upstream, mut streams) = sse_upstream().await;
    let reg = registry(&[("brain", upstream.port(), &[], "")]);
    let gw = start_gateway(Routes::build(&reg, &["brain".into()], |_| true), None, MIB).await;

    let (status, _, reply) = send(gw.addr, Method::GET, super::super::INFLIGHT_PATH, "", "").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        reply,
        serde_json::json!({ "in_flight": {}, "last_finished_ms": {} })
    );

    let (mut body, _headers, _task) = start_stream(&gw).await;
    let chunks = streams.recv().await.unwrap();
    let (status, _, reply) = send(gw.addr, Method::GET, super::super::INFLIGHT_PATH, "", "").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["in_flight"]["brain"], 1);

    drop(chunks);
    while next_chunk(&mut body).await.is_some() {}
}

#[test]
fn inflight_endpoint_refuses_callers_from_other_machines() {
    let in_flight = InFlight::default();
    let remote: SocketAddr = "192.0.2.7:50000".parse().unwrap();
    assert_eq!(
        inflight_response(&in_flight, remote).status(),
        StatusCode::FORBIDDEN
    );
    for local in ["127.0.0.1:1", "[::1]:1", "[::ffff:127.0.0.1]:1"] {
        let peer: SocketAddr = local.parse().unwrap();
        assert_eq!(inflight_response(&in_flight, peer).status(), StatusCode::OK);
    }
}

#[path = "wake_tests.rs"]
mod wake;
