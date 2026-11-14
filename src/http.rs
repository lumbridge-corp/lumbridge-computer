//! A read-only HTTP control API.
//!
//! This exists so a web UI can read a node's state without shelling out to the CLI. It is a
//! second transport over the operations the MCP server already models — `collect_status`,
//! `collect_models`, `collect_scenes`, `collect_scene`, `collect_evals` — deliberately not a
//! second implementation of them, so the two surfaces cannot drift.
//!
//! Hand-rolled on `std::net`, matching `gateway.rs`. An HTTP framework would pull a dependency
//! tree an order of magnitude larger than the whole rest of this binary, to serve a small set
//! of routes that return pre-serialised JSON.
//!
//! # What this deliberately does not do
//!
//! **It never mutates.** No activate, no stop, no eval run. A read-only surface that a browser
//! can reach is a much smaller thing to get right than one that can move a node's memory around,
//! and the read half is what a dashboard actually needs.
//!
//! # Why the defaults are what they are
//!
//! - **Loopback only.** The listen address defaults to `127.0.0.1`. There is no authentication
//!   worth the name here, so a bind to `0.0.0.0` publishes your node's inventory to the network.
//! - **CORS off.** No origin is allowed unless named with `--allow-origin`. Allowing `*` would
//!   let *any* page you visit read what models you run, which is a fingerprint of your machine.
//! - **The token is optional but checked in constant time.** Loopback plus an origin allowlist
//!   already stops the browser attack; the token is for the case where someone puts this behind
//!   a proxy anyway.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use crate::mcp::{collect_evals, collect_models, collect_scene, collect_scenes, collect_status};

/// Requests are tiny and come from localhost; anything slower than this is not a browser.
const READ_TIMEOUT: Duration = Duration::from_secs(5);
/// A request line plus headers. Anything larger is not a request we serve.
const MAX_HEAD_BYTES: usize = 8 * 1024;

pub struct Config {
    pub root: PathBuf,
    /// Origins permitted to read this API from a browser. Empty means none.
    pub allowed_origins: Vec<String>,
    /// When set, every request must carry `Authorization: Bearer <token>`.
    pub token: Option<String>,
}

pub fn run(listen: &str, config: Config) -> Result<()> {
    let listener = TcpListener::bind(listen)
        .with_context(|| format!("binding the Lumbridge Computer API at {listen}"))?;
    println!(
        "Lumbridge Computer API on {listen} (read-only) · origins: {} · token: {}",
        if config.allowed_origins.is_empty() {
            "none".to_string()
        } else {
            config.allowed_origins.join(", ")
        },
        if config.token.is_some() {
            "required"
        } else {
            "none"
        },
    );
    if !listen.starts_with("127.") && !listen.starts_with("localhost") {
        eprintln!(
            "warning: {listen} is not loopback. This API has no authentication by default and \
             reveals which models this node runs."
        );
    }
    serve(listener, config)
}

fn serve(listener: TcpListener, config: Config) -> Result<()> {
    let config = std::sync::Arc::new(config);
    for incoming in listener.incoming() {
        let stream = match incoming {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("api accept failed: {error}");
                continue;
            }
        };
        let config = config.clone();
        thread::spawn(move || {
            if let Err(error) = handle(stream, &config) {
                eprintln!("api request failed: {error:#}");
            }
        });
    }
    Ok(())
}

struct Request {
    method: String,
    path: String,
    query: BTreeMap<String, String>,
    origin: Option<String>,
    authorization: Option<String>,
}

/// Read the request line and headers. The body is ignored: every route is a GET.
fn read_request(stream: &TcpStream) -> Result<Option<Request>> {
    let mut reader = BufReader::new(stream);
    let mut head = String::new();
    let mut total = 0usize;

    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Ok(None); // client hung up
        }
        total += n;
        if total > MAX_HEAD_BYTES {
            return Ok(None);
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        head.push_str(&line);
    }

    let mut lines = head.lines();
    let Some(request_line) = lines.next() else {
        return Ok(None);
    };
    let mut parts = request_line.split_whitespace();
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        return Ok(None);
    };

    let mut origin = None;
    let mut authorization = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().to_string();
        match name.trim().to_ascii_lowercase().as_str() {
            "origin" => origin = Some(value),
            "authorization" => authorization = Some(value),
            _ => {}
        }
    }

    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let query = query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| pair.split_once('=').unwrap_or((pair, "")))
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();

    Ok(Some(Request {
        method: method.to_string(),
        path: path.to_string(),
        query,
        origin,
        authorization,
    }))
}

/// Constant-time comparison so a token cannot be recovered a byte at a time from response timing.
fn token_ok(expected: &str, supplied: Option<&String>) -> bool {
    let Some(supplied) = supplied.and_then(|v| v.strip_prefix("Bearer ")) else {
        return false;
    };
    let a = expected.as_bytes();
    let b = supplied.as_bytes();
    // Length is compared without branching on it beyond the final AND.
    let mut diff = (a.len() ^ b.len()) as u8;
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    diff == 0
}

fn handle(mut stream: TcpStream, config: &Config) -> Result<()> {
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    stream.set_write_timeout(Some(READ_TIMEOUT))?;

    let Some(request) = read_request(&stream)? else {
        return Ok(());
    };

    // Echo the origin only when it is on the allowlist. Never `*`: this API describes the
    // machine it runs on, so any-origin access means any page can fingerprint the node.
    let allow_origin = request
        .origin
        .as_ref()
        .filter(|o| config.allowed_origins.iter().any(|a| a == *o))
        .cloned();

    if request.method == "OPTIONS" {
        return write_response(&mut stream, 204, "", allow_origin.as_deref(), true);
    }

    if let Some(expected) = &config.token {
        if !token_ok(expected, request.authorization.as_ref()) {
            return write_json(
                &mut stream,
                401,
                r#"{"error":"unauthorized"}"#,
                allow_origin.as_deref(),
            );
        }
    }

    if request.method != "GET" {
        return write_json(
            &mut stream,
            405,
            r#"{"error":"this API is read-only"}"#,
            allow_origin.as_deref(),
        );
    }

    let (status, body) = route(&request.path, &request.query, &config.root);
    write_json(&mut stream, status, &body, allow_origin.as_deref())
}

fn route(path: &str, query: &BTreeMap<String, String>, root: &Path) -> (u16, String) {
    let since = query.get("since").map(String::as_str).unwrap_or("24h");
    let rendered = match path {
        "/v1/health" => Ok(r#"{"ok":true,"service":"lumbridge-computer","api":"v1"}"#.to_string()),
        "/v1/status" => collect_status(root).and_then(|r| Ok(serde_json::to_string(&r)?)),
        "/v1/models" => collect_models(root).and_then(|r| Ok(serde_json::to_string(&r)?)),
        "/v1/scenes" => collect_scenes(root).and_then(|r| Ok(serde_json::to_string(&r)?)),
        "/v1/evals" => collect_evals(root).and_then(|r| Ok(serde_json::to_string(&r)?)),
        "/v1/usage" => crate::telemetry::report(root, since)
            .and_then(|report| Ok(serde_json::to_string(&report)?)),
        "/v1/usage/agents" => crate::telemetry::report(root, since)
            .and_then(|report| Ok(serde_json::to_string(&report.agents)?)),
        "/v1/usage/concurrency" => crate::telemetry::report(root, since)
            .and_then(|report| Ok(serde_json::to_string(&report.concurrency)?)),
        other => match other.strip_prefix("/v1/scenes/") {
            // Exactly one segment: /v1/scenes/a/b is not a route.
            Some(name) if !name.is_empty() && !name.contains('/') => {
                collect_scene(root, name).and_then(|r| Ok(serde_json::to_string(&r)?))
            }
            _ => return (404, r#"{"error":"no such route"}"#.to_string()),
        },
    };

    match rendered {
        Ok(body) => (200, body),
        // A collector fails when the thing does not exist (an unknown Scene) or when the node's
        // own config is unreadable. The message is the operator's, and this is a loopback API,
        // so passing it through is more useful than flattening it to "error".
        Err(error) => (
            404,
            serde_json::json!({ "error": format!("{error:#}") }).to_string(),
        ),
    }
}

fn write_json(stream: &mut TcpStream, status: u16, body: &str, origin: Option<&str>) -> Result<()> {
    write_response(stream, status, body, origin, false)
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    body: &str,
    origin: Option<&str>,
    preflight: bool,
) -> Result<()> {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Error",
    };
    let mut head = format!("HTTP/1.1 {status} {reason}\r\n");
    head.push_str("Content-Type: application/json\r\n");
    head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    // This is live node state; a cached answer is a wrong answer.
    head.push_str("Cache-Control: no-store\r\n");
    head.push_str("Connection: close\r\n");
    if let Some(origin) = origin {
        head.push_str(&format!("Access-Control-Allow-Origin: {origin}\r\n"));
        // Tell caches the body varies by origin, so an allowed origin's response can never be
        // replayed to a disallowed one.
        head.push_str("Vary: Origin\r\n");
        if preflight {
            head.push_str("Access-Control-Allow-Methods: GET, OPTIONS\r\n");
            head.push_str("Access-Control-Allow-Headers: Authorization\r\n");
            head.push_str("Access-Control-Max-Age: 600\r\n");
        }
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;
    Ok(())
}

/// Drain and discard — kept for symmetry with future routes that accept a body.
#[allow(dead_code)]
fn discard_body(reader: &mut impl Read) {
    let mut sink = Vec::new();
    let _ = reader.read_to_end(&mut sink);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_comparison_rejects_wrong_and_missing_and_prefixless() {
        let t = "s3cret-token";
        assert!(token_ok(t, Some(&format!("Bearer {t}"))));
        assert!(!token_ok(t, None));
        assert!(!token_ok(t, Some(&t.to_string()))); // no "Bearer " prefix
        assert!(!token_ok(t, Some(&"Bearer wrong".to_string())));
        // A correct prefix must not pass — this is the bug constant-time comparison exists for.
        assert!(!token_ok(t, Some(&"Bearer s3cret".to_string())));
        assert!(!token_ok(t, Some(&"Bearer s3cret-token-plus".to_string())));
    }

    #[test]
    fn unknown_routes_404_and_scene_paths_take_exactly_one_segment() {
        let root = Path::new("/nonexistent-root-for-routing-test");
        let query = BTreeMap::new();
        assert_eq!(route("/v1/nope", &query, root).0, 404);
        assert_eq!(route("/v1/scenes/a/b", &query, root).0, 404);
        assert_eq!(route("/v1/scenes/", &query, root).0, 404);
        // Health needs no filesystem, so it answers even on a bogus root.
        assert_eq!(route("/v1/health", &query, root).0, 200);
    }

    #[test]
    fn health_body_is_valid_json() {
        let (status, body) = route("/v1/health", &BTreeMap::new(), Path::new("/tmp"));
        assert_eq!(status, 200);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["ok"], true);
        assert_eq!(
            body,
            r#"{"ok":true,"service":"lumbridge-computer","api":"v1"}"#
        );
    }
}
