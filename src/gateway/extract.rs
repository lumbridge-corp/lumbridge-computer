//! Reading (and optionally rewriting) the `model` a request names, without
//! touching anything else in it.
//!
//! The gateway forwards the client's original bytes, so these helpers only ever
//! look: JSON is scanned with a two-field struct (serde skips every other value
//! without building a tree, which matters for multi-MB vision/audio bodies),
//! multipart is scanned for one form field, and query strings are decoded for
//! one key. Rewriting is opt-in per model and touches only the `model` value.

use serde::de::IgnoredAny;
use serde::Deserialize;
use serde_json::value::RawValue;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ops::Range;

/// What routing needs from a request.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Peek {
    pub model: Option<String>,
    pub stream: bool,
    /// The body claimed to be JSON but isn't a JSON object.
    pub invalid_json: bool,
}

#[derive(Deserialize)]
struct JsonPeek<'a> {
    #[serde(borrow, default)]
    model: Option<Cow<'a, str>>,
    #[serde(default)]
    stream: Option<Flag>,
}

/// `stream` is read only for logging, so an odd value must not make the whole
/// body unparseable (which would lose `model` too).
#[derive(Deserialize)]
#[serde(untagged)]
enum Flag {
    Bool(bool),
    Other(IgnoredAny),
}

pub fn json(body: &[u8]) -> Peek {
    // Derived structs also accept JSON arrays (fields by position); a request
    // body that isn't an object has no `model`.
    let invalid = Peek {
        invalid_json: true,
        ..Peek::default()
    };
    if body.iter().find(|b| !b.is_ascii_whitespace()) != Some(&b'{') {
        return invalid;
    }
    match serde_json::from_slice::<JsonPeek>(body) {
        Ok(peek) => Peek {
            model: peek.model.map(Cow::into_owned).filter(|m| !m.is_empty()),
            stream: matches!(peek.stream, Some(Flag::Bool(true))),
            invalid_json: false,
        },
        Err(_) => invalid,
    }
}

/// Replace the top-level `model` in a JSON object. Every other value is copied
/// verbatim as raw JSON; only the top-level key order may change.
pub fn rewrite_json(body: &[u8], model: &str) -> Option<Vec<u8>> {
    let mut object: BTreeMap<String, Box<RawValue>> = serde_json::from_slice(body).ok()?;
    let value = RawValue::from_string(serde_json::to_string(model).ok()?).ok()?;
    object.insert("model".to_string(), value);
    serde_json::to_vec(&object).ok()
}

/// The `boundary` parameter of a `multipart/form-data` content type.
pub fn multipart_boundary(content_type: &str) -> Option<&str> {
    let mut params = content_type.split(';');
    let mime = params.next()?.trim();
    if !mime.eq_ignore_ascii_case("multipart/form-data") {
        return None;
    }
    params.find_map(|param| {
        let (key, value) = param.trim().split_once('=')?;
        key.trim()
            .eq_ignore_ascii_case("boundary")
            .then(|| value.trim().trim_matches('"'))
            .filter(|b| !b.is_empty())
    })
}

/// Byte range of the value of form field `name` in a multipart body.
///
/// A minimal RFC 7578 scan: parts are delimited by `--boundary`, each has
/// headers up to a blank line, and the value ends at `\r\n--boundary`. Parts
/// with a `filename` are skipped without being inspected beyond their headers.
pub fn multipart_field(body: &[u8], boundary: &str, name: &str) -> Option<Range<usize>> {
    let delimiter = format!("--{boundary}");
    let close = format!("\r\n--{boundary}");
    let mut at = find(body, delimiter.as_bytes(), 0)? + delimiter.len();
    loop {
        if body.get(at..at + 2) == Some(b"--") {
            return None; // closing delimiter
        }
        let headers_start = find(body, b"\r\n", at)? + 2;
        let headers_end = find(body, b"\r\n\r\n", headers_start.saturating_sub(2))?;
        let value_start = headers_end + 4;
        let value_end = find(body, close.as_bytes(), value_start)?;
        let headers = std::str::from_utf8(&body[headers_start.min(headers_end)..headers_end]).ok();
        if headers.is_some_and(|h| field_name(h) == Some(name)) {
            return Some(value_start..value_end);
        }
        at = value_end + close.len();
    }
}

/// `name` from a part's `Content-Disposition: form-data; name="..."`.
fn field_name(headers: &str) -> Option<&str> {
    headers.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if !key.trim().eq_ignore_ascii_case("content-disposition") {
            return None;
        }
        value.split(';').skip(1).find_map(|param| {
            let (k, v) = param.trim().split_once('=')?;
            (k.trim() == "name").then(|| v.trim().trim_matches('"'))
        })
    })
}

pub fn multipart(body: &[u8], boundary: &str) -> Peek {
    let field = |name| {
        multipart_field(body, boundary, name)
            .and_then(|range| std::str::from_utf8(&body[range]).ok())
            .map(|value| value.trim().to_string())
    };
    Peek {
        model: field("model").filter(|m| !m.is_empty()),
        stream: field("stream").is_some_and(|s| s == "true"),
        invalid_json: false,
    }
}

/// The decoded value of `key` in a URL query string.
pub fn query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        (percent_decode(k) == key).then(|| percent_decode(v))
    })
}

/// `query` with every `key` value replaced by `value`.
pub fn rewrite_query(query: &str, key: &str, value: &str) -> String {
    query
        .split('&')
        .map(|pair| {
            let k = pair.split_once('=').map_or(pair, |(k, _)| k);
            if percent_decode(k) == key {
                format!("{k}={}", percent_encode(value))
            } else {
                pair.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("&")
}

/// vLLM's dev-mode routes (`VLLM_SERVER_DEV_MODE=1`, which sleep needs; vLLM
/// 0.27.1 `entrypoints/serve/dev/`): `/collective_rpc` runs any worker method,
/// the rest pause, sleep, reset or replace weights. Never reachable through
/// the gateway, whatever the model: a sleepable engine binds 127.0.0.1 so that
/// the gateway is its only door.
pub const ENGINE_CONTROL_PATHS: &[&str] = &[
    "/collective_rpc",
    "/sleep",
    "/wake_up",
    "/is_sleeping",
    "/pause",
    "/resume",
    "/is_paused",
    "/abort_requests",
    "/reset_prefix_cache",
    "/reset_mm_cache",
    "/reset_encoder_cache",
    "/server_info",
    "/get_world_size",
    "/weight_info",
    "/init_weight_transfer_engine",
    "/start_weight_update",
    "/start_draft_weight_update",
    "/update_weights",
    "/finish_weight_update",
    "/update_weight_version",
];

/// True when `path` names an engine-control route, as the engine would see it:
/// percent-decoded (uvicorn decodes once), extra slashes and case ignored.
pub fn is_engine_control(path: &str) -> bool {
    let decoded = percent_decode(path);
    let trimmed = decoded.trim_matches('/');
    ENGINE_CONTROL_PATHS
        .iter()
        .any(|control| control[1..].eq_ignore_ascii_case(trimmed))
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' => match (hex(bytes.get(i + 1)), hex(bytes.get(i + 2))) {
                (Some(hi), Some(lo)) => {
                    out.push(hi << 4 | lo);
                    i += 3;
                    continue;
                }
                _ => out.push(b'%'),
            },
            byte => out.push(byte),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(byte: Option<&u8>) -> Option<u8> {
    match byte? {
        b @ b'0'..=b'9' => Some(b - b'0'),
        b @ b'a'..=b'f' => Some(b - b'a' + 10),
        b @ b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn percent_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    haystack
        .get(from..)?
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|offset| from + offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_control_routes_are_recognised_however_they_are_spelled() {
        for path in [
            "/collective_rpc",
            "/sleep",
            "/wake_up",
            "/sleep/",
            "//collective_rpc",
            "/%63ollective_rpc",
            "/Collective_RPC",
            "/update_weights",
            "/reset_prefix_cache",
        ] {
            assert!(is_engine_control(path), "{path}");
        }
        for path in [
            "/v1/chat/completions",
            "/health",
            "/metrics",
            "/v1/sleep",
            "/sleepy",
            "/",
        ] {
            assert!(!is_engine_control(path), "{path}");
        }
    }

    #[test]
    fn json_peek_reads_model_and_stream_and_ignores_everything_else() {
        let image = "A".repeat(2 << 20);
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{image}"}}}}]}}],"stream":true,"model":"brain\/alt","extra":{{"n":[1,2,3]}}}}"#
        );
        assert_eq!(
            json(body.as_bytes()),
            Peek {
                model: Some("brain/alt".into()),
                stream: true,
                invalid_json: false,
            }
        );
    }

    #[test]
    fn json_peek_survives_odd_stream_values_and_rejects_non_objects() {
        assert_eq!(
            json(br#"{"model":"embed","stream":"yes"}"#)
                .model
                .as_deref(),
            Some("embed")
        );
        assert_eq!(json(br#"{"model":""}"#).model, None);
        assert_eq!(json(br#"{"model":7}"#).model, None);
        assert_eq!(json(b"not json").model, None);
        assert!(json(b"not json").invalid_json);
        assert!(!json(br#"{"messages":[]}"#).invalid_json);
        assert_eq!(json(br#"["model"]"#).model, None);
    }

    #[test]
    fn json_rewrite_changes_only_the_model_value() {
        let body = br#"{"stream":true,"model":"brain","messages":[{"role":"user","content":"hi \u00e9"}]}"#;
        let out = rewrite_json(body, "osprey").unwrap();
        let before: serde_json::Value = serde_json::from_slice(body).unwrap();
        let mut after: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(after["model"], "osprey");
        after["model"] = "brain".into();
        assert_eq!(before, after);
        // Raw values are copied byte-for-byte, escapes included.
        assert!(String::from_utf8(out).unwrap().contains(r#""hi \u00e9""#));
    }

    fn form(boundary: &str, parts: &[(&str, Option<&str>, &[u8])]) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, filename, value) in parts {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            match filename {
                Some(f) => body.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{f}\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes(),
                ),
                None => body.extend_from_slice(
                    format!("content-disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
                ),
            }
            body.extend_from_slice(value);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    #[test]
    fn multipart_model_is_found_after_a_binary_file_part() {
        let boundary = "----lumbridge-computer7MA4YWxk";
        let audio: Vec<u8> = (0..=255u8).cycle().take(100_000).collect();
        let body = form(
            boundary,
            &[
                ("file", Some("a.wav"), &audio),
                ("model", None, b"whisper-large"),
                ("stream", None, b"true"),
            ],
        );
        assert_eq!(
            multipart(&body, boundary),
            Peek {
                model: Some("whisper-large".into()),
                stream: true,
                invalid_json: false,
            }
        );
        let ct = format!("multipart/form-data; boundary=\"{boundary}\"");
        assert_eq!(multipart_boundary(&ct), Some(boundary));
        assert_eq!(multipart_boundary("application/json"), None);
    }

    #[test]
    fn multipart_without_a_model_field_has_none() {
        let body = form("b", &[("file", Some("model"), b"model")]);
        assert_eq!(multipart(&body, "b").model, None);
        assert_eq!(multipart(b"garbage", "b").model, None);
    }

    #[test]
    fn query_model_is_percent_decoded_and_rewritable() {
        let query = "a=1&model=unsloth%2FQwen3.6+fast&b";
        assert_eq!(
            query_param(query, "model").as_deref(),
            Some("unsloth/Qwen3.6 fast")
        );
        assert_eq!(query_param(query, "missing"), None);
        assert_eq!(query_param("model=%zz", "model").as_deref(), Some("%zz"));
        assert_eq!(
            rewrite_query(query, "model", "brain/x"),
            "a=1&model=brain%2Fx&b"
        );
    }
}
