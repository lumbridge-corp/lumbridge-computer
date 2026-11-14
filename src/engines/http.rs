//! Dependency-free HTTP GET for local Prometheus endpoints.
//!
//! Moved here from `telemetry` so the scraper, telemetry and liveness share
//! one client. We avoid an HTTP crate on purpose: every endpoint is a local
//! engine, and a blocking `TcpStream` with explicit timeouts is all we need.

use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

const MAX_METRICS_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// GET `url` with a connect timeout and a per-read timeout. Callers on the
/// agent's one-second supervision loop pass short ones.
pub(crate) fn fetch_http_within(url: &str, connect: Duration, read: Duration) -> Result<String> {
    let (host, port, path) = parse_http_url(url)?;
    let address = (host.as_str(), port)
        .to_socket_addrs()?
        .next()
        .with_context(|| format!("resolving {host}:{port}"))?;
    let mut stream = TcpStream::connect_timeout(&address, connect)
        .with_context(|| format!("connecting to {url}"))?;
    stream.set_read_timeout(Some(read))?;
    stream.set_write_timeout(Some(connect))?;
    // One write: `write!` on an unbuffered stream sends each formatted piece
    // as its own segment, and a server that answers after the first piece
    // and closes with the rest unread resets the connection.
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nAccept: text/plain\r\nAccept-Encoding: identity\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes())?;
    let mut response = Vec::new();
    stream
        .take((MAX_METRICS_RESPONSE_BYTES + 1) as u64)
        .read_to_end(&mut response)?;
    if response.len() > MAX_METRICS_RESPONSE_BYTES {
        bail!("metrics response exceeded {MAX_METRICS_RESPONSE_BYTES} bytes");
    }
    let head_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .context("metrics endpoint returned no HTTP header terminator")?;
    let head = String::from_utf8_lossy(&response[..head_end]);
    let status = head.lines().next().unwrap_or_default();
    if status.split_whitespace().nth(1) != Some("200") {
        bail!("metrics endpoint returned {status}");
    }
    let body = &response[head_end + 4..];
    let body = if head
        .lines()
        .any(|line| line.eq_ignore_ascii_case("transfer-encoding: chunked"))
    {
        decode_chunked(body)?
    } else {
        body.to_vec()
    };
    String::from_utf8(body).context("metrics response was not UTF-8")
}

/// Small control responses (sleep, wake, `is_sleeping`) never need more.
const MAX_CONTROL_RESPONSE_BYTES: usize = 1024 * 1024;

/// Send one request to a loopback engine and return its status and body,
/// whatever the status. Only `127.0.0.1`/`localhost` are accepted: engine
/// control endpoints (vLLM's dev-mode `/sleep`) must never be reached remotely.
pub(crate) fn request_within(
    method: &str,
    url: &str,
    body: Option<&str>,
    connect: Duration,
    read: Duration,
) -> Result<(u16, String)> {
    let (host, port, path) = parse_http_url(url)?;
    if host != "127.0.0.1" && host != "localhost" {
        bail!("engine control requests are loopback only, got host '{host}'");
    }
    let address = (host.as_str(), port)
        .to_socket_addrs()?
        .next()
        .with_context(|| format!("resolving {host}:{port}"))?;
    let mut stream = TcpStream::connect_timeout(&address, connect)
        .with_context(|| format!("connecting to {url}"))?;
    stream.set_read_timeout(Some(read))?;
    stream.set_write_timeout(Some(connect.max(Duration::from_secs(1))))?;
    let body = body.unwrap_or("");
    let content = if body.is_empty() {
        String::new()
    } else {
        "Content-Type: application/json\r\n".to_string()
    };
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nAccept: application/json\r\nAccept-Encoding: identity\r\n{content}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes())?;
    let mut response = Vec::new();
    stream
        .take((MAX_CONTROL_RESPONSE_BYTES + 1) as u64)
        .read_to_end(&mut response)
        .with_context(|| format!("reading the answer to {method} {url}"))?;
    if response.len() > MAX_CONTROL_RESPONSE_BYTES {
        bail!("response to {method} {url} exceeded {MAX_CONTROL_RESPONSE_BYTES} bytes");
    }
    let head_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .with_context(|| format!("{method} {url} returned no HTTP header terminator"))?;
    let head = String::from_utf8_lossy(&response[..head_end]);
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .with_context(|| format!("{method} {url} returned no HTTP status"))?;
    let body = &response[head_end + 4..];
    let body = if head
        .lines()
        .any(|line| line.eq_ignore_ascii_case("transfer-encoding: chunked"))
    {
        decode_chunked(body)?
    } else {
        body.to_vec()
    };
    Ok((status, String::from_utf8_lossy(&body).into_owned()))
}

fn decode_chunked(mut body: &[u8]) -> Result<Vec<u8>> {
    let mut decoded = Vec::new();
    loop {
        let line_end = body
            .windows(2)
            .position(|window| window == b"\r\n")
            .context("invalid chunked metrics response")?;
        let size_text = std::str::from_utf8(&body[..line_end])?
            .split(';')
            .next()
            .unwrap_or_default();
        let size = usize::from_str_radix(size_text.trim(), 16)?;
        body = &body[line_end + 2..];
        if size == 0 {
            break;
        }
        if body.len() < size + 2 {
            bail!("truncated chunked metrics response");
        }
        decoded.extend_from_slice(&body[..size]);
        body = &body[size + 2..];
    }
    Ok(decoded)
}

pub(crate) fn parse_http_url(url: &str) -> Result<(String, u16, String)> {
    let rest = url
        .strip_prefix("http://")
        .with_context(|| format!("telemetry metrics URL must use local http://, got '{url}'"))?;
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = authority
        .rsplit_once(':')
        .with_context(|| format!("metrics URL must include a port: '{url}'"))?;
    let port = port.parse::<u16>().context("parsing metrics URL port")?;
    if host.is_empty() {
        bail!("metrics URL has an empty host");
    }
    Ok((host.to_string(), port, format!("/{path}")))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    /// Read through the blank line, so closing never discards unread
    /// request bytes (which would turn the close into a reset).
    pub(crate) fn read_request_head(stream: &mut TcpStream) -> String {
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            match stream.read(&mut byte) {
                Ok(1) => head.push(byte[0]),
                _ => break,
            }
        }
        String::from_utf8_lossy(&head).into_owned()
    }

    fn serve_once(response: &'static [u8]) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            read_request_head(&mut stream);
            stream.write_all(response).unwrap();
        });
        port
    }

    #[test]
    fn chunked_bodies_are_reassembled() {
        let port = serve_once(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nx 1\n\r\n0\r\n\r\n",
        );
        let body = fetch_http_within(
            &format!("http://127.0.0.1:{port}/metrics"),
            Duration::from_secs(1),
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(body, "x 1\n");
    }

    #[test]
    fn a_non_200_status_is_an_error() {
        let port = serve_once(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
        let error = fetch_http_within(
            &format!("http://127.0.0.1:{port}/metrics"),
            Duration::from_secs(1),
            Duration::from_secs(1),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("404"));
    }
}
