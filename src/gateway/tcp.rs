//! Streaming-safe TCP gateway for OpenAI-compatible model servers
//! (`--gateway-mode tcp`, the pre-M1 path kept unchanged for rollback).
//!
//! This mode deliberately stays below HTTP: it forwards bytes unchanged to one
//! upstream, so chunked responses and server-sent-event token streams retain
//! their timing and semantics while clients keep one stable Lumbridge Computer address.

use anyhow::{Context, Result};
use std::io;
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Instant;

/// Identifies both byte-pump halves of a proxied connection in logs.
static CONN_ID: AtomicU64 = AtomicU64::new(0);

pub(super) fn run(listen: &str, upstream: &str) -> Result<()> {
    let listener = TcpListener::bind(listen)
        .with_context(|| format!("binding Lumbridge Computer gateway at {listen}"))?;
    println!("Lumbridge Computer gateway {listen} -> {upstream}");
    serve_listener(listener, upstream, None)
}

fn serve_listener(
    listener: TcpListener,
    upstream: &str,
    max_connections: Option<usize>,
) -> Result<()> {
    let mut accepted = 0usize;
    for incoming in listener.incoming() {
        let client = match incoming {
            Ok(client) => client,
            Err(error) => {
                eprintln!("gateway accept failed: {error}");
                continue;
            }
        };
        let upstream = upstream.to_string();
        let id = CONN_ID.fetch_add(1, Ordering::Relaxed);
        thread::spawn(move || {
            if let Err(error) = proxy(id, client, &upstream) {
                eprintln!("gateway request failed: conn={id} {error:#}");
            }
        });
        accepted += 1;
        if max_connections.is_some_and(|limit| accepted >= limit) {
            break;
        }
    }
    Ok(())
}

/// Shuts both sockets down on every return path. Dropping one descriptor is not
/// sufficient because each byte-pump owns a cloned descriptor for the same
/// socket; a blocked clone otherwise keeps a failed OpenAI request alive forever.
struct ShutdownGuard {
    client: TcpStream,
    upstream: TcpStream,
}

impl Drop for ShutdownGuard {
    fn drop(&mut self) {
        self.client.shutdown(Shutdown::Both).ok();
        self.upstream.shutdown(Shutdown::Both).ok();
    }
}

fn proxy(id: u64, mut client: TcpStream, upstream_addr: &str) -> Result<()> {
    client.set_nodelay(true).ok();
    let mut upstream = TcpStream::connect(upstream_addr)
        .with_context(|| format!("connecting gateway upstream {upstream_addr}"))?;
    upstream.set_nodelay(true).ok();

    let _guard = ShutdownGuard {
        client: client.try_clone()?,
        upstream: upstream.try_clone()?,
    };

    let started = Instant::now();
    let mut client_reader = client.try_clone()?;
    let mut upstream_writer = upstream.try_clone()?;
    let request = thread::spawn(move || -> io::Result<u64> {
        match io::copy(&mut client_reader, &mut upstream_writer) {
            Ok(copied) => {
                upstream_writer.shutdown(Shutdown::Write).ok();
                Ok(copied)
            }
            Err(error) => {
                eprintln!("gateway request-copy failed: conn={id} {error}");
                Err(error)
            }
        }
    });

    let downstream = io::copy(&mut upstream, &mut client);
    client.shutdown(Shutdown::Write).ok();
    let elapsed = started.elapsed().as_millis();
    let down = match downstream {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("gateway response-copy failed: conn={id} after {elapsed}ms {error}");
            return Err(error.into());
        }
    };
    let up = request
        .join()
        .map_err(|_| anyhow::anyhow!("gateway request-copy thread panicked"))??;
    eprintln!("gateway conn={id} ok up={up}B down={down}B {elapsed}ms");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    #[test]
    fn gateway_forwards_bidirectional_bytes_without_buffering_protocols() {
        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let upstream_addr = upstream.local_addr().unwrap();
        let upstream_thread = thread::spawn(move || {
            let (mut socket, _) = upstream.accept().unwrap();
            let mut request = [0u8; 4];
            socket.read_exact(&mut request).unwrap();
            assert_eq!(&request, b"ping");
            socket.write_all(b"pong").unwrap();
        });

        let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
        let gateway_addr = gateway.local_addr().unwrap();
        let gateway_thread = thread::spawn(move || {
            serve_listener(gateway, &upstream_addr.to_string(), Some(1)).unwrap();
        });

        let mut client = TcpStream::connect(gateway_addr).unwrap();
        client.write_all(b"ping").unwrap();
        client.shutdown(Shutdown::Write).unwrap();
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        assert_eq!(response, b"pong");

        upstream_thread.join().unwrap();
        gateway_thread.join().unwrap();
    }

    #[test]
    fn upstream_reset_does_not_strand_the_client() {
        use std::os::fd::AsRawFd;
        use std::time::Duration;

        let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
        let upstream_addr = upstream.local_addr().unwrap();
        let upstream_thread = thread::spawn(move || {
            let (socket, _) = upstream.accept().unwrap();
            let linger = libc::linger {
                l_onoff: 1,
                l_linger: 0,
            };
            unsafe {
                libc::setsockopt(
                    socket.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_LINGER,
                    &linger as *const _ as *const libc::c_void,
                    std::mem::size_of::<libc::linger>() as libc::socklen_t,
                );
            }
            thread::sleep(Duration::from_millis(50));
            drop(socket);
        });

        let gateway = TcpListener::bind("127.0.0.1:0").unwrap();
        let gateway_addr = gateway.local_addr().unwrap();
        let gateway_thread = thread::spawn(move || {
            serve_listener(gateway, &upstream_addr.to_string(), Some(1)).unwrap();
        });

        let mut client = TcpStream::connect(gateway_addr).unwrap();
        client
            .write_all(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut response = Vec::new();
        if let Err(error) = client.read_to_end(&mut response) {
            assert!(
                !matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ),
                "client was stranded after upstream reset: {error:?}"
            );
        }

        upstream_thread.join().unwrap();
        gateway_thread.join().unwrap();
    }
}
