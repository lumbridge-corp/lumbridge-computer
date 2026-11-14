//! Lumbridge Computer's model gateway: one address for every model on the box.
//!
//! Two modes share the listen address:
//! - `http` (default): an OpenAI-aware L7 proxy that reads the request's `model`
//!   and forwards it to whichever running model serves that name, streaming the
//!   response through untouched. See `docs/gateway.md`.
//! - `tcp`: the pre-M1 byte pump to a single upstream, kept for rollback.

mod extract;
mod http;
mod routes;
mod tcp;

use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::states::Lifecycle;

/// Which gateway implementation serves the listen address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    /// Route by the request's `model` (OpenAI-aware, the default).
    Http,
    /// Forward raw bytes to `--upstream` (the pre-M1 behaviour, for rollback).
    Tcp,
}

pub const DEFAULT_MAX_BODY_MIB: usize = 64;

/// Where `status` looks for a running agent's gateway; matches the `--listen` default.
pub const DEFAULT_LISTEN: &str = "127.0.0.1:8011";

/// The gateway's own read-only endpoint for in-flight counts. Its own routes
/// live under `/lumbridge-computer/v1/`, so they can never collide with an
/// OpenAI path a client forwards to a model.
pub const INFLIGHT_PATH: &str = "/lumbridge-computer/v1/inflight";

/// GET the in-flight counts body from the agent at `addr`, or `None` when no
/// agent answers with one (so `status` stays instant when no agent runs).
pub fn fetch_inflight(addr: &str, connect: Duration, read: Duration) -> Option<String> {
    let url = format!("http://{addr}{INFLIGHT_PATH}");
    crate::engines::http::fetch_http_within(&url, connect, read)
        .ok()
        .filter(|body| body.contains("\"in_flight\""))
}

pub struct Options {
    pub mode: Mode,
    pub listen: String,
    /// `tcp`: the only upstream. `http`: where requests that carry no `model`
    /// (and don't need one) go, e.g. `/metrics` or `/tokenize`.
    pub upstream: String,
    /// Lumbridge Computer root: the registry and state file the route table is built from.
    pub root: PathBuf,
    /// Largest request body the `http` mode buffers to read `model` from.
    pub max_body_bytes: usize,
    /// How long a request for a stopped model is held while it cold-starts. A
    /// sleeping model uses its `serve.sleep.wake_timeout_sec` instead.
    pub wake_timeout: Duration,
    /// Requests held per model while it wakes; the next one gets a 503.
    pub max_held_per_model: u32,
}

/// A cold start of a 35B brain takes 4–6 minutes; a wake from sleep, seconds
/// (that deadline is per model, `serve.sleep.wake_timeout_sec`).
pub const DEFAULT_WAKE_TIMEOUT_SEC: u64 = 600;
pub const DEFAULT_MAX_HELD_PER_MODEL: u32 = 32;

/// `lifecycle` is the M3 model-states implementation. `None` keeps the pre-M3
/// gateway exactly: only running models route, and scene modes are ignored.
pub fn run(
    options: &Options,
    in_flight: InFlight,
    lifecycle: Option<Arc<dyn Lifecycle>>,
) -> Result<()> {
    match options.mode {
        Mode::Tcp => tcp::run(&options.listen, &options.upstream),
        Mode::Http => http::run(options, in_flight, lifecycle),
    }
}

/// Requests the gateway currently holds open per registry model id, counted from
/// routing until the response body finishes or the client goes away (streams
/// included, and requests held while their model wakes). Supervision reads it to
/// tell a busy engine from a wedged one; the idle sleeper reads `idle_for`.
///
/// A cheap clone: every clone shares the same counters.
#[derive(Debug, Clone, Default)]
pub struct InFlight {
    // One entry per model id, created on first use and never removed (ids are
    // bounded by the registry). The map lock is only written when a new id
    // appears; every other access is a read lock plus an atomic op.
    counters: Arc<RwLock<HashMap<String, Arc<ModelCounters>>>>,
}

#[derive(Debug, Default)]
struct ModelCounters {
    in_flight: AtomicU32,
    /// The subset of `in_flight` held while the model wakes.
    waiting: AtomicU32,
    /// When the last request for this model finished, Unix ms (0 = never): the
    /// idle clock. Wall-clock so it serialises for `/lumbridge-computer/v1/inflight` and
    /// compares with the persisted `state_since_ms`.
    last_finished_ms: AtomicU64,
}

fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |d| d.as_millis() as u64)
}

impl InFlight {
    pub fn get(&self, model_id: &str) -> u32 {
        self.entry(model_id)
            .map_or(0, |c| c.in_flight.load(Ordering::Acquire))
    }

    /// Every model id the gateway has routed to, with its current count (often 0).
    /// Sorted, so `status --json` is stable to diff.
    pub fn snapshot(&self) -> BTreeMap<String, u32> {
        self.collect(|c| c.in_flight.load(Ordering::Acquire))
    }

    /// Requests held per model while it wakes (also counted in `snapshot`).
    pub fn waiting_snapshot(&self) -> BTreeMap<String, u32> {
        self.collect(|c| c.waiting.load(Ordering::Acquire))
    }

    /// How long `model_id` has been idle: `Some(ZERO)` while any request is in
    /// flight or held, `None` if no request for it has finished since the
    /// gateway started (the caller falls back to its own clock, e.g. ready-since).
    // The binary's idle sleeper reads `last_finished_ms` through `states::Host`
    // (it compares with persisted timestamps); this is the same clock as a Duration.
    #[allow(dead_code)]
    pub fn idle_for(&self, model_id: &str) -> Option<Duration> {
        let counters = self.entry(model_id)?;
        if counters.in_flight.load(Ordering::Acquire) > 0 {
            return Some(Duration::ZERO);
        }
        let last = counters.last_finished_ms.load(Ordering::Acquire);
        (last > 0).then(|| Duration::from_millis(unix_now_ms().saturating_sub(last)))
    }

    /// Requests held for `model_id` while it wakes (a subset of `get`).
    pub fn waiting(&self, model_id: &str) -> u32 {
        self.entry(model_id)
            .map_or(0, |c| c.waiting.load(Ordering::Acquire))
    }

    /// When the last request for `model_id` finished (Unix ms), or `None` if
    /// none has since the gateway started.
    pub fn last_finished_ms(&self, model_id: &str) -> Option<u64> {
        self.entry(model_id)
            .map(|c| c.last_finished_ms.load(Ordering::Acquire))
            .filter(|ms| *ms > 0)
    }

    /// `last_finished_ms` for every model that has finished a request.
    pub fn last_finished_snapshot(&self) -> BTreeMap<String, u64> {
        let counters = self.counters.read().unwrap_or_else(|e| e.into_inner());
        counters
            .iter()
            .filter_map(|(id, c)| {
                let ms = c.last_finished_ms.load(Ordering::Acquire);
                (ms > 0).then(|| (id.clone(), ms))
            })
            .collect()
    }

    /// Count one request against `model_id` until the guard drops.
    pub(crate) fn begin(&self, model_id: &str) -> InFlightGuard {
        let counters = self.entry_or_insert(model_id);
        counters.in_flight.fetch_add(1, Ordering::AcqRel);
        InFlightGuard(counters)
    }

    /// Mark one request as held for `model_id`'s wake until the guard drops.
    /// Returns the held count including this one, for the per-model cap.
    pub(crate) fn begin_waiting(&self, model_id: &str) -> (WaitingGuard, u32) {
        let counters = self.entry_or_insert(model_id);
        let held = counters.waiting.fetch_add(1, Ordering::AcqRel) + 1;
        (WaitingGuard(counters), held)
    }

    fn entry(&self, model_id: &str) -> Option<Arc<ModelCounters>> {
        let counters = self.counters.read().unwrap_or_else(|e| e.into_inner());
        counters.get(model_id).cloned()
    }

    fn entry_or_insert(&self, model_id: &str) -> Arc<ModelCounters> {
        self.entry(model_id).unwrap_or_else(|| {
            let mut counters = self.counters.write().unwrap_or_else(|e| e.into_inner());
            counters.entry(model_id.to_string()).or_default().clone()
        })
    }

    fn collect(&self, read: impl Fn(&ModelCounters) -> u32) -> BTreeMap<String, u32> {
        let counters = self.counters.read().unwrap_or_else(|e| e.into_inner());
        counters
            .iter()
            .map(|(id, c)| (id.clone(), read(c)))
            .collect()
    }
}

/// Supervision reads the gateway's counts: a model with requests in flight is
/// busy, so a slow probe is never mistaken for a hang.
impl crate::health::InFlight for InFlight {
    fn get(&self, model_id: &str) -> u32 {
        InFlight::get(self, model_id)
    }
}

/// Decrements its model's counter exactly once, however the request ends, and
/// stamps the idle clock.
#[derive(Debug)]
pub(crate) struct InFlightGuard(Arc<ModelCounters>);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        // Stamp before decrementing, so `idle_for` never sees zero in flight
        // with a timestamp older than this request.
        self.0
            .last_finished_ms
            .fetch_max(unix_now_ms(), Ordering::AcqRel);
        self.0.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug)]
pub(crate) struct WaitingGuard(Arc<ModelCounters>);

impl Drop for WaitingGuard {
    fn drop(&mut self) {
        self.0.waiting.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_flight_counts_per_model_and_clones_share_counters() {
        let in_flight = InFlight::default();
        let reader = in_flight.clone();
        assert_eq!(reader.get("brain"), 0);
        let a = in_flight.begin("brain");
        let b = in_flight.begin("brain");
        let c = in_flight.begin("embed");
        assert_eq!(reader.get("brain"), 2);
        assert_eq!(crate::health::InFlight::get(&reader, "embed"), 1);
        drop(a);
        drop(c);
        assert_eq!(reader.get("brain"), 1);
        assert_eq!(reader.get("embed"), 0);
        assert_eq!(
            reader.snapshot().into_iter().collect::<Vec<_>>(),
            vec![("brain".to_string(), 1), ("embed".to_string(), 0)]
        );
        drop(b);
        assert_eq!(reader.get("brain"), 0);
    }

    #[test]
    fn idle_for_is_zero_while_busy_and_counts_from_the_last_finished_request() {
        let in_flight = InFlight::default();
        assert_eq!(in_flight.idle_for("brain"), None, "never used");
        let a = in_flight.begin("brain");
        assert_eq!(in_flight.idle_for("brain"), Some(Duration::ZERO));
        let (held, count) = in_flight.begin_waiting("brain");
        assert_eq!(count, 1);
        assert_eq!(in_flight.waiting_snapshot()["brain"], 1);
        drop(held);
        assert_eq!(in_flight.waiting_snapshot()["brain"], 0);
        drop(a);
        let idle = in_flight.idle_for("brain").unwrap();
        assert!(idle < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(20));
        assert!(in_flight.idle_for("brain").unwrap() >= Duration::from_millis(20));
    }

    #[test]
    fn the_last_finished_time_is_stamped_when_a_request_ends() {
        let in_flight = InFlight::default();
        let guard = in_flight.begin("brain");
        assert_eq!(in_flight.last_finished_ms("brain"), None);
        assert_eq!(in_flight.last_finished_ms("never"), None);
        drop(guard);
        let stamped = in_flight.last_finished_ms("brain").unwrap();
        assert!(stamped > 1_700_000_000_000);
        assert_eq!(in_flight.last_finished_snapshot()["brain"], stamped);
    }
}
