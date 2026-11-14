//! Unified-memory sensing. On a shared-memory box there's one pool, so `MemAvailable`
//! from /proc/meminfo is the single source of truth the Governor guards.

use anyhow::{Context, Result};
use std::fs;

pub struct MemInfo {
    pub total_gb: f64,
    pub available_gb: f64,
}

pub fn read() -> Result<MemInfo> {
    #[cfg(test)]
    if let Some((total_gb, available_gb)) = TEST_MEMORY.with(|value| value.get()) {
        return Ok(MemInfo {
            total_gb,
            available_gb,
        });
    }
    let s = fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;
    Ok(parse_meminfo(&s))
}

// Test-only and thread-local: lifecycle fixtures do not need a large CI host.
// Production always reads /proc/meminfo and has no override mechanism.
#[cfg(test)]
thread_local! {
    static TEST_MEMORY: std::cell::Cell<Option<(f64, f64)>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
pub struct TestMemory(Option<(f64, f64)>);

#[cfg(test)]
impl Drop for TestMemory {
    fn drop(&mut self) {
        TEST_MEMORY.with(|value| value.set(self.0));
    }
}

#[cfg(test)]
pub fn test_memory(total_gb: f64, available_gb: f64) -> TestMemory {
    TestMemory(TEST_MEMORY.with(|value| value.replace(Some((total_gb, available_gb)))))
}

/// Split out from `read` so the parser is testable without a real /proc.
pub fn parse_meminfo(s: &str) -> MemInfo {
    let mut total = 0.0;
    let mut available = 0.0;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("MemTotal:") {
            total = kb_to_gb(v);
        } else if let Some(v) = line.strip_prefix("MemAvailable:") {
            available = kb_to_gb(v);
        }
    }
    MemInfo {
        total_gb: total,
        available_gb: available,
    }
}

/// Parse a "   123456 kB" meminfo value into GB.
fn kb_to_gb(v: &str) -> f64 {
    v.split_whitespace()
        .next()
        .and_then(|n| n.parse::<f64>().ok())
        .unwrap_or(0.0)
        / 1024.0
        / 1024.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPARK: &str = "\
MemTotal:       127512345 kB
MemFree:         2048000 kB
MemAvailable:   104857600 kB
Buffers:          123456 kB
";

    #[test]
    fn reads_total_and_available_and_ignores_other_fields() {
        let m = parse_meminfo(SPARK);
        assert!((m.total_gb - 121.6).abs() < 0.1, "total was {}", m.total_gb);
        assert!(
            (m.available_gb - 100.0).abs() < 0.1,
            "available was {}",
            m.available_gb
        );
    }

    #[test]
    fn missing_fields_read_as_zero_rather_than_panicking() {
        // A zero here is what makes admission refuse, so an unparseable /proc must not
        // look like an empty box with room to spare.
        let m = parse_meminfo("SomethingElse: 1 kB\n");
        assert_eq!(m.total_gb, 0.0);
        assert_eq!(m.available_gb, 0.0);
    }

    #[test]
    fn malformed_values_do_not_panic() {
        let m = parse_meminfo("MemTotal:       not-a-number kB\nMemAvailable:\n");
        assert_eq!(m.total_gb, 0.0);
        assert_eq!(m.available_gb, 0.0);
    }

    #[test]
    fn memavailable_is_not_confused_with_memfree() {
        // MemFree is much smaller than MemAvailable on a box with page cache; picking the
        // wrong one would make the watchdog fire constantly.
        let m = parse_meminfo(SPARK);
        assert!(m.available_gb > 50.0);
    }
}
