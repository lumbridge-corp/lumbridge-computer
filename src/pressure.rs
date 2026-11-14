//! Memory pressure (PSI) as an early warning, next to the `MemAvailable` floor.
//!
//! `MemAvailable` says how much could be handed out; it does not say whether the
//! kernel is already struggling to hand it out. On a unified-memory box the gap
//! matters: reclaim and compaction stall tasks well before the number reaches the
//! watchdog floor, and DGX Spark users report boxes that froze with `MemAvailable`
//! still looking healthy. `/proc/pressure/memory` measures exactly that struggle:
//! the share of wall time in which some (or all) runnable tasks were stalled
//! waiting for memory.
//!
//! This module only *warns*. Nothing is stopped or refused on pressure: the floor
//! and admission stay the only actors until we have watched these numbers on
//! real boxes long enough to trust a threshold with a model's life.

use std::fs;

/// "Some tasks stalled" over the last 10 seconds, in percent, that is worth a log line.
pub const WARN_SOME_AVG10: f64 = 10.0;
/// "All tasks stalled" over the last 10 seconds, in percent, that is worth a log line.
/// Much lower: `full` is time in which nothing but reclaim made progress.
pub const WARN_FULL_AVG10: f64 = 2.0;
/// Standing pressure is logged again this often.
pub const REPEAT_SEC: u64 = 300;

/// One reading of `/proc/pressure/memory`, in percent of wall time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pressure {
    pub some_avg10: f64,
    pub some_avg60: f64,
    pub full_avg10: f64,
    pub full_avg60: f64,
}

impl Pressure {
    pub fn elevated(&self) -> bool {
        self.some_avg10 >= WARN_SOME_AVG10 || self.full_avg10 >= WARN_FULL_AVG10
    }
}

/// `None` when the kernel has no PSI (`CONFIG_PSI` off, or `psi=0`) or the file is
/// unreadable: pressure is an extra signal, never a reason to fail.
pub fn read() -> Option<Pressure> {
    parse(&fs::read_to_string("/proc/pressure/memory").ok()?)
}

/// Split out from `read` so the parser is testable without a real /proc.
pub fn parse(text: &str) -> Option<Pressure> {
    let field = |line: &str, key: &str| -> Option<f64> {
        line.split_whitespace()
            .find_map(|part| part.strip_prefix(key))
            .and_then(|value| value.parse().ok())
    };
    let line = |kind: &str| text.lines().find(|line| line.starts_with(kind));
    let some = line("some")?;
    // `full` is absent at the system level on kernels before 5.13; read it as calm.
    let full = line("full").unwrap_or("");
    Some(Pressure {
        some_avg10: field(some, "avg10=")?,
        some_avg60: field(some, "avg60=")?,
        full_avg10: field(full, "avg10=").unwrap_or(0.0),
        full_avg60: field(full, "avg60=").unwrap_or(0.0),
    })
}

/// Turns a stream of readings into the few log lines worth printing: when pressure
/// becomes elevated, every `REPEAT_SEC` while it stays so, and when it eases.
#[derive(Debug, Default)]
pub struct Watch {
    warned_at: Option<u64>,
}

impl Watch {
    /// `now` is any monotonic clock in seconds. `available_gb` is only quoted.
    pub fn observe(
        &mut self,
        pressure: Option<Pressure>,
        available_gb: Option<f64>,
        now: u64,
    ) -> Option<String> {
        let pressure = pressure?;
        let available = available_gb.map_or(String::new(), |gb| {
            format!(", {gb:.1} GB available to models")
        });
        if pressure.elevated() {
            let due = self
                .warned_at
                .is_none_or(|at| now.saturating_sub(at) >= REPEAT_SEC);
            if !due {
                return None;
            }
            self.warned_at = Some(now);
            return Some(format!(
                "memory pressure is elevated: tasks stalled {:.1}% of the last 10 s \
                 (all tasks {:.1}%; 60 s: {:.1}% / {:.1}%){available}. Warning only, nothing was stopped",
                pressure.some_avg10, pressure.full_avg10, pressure.some_avg60, pressure.full_avg60
            ));
        }
        self.warned_at.take().map(|_| {
            format!(
                "memory pressure eased: tasks stalled {:.1}% of the last 10 s{available}",
                pressure.some_avg10
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CALM: &str = "some avg10=0.00 avg60=0.00 avg300=0.00 total=358981823\n\
                        full avg10=0.00 avg60=0.00 avg300=0.00 total=358611228\n";

    fn reading(some: f64, full: f64) -> Option<Pressure> {
        Some(Pressure {
            some_avg10: some,
            some_avg60: some / 2.0,
            full_avg10: full,
            full_avg60: full / 2.0,
        })
    }

    #[test]
    fn parses_the_kernel_format_and_tolerates_a_missing_full_line() {
        assert_eq!(
            parse(CALM),
            Some(Pressure {
                some_avg10: 0.0,
                some_avg60: 0.0,
                full_avg10: 0.0,
                full_avg60: 0.0
            })
        );
        let busy = "some avg10=23.50 avg60=11.25 avg300=3.00 total=1\n\
                    full avg10=4.75 avg60=1.50 avg300=0.40 total=1\n";
        let p = parse(busy).unwrap();
        assert_eq!(
            (p.some_avg10, p.some_avg60, p.full_avg10, p.full_avg60),
            (23.5, 11.25, 4.75, 1.5)
        );
        // Kernels before 5.13 have no system-wide `full` line.
        let old = parse("some avg10=12.00 avg60=1.00 avg300=0.00 total=1\n").unwrap();
        assert_eq!((old.some_avg10, old.full_avg10), (12.0, 0.0));
        // Garbage or an empty file is "no reading", not a calm one.
        assert_eq!(parse(""), None);
        assert_eq!(parse("some avg10=abc avg60=0.00\n"), None);
    }

    #[test]
    fn either_kind_of_stall_makes_it_elevated() {
        assert!(!reading(9.9, 1.9).unwrap().elevated());
        assert!(reading(WARN_SOME_AVG10, 0.0).unwrap().elevated());
        assert!(reading(0.5, WARN_FULL_AVG10).unwrap().elevated());
    }

    #[test]
    fn it_speaks_on_onset_at_intervals_and_on_recovery_and_is_otherwise_silent() {
        let mut watch = Watch::default();
        // Calm, and a kernel without PSI: nothing to say.
        assert_eq!(watch.observe(reading(0.0, 0.0), Some(20.0), 0), None);
        assert_eq!(watch.observe(None, Some(20.0), 5), None);

        let onset = watch.observe(reading(25.0, 3.0), Some(6.5), 10).unwrap();
        assert!(onset.contains("elevated") && onset.contains("25.0%"));
        assert!(onset.contains("6.5 GB available to models"));
        assert!(onset.contains("nothing was stopped"));
        // Still elevated: quiet until the repeat interval.
        for now in [15, 60, 10 + REPEAT_SEC - 1] {
            assert_eq!(watch.observe(reading(30.0, 5.0), Some(5.0), now), None);
        }
        assert!(watch
            .observe(reading(30.0, 5.0), None, 10 + REPEAT_SEC)
            .is_some());

        let eased = watch.observe(reading(1.0, 0.0), Some(15.0), 400).unwrap();
        assert!(eased.contains("eased"));
        // Recovery is said once; the next onset is news again.
        assert_eq!(watch.observe(reading(1.0, 0.0), Some(15.0), 405), None);
        assert!(watch.observe(reading(12.0, 0.0), Some(9.0), 410).is_some());
    }
}
