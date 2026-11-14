//! Measured model memory and learned footprints (M2).
//!
//! `footprint_gb` in the registry is a number someone typed. This module
//! measures what each model really holds and remembers it:
//!
//! - **host** memory from the model's own scope (`memory.current`, `memory.stat`,
//!   `memory.swap.current`, `memory.peak`), or, for unscoped/adopted models,
//!   from `/proc/<pid>/smaps_rollup` summed over the process group;
//! - **GPU** memory from `nvidia-smi --query-compute-apps`, because on GB10
//!   CUDA allocations are driver pages that appear in *neither* the cgroup nor
//!   smaps (see docs/memory.md for measured numbers). The aggregate
//!   `memory.used` is "[N/A]" on GB10; the per-process column is not.
//!
//! A model's *demand* is host + GPU + swap: swapped host pages still belong to
//! the model, and under a scope's `MemorySwapMax=0` they become resident.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::config::{AdmissionMode, Registry};
use crate::proc::State;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
/// Footprint samples are kept this long.
const RETENTION_MS: i64 = 30 * 24 * 60 * 60 * 1000;
/// Window `admission: measured` learns from.
pub const ADMISSION_WINDOW: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// A model version needs this many samples (10 minutes at the sampling interval)
/// before admission trusts them: a model that is still loading weighs almost nothing.
pub const MIN_VERSION_SAMPLES: usize = 20;
/// How often the resident agent samples.
pub const SAMPLE_INTERVAL: Duration = Duration::from_secs(30);

/// Where `/proc` and the cgroup v2 mount live; tests point these at temp dirs.
#[derive(Debug, Clone)]
pub struct Roots {
    pub proc: PathBuf,
    pub cgroup: PathBuf,
}

impl Default for Roots {
    fn default() -> Self {
        Roots {
            proc: PathBuf::from("/proc"),
            cgroup: PathBuf::from("/sys/fs/cgroup"),
        }
    }
}

/// One live reading of one model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measurement {
    /// `scope` (own cgroup), `container` (a `docker run` model's container
    /// cgroup) or `unscoped` (process-group smaps).
    pub source: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cgroup: Option<String>,
    /// Unreclaimable host memory: anon + shmem (+ kernel, for a scope).
    pub host_gb: f64,
    /// Page cache charged to the model; reclaimable, so not part of demand.
    pub file_gb: f64,
    /// CUDA memory from NVML; `None` when nvidia-smi could not be read.
    pub gpu_gb: Option<f64>,
    pub swap_gb: f64,
    /// Kernel high-water mark of the scope (`memory.peak`, includes cache).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_host_gb: Option<f64>,
    pub pids: usize,
}

impl Measurement {
    pub fn resident_gb(&self) -> f64 {
        self.host_gb + self.gpu_gb.unwrap_or(0.0)
    }

    /// What this model would hold with no swap: the number to compare with
    /// `footprint_gb`.
    pub fn demand_gb(&self) -> f64 {
        self.resident_gb() + self.swap_gb
    }
}

/// Per-process CUDA memory. A trait so tests never run nvidia-smi.
pub trait GpuSource {
    /// MiB per pid, or `None` if the source is unavailable.
    fn per_pid_mib(&self) -> Option<HashMap<i32, f64>>;
}

/// `nvidia-smi --query-compute-apps`, bounded by a timeout: under memory
/// pressure (exactly when the watchdog asks) a wedged nvidia-smi must not
/// stall the floor.
pub struct NvidiaSmi;

impl GpuSource for NvidiaSmi {
    fn per_pid_mib(&self) -> Option<HashMap<i32, f64>> {
        let mut child = Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            match child.try_wait() {
                Ok(Some(status)) if status.success() => break,
                Ok(Some(_)) | Err(_) => return None,
                Ok(None) if Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            }
        }
        let mut out = String::new();
        child.stdout.take()?.read_to_string(&mut out).ok()?;
        Some(parse_compute_apps(&out))
    }
}

/// Parse `pid, used_memory` CSV lines (MiB). `[N/A]` rows are skipped.
pub fn parse_compute_apps(text: &str) -> HashMap<i32, f64> {
    let mut map = HashMap::new();
    for line in text.lines() {
        let mut fields = line.split(',').map(str::trim);
        let (Some(pid), Some(mib)) = (fields.next(), fields.next()) else {
            continue;
        };
        if let (Ok(pid), Ok(mib)) = (pid.parse::<i32>(), mib.parse::<f64>()) {
            *map.entry(pid).or_insert(0.0) += mib;
        }
    }
    map
}

/// The model's cgroup path (relative to the cgroup root) when it lives in its
/// own scope: one Lumbridge Computer created, or a `run-*.scope` from a registry command
/// that wraps itself in systemd-run. Any other cgroup (the agent's service, a
/// terminal's session scope) is shared, so it says nothing about one model.
pub fn own_scope(roots: &Roots, pid: i32) -> Option<String> {
    let path = cgroup_of(roots, pid)?;
    let leaf = path.rsplit('/').next()?;
    let ours = leaf.starts_with(crate::scope::UNIT_PREFIX) || leaf.starts_with("run-");
    (ours && leaf.ends_with(".scope")).then_some(path)
}

/// `pid`'s cgroup v2 path relative to the cgroup root (the `0::` line of
/// `/proc/<pid>/cgroup`, without its leading `/`).
pub fn cgroup_of(roots: &Roots, pid: i32) -> Option<String> {
    let text = fs::read_to_string(roots.proc.join(pid.to_string()).join("cgroup")).ok()?;
    let path = text
        .lines()
        .find_map(|line| line.strip_prefix("0::"))?
        .trim()
        .trim_start_matches('/');
    Some(path.to_string())
}

/// True when `path` is `cgroup` or one of its descendants.
fn within(path: &str, cgroup: &str) -> bool {
    path == cgroup
        || path
            .strip_prefix(cgroup)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// The nvidia-smi pids whose cgroup is `cgroup` or below it. A container's
/// pids are host pids in nvidia-smi but are not in Lumbridge Computer's process group,
/// so they are found by cgroup, not by group membership.
pub fn gpu_pids_in(roots: &Roots, gpu: &HashMap<i32, f64>, cgroup: &str) -> Vec<i32> {
    gpu.keys()
        .copied()
        .filter(|pid| cgroup_of(roots, *pid).is_some_and(|path| within(&path, cgroup)))
        .collect()
}

fn read_bytes(path: &Path) -> Option<f64> {
    fs::read_to_string(path).ok()?.trim().parse::<f64>().ok()
}

fn stat_field(stat: &str, key: &str) -> f64 {
    stat.lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            (parts.next() == Some(key)).then(|| parts.next())?
        })
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// pids of every process whose process group is `pgid`.
fn group_members(roots: &Roots, pgid: i32) -> Vec<i32> {
    let Ok(entries) = fs::read_dir(&roots.proc) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse::<i32>().ok())
        .filter(|pid| {
            fs::read_to_string(roots.proc.join(pid.to_string()).join("stat"))
                .ok()
                .and_then(|stat| {
                    let close = stat.rfind(')')?;
                    stat[close + 1..].split_whitespace().nth(2)?.parse().ok()
                })
                == Some(pgid)
        })
        .collect()
}

fn gpu_gb(gpu: Option<&HashMap<i32, f64>>, pids: &[i32]) -> Option<f64> {
    gpu.map(|map| pids.iter().filter_map(|p| map.get(p)).sum::<f64>() / 1024.0)
}

/// A cgroup's own accounting: `memory.current - file + shmem` is the
/// unreclaimable host side (shmem is counted inside `file`).
fn measure_cgroup(
    roots: &Roots,
    cgroup: &str,
    source: &'static str,
    gpu: Option<f64>,
) -> Result<Measurement> {
    let dir = roots.cgroup.join(cgroup);
    let current = read_bytes(&dir.join("memory.current"))
        .with_context(|| format!("reading {}/memory.current", dir.display()))?;
    let stat = fs::read_to_string(dir.join("memory.stat")).unwrap_or_default();
    let file = stat_field(&stat, "file");
    let shmem = stat_field(&stat, "shmem");
    let pids = fs::read_to_string(dir.join("cgroup.procs"))
        .unwrap_or_default()
        .lines()
        .filter(|l| l.trim().parse::<i32>().is_ok())
        .count();
    Ok(Measurement {
        source,
        host_gb: (current - file + shmem).max(0.0) / GIB,
        file_gb: (file - shmem).max(0.0) / GIB,
        gpu_gb: gpu,
        swap_gb: read_bytes(&dir.join("memory.swap.current")).unwrap_or(0.0) / GIB,
        peak_host_gb: read_bytes(&dir.join("memory.peak")).map(|b| b / GIB),
        pids,
        cgroup: Some(cgroup.to_string()),
    })
}

fn cgroup_pids(roots: &Roots, cgroup: &str) -> Vec<i32> {
    fs::read_to_string(roots.cgroup.join(cgroup).join("cgroup.procs"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect()
}

/// Pss (host, file, swap) in bytes summed over process group `pgid`, and its pids.
fn group_pss(roots: &Roots, pgid: i32) -> (f64, f64, f64, Vec<i32>) {
    let pids = group_members(roots, pgid);
    let (mut host, mut file, mut swap) = (0.0, 0.0, 0.0);
    for member in &pids {
        let Ok(rollup) =
            fs::read_to_string(roots.proc.join(member.to_string()).join("smaps_rollup"))
        else {
            continue; // exited between listing and reading
        };
        let kb = |key: &str| {
            rollup
                .lines()
                .find_map(|l| l.strip_prefix(key))
                .and_then(|v| v.split_whitespace().next()?.parse::<f64>().ok())
                .unwrap_or(0.0)
                * 1024.0
        };
        // Pss, not Rss: forked workers share the parent's pages.
        host += kb("Pss_Anon:") + kb("Pss_Shmem:");
        file += kb("Pss_File:");
        swap += kb("SwapPss:");
    }
    (host, file, swap, pids)
}

/// Measure the model whose process-group leader is `pid`.
pub fn measure(roots: &Roots, pid: i32, gpu: Option<&HashMap<i32, f64>>) -> Result<Measurement> {
    if let Some(cgroup) = own_scope(roots, pid) {
        let pids = cgroup_pids(roots, &cgroup);
        return measure_cgroup(roots, &cgroup, "scope", gpu_gb(gpu, &pids));
    }
    let (host, file, swap, pids) = group_pss(roots, pid);
    if pids.is_empty() {
        anyhow::bail!("process group {pid} has no live members");
    }
    Ok(Measurement {
        source: "unscoped",
        cgroup: None,
        host_gb: host / GIB,
        file_gb: file / GIB,
        gpu_gb: gpu_gb(gpu, &pids),
        swap_gb: swap / GIB,
        peak_host_gb: None,
        pids: pids.len(),
    })
}

/// Measure a `docker run` model: the container's cgroup (host side), every
/// nvidia-smi pid inside that cgroup (GPU), plus the docker CLI's own small
/// Pss, which lives in Lumbridge Computer's process group `cli_pid`, not the container.
pub fn measure_container(
    roots: &Roots,
    cgroup: &str,
    cli_pid: i32,
    gpu: Option<&HashMap<i32, f64>>,
) -> Result<Measurement> {
    let container_gpu = gpu.and_then(|map| gpu_gb(Some(map), &gpu_pids_in(roots, map, cgroup)));
    let mut m = measure_cgroup(roots, cgroup, "container", container_gpu)?;
    let (host, file, swap, cli) = group_pss(roots, cli_pid);
    m.host_gb += host / GIB;
    m.file_gb += file / GIB;
    m.swap_gb += swap / GIB;
    m.pids += cli.len();
    Ok(m)
}

/// Measure one owned record: through its container when it is a `docker run`
/// model whose container can be found, else as before.
fn measure_proc(
    roots: &Roots,
    process: &crate::proc::Proc,
    gpu: Option<&HashMap<i32, f64>>,
    docker: &dyn crate::docker::Docker,
) -> Result<Measurement> {
    let container =
        crate::docker::handle(&roots.proc, process.container_id.as_deref(), process.pid)
            .and_then(|handle| crate::docker::resolve(docker, &handle).ok().flatten())
            .and_then(|id| crate::docker::cgroup(docker, roots, &id));
    match container {
        Some(cgroup) => measure_container(roots, &cgroup, process.pid, gpu),
        None => measure(roots, process.pid, gpu),
    }
}

/// Live measurement of every identity-owned model in `state`.
pub fn measure_owned(
    roots: &Roots,
    state: &State,
    gpu: &dyn GpuSource,
) -> BTreeMap<String, Measurement> {
    measure_owned_with(roots, state, gpu, crate::docker::system())
}

/// `measure_owned` with an explicit docker, for tests.
pub fn measure_owned_with(
    roots: &Roots,
    state: &State,
    gpu: &dyn GpuSource,
    docker: &dyn crate::docker::Docker,
) -> BTreeMap<String, Measurement> {
    let gpu = gpu.per_pid_mib();
    state
        .procs
        .iter()
        .filter(|(_, p)| p.owned_alive())
        .filter_map(|(id, p)| {
            Some((
                id.clone(),
                measure_proc(roots, p, gpu.as_ref(), docker).ok()?,
            ))
        })
        .collect()
}

/// How far each measured model is over its declared footprint (GB, ≥ 0), for
/// the watchdog's shed order.
pub fn overage_gb(
    registry: &Registry,
    measured: &BTreeMap<String, Measurement>,
) -> BTreeMap<String, f64> {
    measured
        .iter()
        .filter_map(|(id, m)| {
            let declared = registry.models.get(id)?.footprint_gb;
            Some((id.clone(), (m.demand_gb() - declared).max(0.0)))
        })
        .collect()
}

// ---- durable samples --------------------------------------------------------

pub struct Samples {
    conn: Connection,
}

impl Samples {
    /// Opens the telemetry database and adds `footprint_samples` if missing.
    /// Additive only: no existing table or schema version is touched.
    pub fn open(root: &Path) -> Result<Self> {
        let path = crate::telemetry::db_path(root);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let conn = Connection::open(&path)
            .with_context(|| format!("opening telemetry database {}", path.display()))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS footprint_samples (
               timestamp_ms INTEGER NOT NULL,
               model TEXT NOT NULL,
               source TEXT NOT NULL,
               host_gb REAL NOT NULL,
               file_gb REAL NOT NULL,
               gpu_gb REAL,
               swap_gb REAL NOT NULL,
               peak_host_gb REAL,
               PRIMARY KEY (timestamp_ms, model)
             );
             CREATE INDEX IF NOT EXISTS footprint_samples_model_time
               ON footprint_samples(model, timestamp_ms);",
        )?;
        // Added 2026-11: which definition of the model a sample belongs to
        // (`model_version`). Rows from before have NULL and belong to no version.
        let has_version = conn
            .prepare("SELECT 1 FROM pragma_table_info('footprint_samples') WHERE name = 'version'")?
            .exists([])?;
        if !has_version {
            conn.execute_batch("ALTER TABLE footprint_samples ADD COLUMN version TEXT;")?;
        }
        Ok(Samples { conn })
    }

    /// A sample that belongs to no model definition, like every row written before
    /// versions existed. Only tests write these now.
    #[cfg(test)]
    pub fn record(&self, timestamp_ms: i64, model: &str, m: &Measurement) -> Result<()> {
        self.record_versioned(timestamp_ms, model, None, m)
    }

    /// `record`, tagged with the model definition it was measured under.
    pub fn record_versioned(
        &self,
        timestamp_ms: i64,
        model: &str,
        version: Option<&str>,
        m: &Measurement,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO footprint_samples
               (timestamp_ms, model, source, host_gb, file_gb, gpu_gb, swap_gb, peak_host_gb, version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                timestamp_ms,
                model,
                m.source,
                m.host_gb,
                m.file_gb,
                m.gpu_gb,
                m.swap_gb,
                m.peak_host_gb,
                version
            ],
        )?;
        Ok(())
    }

    pub fn prune(&self, now_ms: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM footprint_samples WHERE timestamp_ms < ?1",
            params![now_ms - RETENTION_MS],
        )?;
        Ok(())
    }

    /// Models with any sample since `since_ms`.
    pub fn models(&self, since_ms: i64) -> Result<Vec<String>> {
        let mut statement = self.conn.prepare(
            "SELECT DISTINCT model FROM footprint_samples WHERE timestamp_ms >= ?1 ORDER BY model",
        )?;
        let rows = statement
            .query_map(params![since_ms], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    pub fn learned(&self, model: &str, since_ms: i64) -> Result<Option<Learned>> {
        let mut statement = self.conn.prepare(
            "SELECT host_gb, gpu_gb, swap_gb FROM footprint_samples
             WHERE model = ?1 AND timestamp_ms >= ?2",
        )?;
        let rows: Vec<(f64, Option<f64>, f64)> = statement
            .query_map(params![model, since_ms], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(Learned::from_rows(&rows))
    }

    /// `learned`, from the samples of one model definition only, and only once there
    /// are `MIN_VERSION_SAMPLES` of them. This is what admission learns from: the id
    /// `embed` has been two different models, and the 7-day window must not let the
    /// old one's size stand in for the new one's.
    pub fn learned_version(
        &self,
        model: &str,
        version: &str,
        since_ms: i64,
    ) -> Result<Option<Learned>> {
        let mut statement = self.conn.prepare(
            "SELECT host_gb, gpu_gb, swap_gb FROM footprint_samples
             WHERE model = ?1 AND version = ?2 AND timestamp_ms >= ?3",
        )?;
        let rows: Vec<(f64, Option<f64>, f64)> = statement
            .query_map(params![model, version, since_ms], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(Learned::from_rows(&rows).filter(|learned| learned.samples >= MIN_VERSION_SAMPLES))
    }
}

/// A short, stable name for the definition a model is launched from: its whole
/// `serve` section (engine, weights, argv, args, env, port). Any change to it is a
/// different thing to measure, since context length, quantisation and
/// `gpu-memory-utilization` all move the footprint. `footprint_gb`, health and
/// supervision settings are left out: retyping the declared number must not throw
/// the measurements away. FNV-1a rather than `DefaultHasher`, whose output may
/// change between Rust releases and would orphan every stored sample.
pub fn model_version(model: &crate::config::Model) -> String {
    let text = serde_json::to_string(&model.serve).unwrap_or_default();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// What the samples in a window say about one model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Learned {
    pub samples: usize,
    /// False when no sample had a GPU reading: the numbers are host-only.
    pub gpu_seen: bool,
    pub peak_resident_gb: f64,
    pub peak_demand_gb: f64,
    pub p99_demand_gb: f64,
    /// p99 of host + GPU, swap left out: what admission budgets in `measured` mode.
    /// Not serialized: `footprints --json` keeps its shape.
    #[serde(skip)]
    pub p99_resident_gb: f64,
    pub max_swap_gb: f64,
}

impl Learned {
    /// Rows are `(host, gpu, swap)`. Samples without a GPU reading are ignored
    /// when any sample has one: a missed nvidia-smi must not look like a model
    /// that suddenly shrank by its whole KV cache.
    pub fn from_rows(rows: &[(f64, Option<f64>, f64)]) -> Option<Learned> {
        let gpu_seen = rows.iter().any(|r| r.1.is_some());
        let used: Vec<&(f64, Option<f64>, f64)> =
            rows.iter().filter(|r| !gpu_seen || r.1.is_some()).collect();
        if used.is_empty() {
            return None;
        }
        let resident: Vec<f64> = used.iter().map(|r| r.0 + r.1.unwrap_or(0.0)).collect();
        let mut demand: Vec<f64> = used
            .iter()
            .zip(&resident)
            .map(|(r, res)| res + r.2)
            .collect();
        demand.sort_by(f64::total_cmp);
        // Nearest-rank p99.
        let rank = ((0.99 * demand.len() as f64).ceil() as usize).clamp(1, demand.len());
        let mut resident_sorted = resident.clone();
        resident_sorted.sort_by(f64::total_cmp);
        Some(Learned {
            samples: used.len(),
            gpu_seen,
            peak_resident_gb: resident.iter().copied().fold(0.0, f64::max),
            peak_demand_gb: demand[demand.len() - 1],
            p99_demand_gb: demand[rank - 1],
            p99_resident_gb: resident_sorted[rank - 1],
            max_swap_gb: used.iter().map(|r| r.2).fold(0.0, f64::max),
        })
    }

    /// p99 demand plus max(1 GB, 10 %), rounded up to a whole GB.
    pub fn suggested_footprint_gb(&self) -> f64 {
        let p99 = self.p99_demand_gb;
        (p99 + (0.10 * p99).max(1.0)).ceil()
    }
}

// ---- admission --------------------------------------------------------------

/// What the admission window says about the model `id` is defined as *today*.
fn learned_current(root: &Path, registry: &Registry, id: &str) -> Option<Learned> {
    let version = model_version(registry.models.get(id)?);
    let now = crate::telemetry::unix_time_ms().ok()?;
    let since = now - ADMISSION_WINDOW.as_millis() as i64;
    let learned = Samples::open(root)
        .ok()?
        .learned_version(id, &version, since)
        .ok()??;
    learned.gpu_seen.then_some(learned)
}

/// What admission budgets for `id`. `declared` (the default) is exactly
/// `footprint_gb`. `measured` is `max(footprint_gb, learned p99 resident size over
/// 7 days)`: it only ever grows a number, so a model never gets admitted on a
/// smaller figure than its author declared.
///
/// Resident (host + GPU), not demand (resident + swap): the budget is RAM, and
/// pages a model has had swapped out are by definition not in it. Counting them
/// made `measured` refuse a running `everything` Scene on paper (about 116 GB
/// against a 111 GB budget) while the box had 16 GB available.
/// Only samples of the model's current definition count (`model_version`).
pub fn admission_gb(root: &Path, registry: &Registry, id: &str) -> f64 {
    let declared = registry.models.get(id).map_or(0.0, |m| m.footprint_gb);
    match registry.admission {
        AdmissionMode::Declared => declared,
        AdmissionMode::Measured => learned_current(root, registry, id)
            .map_or(declared, |learned| learned.p99_resident_gb.max(declared)),
    }
}

/// What the *observed* half of admission expects `id` to take out of the pool once
/// it is up: the largest resident size (host + GPU, swap excluded) sampled in the
/// admission window for its current definition, or its admission figure when that
/// definition has not been measured enough (`MIN_VERSION_SAMPLES`, with a GPU reading).
///
/// Unlike [`admission_gb`] this may be below the declared footprint, in either
/// admission mode. The observed check compares against sensed memory, which already
/// counts every running model at its real size; asking it for a padded, typed
/// number on top charges the padding against the safety margin a second time. That
/// is how a 10 GB-declared, 8.1 GB-real embedder was refused for 40 minutes after a
/// Docker restart on 2026-11-10 with the box 0.2 GB short of `10 + 8`, although the
/// same set of models had been running an hour earlier. Swap is excluded because a
/// starting model takes RAM, not the cold pages an old instance had paged out.
pub fn observed_gb(root: &Path, registry: &Registry, id: &str) -> f64 {
    learned_current(root, registry, id).map_or_else(
        || admission_gb(root, registry, id),
        |learned| learned.peak_resident_gb,
    )
}

/// `Pool::committed_gb` under the registry's admission mode: the running models of
/// `pool`, each at its admission figure. The pool still decides what fits.
///
/// M3: a model the lifecycle holds asleep counts at its sleeping footprint, and
/// an owned process counts while it starts or wakes (see `states::counted_gb`).
pub fn committed_gb(root: &Path, registry: &Registry, pool: &crate::pool::Pool) -> f64 {
    let state = State::load_checked(root).unwrap_or_default();
    crate::states::committed_gb(root, registry, &state, pool)
}

// ---- resident sampler -------------------------------------------------------

/// One sampling pass: measure every owned model and record it.
pub fn sample_once(root: &Path, roots: &Roots, gpu: &dyn GpuSource) -> Result<usize> {
    let state = State::load_checked(root)?;
    let measured = measure_owned(roots, &state, gpu);
    let samples = Samples::open(root)?;
    let now = crate::telemetry::unix_time_ms()?;
    // A registry that does not load must not stop sampling: the rows are then
    // unversioned, visible to `footprints` and ignored by admission.
    let registry = Registry::load(root).ok();
    for (id, m) in &measured {
        let version = registry
            .as_ref()
            .and_then(|registry| registry.models.get(id))
            .map(model_version);
        samples.record_versioned(now, id, version.as_deref(), m)?;
    }
    samples.prune(now)?;
    Ok(measured.len())
}

/// The agent's footprint thread. Errors are logged on change, not every tick.
pub fn run_sampler(root: PathBuf) {
    let roots = Roots::default();
    let mut last_error: Option<String> = None;
    loop {
        match sample_once(&root, &roots, &NvidiaSmi) {
            Ok(_) => {
                if last_error.take().is_some() {
                    eprintln!("footprint: sampling recovered");
                }
            }
            Err(error) => {
                let message = format!("{error:#}");
                if last_error.as_deref() != Some(message.as_str()) {
                    eprintln!("footprint: sampling failed: {message}");
                    last_error = Some(message);
                }
            }
        }
        std::thread::sleep(SAMPLE_INTERVAL);
    }
}

// ---- `lumbridge-computer footprints` --------------------------------------------------

#[derive(Debug, Serialize)]
pub struct Row {
    pub id: String,
    pub declared_gb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current: Option<Measurement>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learned: Option<Learned>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_footprint_gb: Option<f64>,
}

pub fn report(root: &Path, since: &str) -> Result<Vec<Row>> {
    let registry = Registry::load(root)?;
    let state = State::load_checked(root)?;
    let current = measure_owned(&Roots::default(), &state, &NvidiaSmi);
    let now = crate::telemetry::unix_time_ms()?;
    let since_ms = now
        - crate::telemetry::parse_duration(since)?
            .as_millis()
            .min(i64::MAX as u128) as i64;
    let samples = Samples::open(root)?;
    let mut ids: Vec<String> = current.keys().cloned().collect();
    ids.extend(samples.models(since_ms)?);
    ids.sort();
    ids.dedup();
    ids.into_iter()
        .map(|id| {
            let learned = samples.learned(&id, since_ms)?;
            Ok(Row {
                declared_gb: registry.models.get(&id).map_or(0.0, |m| m.footprint_gb),
                current: current.get(&id).cloned(),
                suggested_footprint_gb: learned.as_ref().map(Learned::suggested_footprint_gb),
                learned,
                id,
            })
        })
        .collect()
}

pub fn render_text(rows: &[Row], since: &str) -> String {
    let gb = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.1}"));
    let mut out = format!(
        "{:<22} {:>8} {:>8} {:>6} {:>6} {:>8} {:>8} {:>9}  SOURCE\n",
        "MODEL", "DECLARED", "CURRENT", "GPU", "SWAP", "PEAK", "P99", "SUGGESTED"
    );
    for row in rows {
        let current = row.current.as_ref();
        let learned = row.learned.as_ref();
        out.push_str(&format!(
            "{:<22} {:>8.1} {:>8} {:>6} {:>6} {:>8} {:>8} {:>9}  {}\n",
            row.id,
            row.declared_gb,
            gb(current.map(Measurement::demand_gb)),
            gb(current.and_then(|m| m.gpu_gb)),
            gb(current.map(|m| m.swap_gb)),
            gb(learned.map(|l| l.peak_demand_gb)),
            gb(learned.map(|l| l.p99_demand_gb)),
            gb(row.suggested_footprint_gb),
            current.map_or("not running", |m| m.source),
        ));
    }
    out.push_str(&format!(
        "\nCURRENT/PEAK/P99 = host + GPU + swap (GB); peak and p99 over {since}. \
         SUGGESTED = p99 + max(1 GB, 10%), rounded up.\n"
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lumbridge-computer-footprint-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A fake `/proc` entry: stat with the given pgrp, cgroup line, smaps_rollup.
    fn fake_proc(roots: &Roots, pid: i32, pgrp: i32, cgroup: &str, rollup_kb: [u64; 4]) {
        let dir = roots.proc.join(pid.to_string());
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("stat"),
            format!("{pid} (vllm serve) S 1 {pgrp} {pgrp} 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 99"),
        )
        .unwrap();
        fs::write(dir.join("cgroup"), format!("0::{cgroup}\n")).unwrap();
        let [anon, shmem, file, swap] = rollup_kb;
        fs::write(
            dir.join("smaps_rollup"),
            format!(
                "Rss: {} kB\nPss: {} kB\nPss_Anon: {anon} kB\nPss_File: {file} kB\nPss_Shmem: {shmem} kB\nSwap: {swap} kB\nSwapPss: {swap} kB\n",
                anon + shmem + file,
                anon + shmem + file
            ),
        )
        .unwrap();
    }

    fn roots() -> Roots {
        let base = temp_dir("roots");
        Roots {
            proc: base.join("proc"),
            cgroup: base.join("cgroup"),
        }
    }

    const AGENT: &str =
        "/user.slice/user-1000.slice/user@1000.service/app.slice/lumbridge-computer-agent.service";
    const SCOPE: &str = "user.slice/user-1000.slice/user@1000.service/lumbridge.slice/lumbridge-computer.slice/lumbridge-computer-models.slice/lumbridge-computer-model-brain-1.scope";

    #[test]
    fn an_unscoped_model_is_the_pss_of_its_whole_process_group_plus_its_gpu() {
        let roots = roots();
        // brain leader + engine core in one group; an unrelated process in the
        // same (shared agent) cgroup must not be counted.
        fake_proc(
            &roots,
            100,
            100,
            AGENT,
            [3 * 1048576, 1048576 / 2, 1048576, 0],
        );
        fake_proc(&roots, 101, 100, AGENT, [4 * 1048576, 0, 0, 1048576]);
        fake_proc(&roots, 200, 200, AGENT, [9 * 1048576, 0, 0, 0]);
        let gpu = parse_compute_apps("101, 30720\n200, 1024\n");
        let m = measure(&roots, 100, Some(&gpu)).unwrap();
        assert_eq!(m.source, "unscoped");
        assert_eq!(m.pids, 2);
        assert!((m.host_gb - 7.5).abs() < 1e-9, "{}", m.host_gb);
        assert!((m.file_gb - 1.0).abs() < 1e-9);
        assert!((m.swap_gb - 1.0).abs() < 1e-9);
        assert_eq!(m.gpu_gb, Some(30.0));
        assert!((m.demand_gb() - 38.5).abs() < 1e-9);
    }

    /// The shape of a real capture: a live brain (vLLM,
    /// :8001, pgid 1264867) is three processes in the agent's service cgroup.
    /// smaps sees 7.5 GiB; NVML sees another 29.2 GiB on the engine core.
    #[test]
    fn a_brain_capture_measures_about_37_gb_against_a_declared_40() {
        let roots = roots();
        fake_proc(&roots, 1264867, 1264867, AGENT, [2844496, 0, 140482, 0]);
        fake_proc(&roots, 1265785, 1264867, AGENT, [7124, 0, 973, 0]);
        fake_proc(&roots, 1265786, 1264867, AGENT, [4949556, 92616, 343530, 0]);
        fake_proc(
            &roots,
            19499,
            19499,
            AGENT,
            [1132000, 61000, 50000, 4079000],
        );
        let gpu = parse_compute_apps(
            "18466, 2993\n19499, 27072\n22070, 8857\n31529, 8152\n32896, 4212\n1265786, 29896\n",
        );
        let brain = measure(&roots, 1264867, Some(&gpu)).unwrap();
        assert_eq!(brain.pids, 3);
        assert!((brain.host_gb - 7.53).abs() < 0.01, "{}", brain.host_gb);
        assert!((brain.gpu_gb.unwrap() - 29.20).abs() < 0.01);
        assert!((brain.demand_gb() - 36.72).abs() < 0.01);
        // nemo-speech: 26.4 GiB of CUDA plus 3.9 GiB of swapped host pages.
        let speech = measure(&roots, 19499, Some(&gpu)).unwrap();
        assert!(speech.demand_gb() > 31.0, "{}", speech.demand_gb());
    }

    #[test]
    fn a_scoped_model_reads_its_own_cgroup_and_excludes_page_cache() {
        let roots = roots();
        fake_proc(&roots, 100, 100, &format!("/{SCOPE}"), [0, 0, 0, 0]);
        let dir = roots.cgroup.join(SCOPE);
        fs::create_dir_all(&dir).unwrap();
        let g = 1u64 << 30;
        fs::write(dir.join("memory.current"), format!("{}\n", 10 * g)).unwrap();
        fs::write(dir.join("memory.peak"), format!("{}\n", 12 * g)).unwrap();
        fs::write(dir.join("memory.swap.current"), format!("{}\n", 2 * g)).unwrap();
        fs::write(
            dir.join("memory.stat"),
            format!("anon {}\nfile {}\nshmem {}\nkernel 0\n", 6 * g, 4 * g, g),
        )
        .unwrap();
        fs::write(dir.join("cgroup.procs"), "100\n101\n").unwrap();
        let gpu = parse_compute_apps("101, 20480\n");
        let m = measure(&roots, 100, Some(&gpu)).unwrap();
        assert_eq!(m.source, "scope");
        assert_eq!(m.cgroup.as_deref(), Some(SCOPE));
        // current 10 - file 4 + shmem 1 (shmem is counted inside file).
        assert!((m.host_gb - 7.0).abs() < 1e-9, "{}", m.host_gb);
        assert!((m.file_gb - 3.0).abs() < 1e-9);
        assert_eq!(m.peak_host_gb, Some(12.0));
        assert_eq!(m.gpu_gb, Some(20.0));
        assert!((m.demand_gb() - 29.0).abs() < 1e-9);
    }

    #[test]
    fn only_lumbridge_computer_and_systemd_run_scopes_count_as_a_models_own_cgroup() {
        let roots = roots();
        for (pid, cg, own) in [
            (1, format!("/{SCOPE}"), true),
            (
                2,
                "/user.slice/u/app.slice/run-r8dc7d5d1.scope".to_string(),
                true,
            ),
            (3, AGENT.to_string(), false),
            (
                4,
                "/user.slice/user-1000.slice/session-3.scope".to_string(),
                false,
            ),
        ] {
            fake_proc(&roots, pid, pid, &cg, [0; 4]);
            assert_eq!(own_scope(&roots, pid).is_some(), own, "{cg}");
        }
    }

    const CONTAINER: &str = "system.slice/docker-1be159322466b720589f5a21d0f2bc6ae84d437dd49139fc27efc7d7e8546284.scope";
    const CID: &str = "1be159322466b720589f5a21d0f2bc6ae84d437dd49139fc27efc7d7e8546284";

    /// A containerised brain (`brain-dock`): the docker CLI (pgid
    /// 1917676) sits in the agent's cgroup; EngineCore (1919738) and a helper
    /// (1917796) hold 34 344 + 170 MiB of CUDA in the container's scope.
    fn dock(roots: &Roots) {
        fake_proc(roots, 1917676, 1917676, AGENT, [30 * 1024, 0, 10 * 1024, 0]);
        fake_proc(roots, 1919738, 1919700, &format!("/{CONTAINER}"), [0; 4]);
        fake_proc(roots, 1917796, 1917796, &format!("/{CONTAINER}"), [0; 4]);
        // speech-asr in its own Lumbridge Computer scope: never the container's.
        fake_proc(roots, 2598468, 2598468, &format!("/{SCOPE}"), [0; 4]);
        let dir = roots.cgroup.join(CONTAINER);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("memory.current"), "6561681408\n").unwrap();
        fs::write(
            dir.join("memory.stat"),
            "anon 4335316992\nfile 2047193088\nkernel 166707200\nshmem 122306560\n",
        )
        .unwrap();
        fs::write(dir.join("memory.swap.current"), "1792376832\n").unwrap();
        fs::write(dir.join("cgroup.procs"), "1917700\n1917796\n1919738\n").unwrap();
    }

    const DOCK_GPU: &str = "2598468, 2989\n1917796, 170\n1919738, 34344\n2594397, 4194\n";

    #[test]
    fn gpu_pids_are_attributed_to_a_container_by_their_cgroup() {
        let roots = roots();
        dock(&roots);
        let gpu = parse_compute_apps(DOCK_GPU);
        let mut pids = gpu_pids_in(&roots, &gpu, CONTAINER);
        pids.sort();
        assert_eq!(pids, [1917796, 1919738]);
        // A sibling whose name merely starts with the same path is not inside it.
        assert!(!within(&format!("{CONTAINER}-other"), CONTAINER));
        assert!(within(&format!("{CONTAINER}/child"), CONTAINER));
    }

    #[test]
    fn a_container_model_is_its_container_cgroup_plus_its_gpu_plus_the_cli() {
        let roots = roots();
        dock(&roots);
        let gpu = parse_compute_apps(DOCK_GPU);
        let m = measure_container(&roots, CONTAINER, 1917676, Some(&gpu)).unwrap();
        assert_eq!(m.source, "container");
        assert_eq!(m.cgroup.as_deref(), Some(CONTAINER));
        assert!((m.gpu_gb.unwrap() - 33.705).abs() < 0.01, "{:?}", m.gpu_gb);
        // 6.11 - 1.91 + 0.11 GiB host, plus the CLI's 30 MiB.
        assert!((m.host_gb - 4.348).abs() < 0.01, "{}", m.host_gb);
        assert!((m.swap_gb - 1.669).abs() < 0.01);
        assert_eq!(m.pids, 4);
        assert!((m.demand_gb() - 39.72).abs() < 0.02, "{}", m.demand_gb());
        // What the CLI alone used to report: next to nothing.
        let old = measure(&roots, 1917676, Some(&gpu)).unwrap();
        assert_eq!(old.source, "unscoped");
        assert!(old.demand_gb() < 0.1);
    }

    #[test]
    fn a_docker_record_is_measured_through_its_container_by_id_or_by_legacy_name() {
        use crate::docker::fake::FakeDocker;
        let roots = roots();
        dock(&roots);
        // The fake container's main pid, and the CLI's argv.
        fs::write(
            roots.proc.join("1917676/cmdline"),
            "docker\0run\0--rm\0--init\0--name=brain-dock\0img\0",
        )
        .unwrap();
        let gpu = parse_compute_apps(DOCK_GPU);
        let process = |container_id: Option<&str>| crate::proc::Proc {
            pid: 1917676,
            seq: 1,
            port: Some(8001),
            boot_id: None,
            start_time_ticks: None,
            state: None,
            state_since_ms: None,
            container_id: container_id.map(str::to_string),
        };
        let docker = FakeDocker::default();
        docker.on(
            &format!("inspect -f {{{{.State.Pid}}}} {CID}"),
            Ok("1917796\n"),
        );
        let m = measure_proc(&roots, &process(Some(CID)), Some(&gpu), &docker).unwrap();
        assert_eq!(m.source, "container");
        assert!(m.gpu_gb.unwrap() > 33.0);
        // Legacy record: found by --name.
        docker.on(
            "ps -aq --no-trunc -f name=^brain-dock$",
            Ok(&format!("{CID}\n")),
        );
        let legacy = measure_proc(&roots, &process(None), Some(&gpu), &docker).unwrap();
        assert_eq!(legacy, m);
        // Docker unreachable: a known id still finds the systemd scope ...
        let down = FakeDocker::default();
        down.on("", Err("Cannot connect to the Docker daemon"));
        let by_path = measure_proc(&roots, &process(Some(CID)), Some(&gpu), &down).unwrap();
        assert_eq!(by_path.source, "container");
        // ... a name cannot be resolved, so it is the old CLI-only reading.
        let fallback = measure_proc(&roots, &process(None), Some(&gpu), &down).unwrap();
        assert_eq!(fallback.source, "unscoped");
    }

    #[test]
    fn nvidia_smi_rows_are_summed_per_pid_and_na_rows_skipped() {
        let map = parse_compute_apps("1265786, 29876\n19499, 27072\n7, [N/A]\n\n19499, 1\n");
        assert_eq!(map.get(&1265786), Some(&29876.0));
        assert_eq!(map.get(&19499), Some(&27073.0));
        assert_eq!(map.get(&7), None);
    }

    #[test]
    fn a_missing_gpu_reading_is_unknown_not_zero() {
        let roots = roots();
        fake_proc(&roots, 5, 5, AGENT, [1048576, 0, 0, 0]);
        assert_eq!(measure(&roots, 5, None).unwrap().gpu_gb, None);
        assert_eq!(
            measure(&roots, 5, Some(&HashMap::new())).unwrap().gpu_gb,
            Some(0.0)
        );
    }

    #[test]
    fn suggestions_are_p99_demand_plus_margin_rounded_up() {
        // 100 samples of 30 GB and one 60 GB spike: p99 ignores the spike, peak does not.
        let mut rows = vec![(2.0, Some(26.0), 2.0); 100];
        rows.push((2.0, Some(56.0), 2.0));
        let l = Learned::from_rows(&rows).unwrap();
        assert_eq!(l.samples, 101);
        assert_eq!(l.peak_demand_gb, 60.0);
        assert_eq!(l.p99_demand_gb, 30.0);
        assert_eq!(l.peak_resident_gb, 58.0);
        assert_eq!(l.suggested_footprint_gb(), 33.0);
        // Small models still get a whole GB of margin.
        let small = Learned::from_rows(&[(0.2, Some(2.9), 1.4)]).unwrap();
        assert_eq!(small.suggested_footprint_gb(), 6.0);
    }

    #[test]
    fn samples_without_gpu_are_ignored_once_any_sample_has_gpu() {
        let l = Learned::from_rows(&[(1.0, None, 0.0), (1.0, Some(10.0), 0.0)]).unwrap();
        assert_eq!(l.samples, 1);
        assert!(l.gpu_seen);
        let host_only = Learned::from_rows(&[(1.0, None, 0.5)]).unwrap();
        assert!(!host_only.gpu_seen);
        assert_eq!(Learned::from_rows(&[]), None);
    }

    fn registry(admission: &str) -> Registry {
        Registry::parse(&format!(
            "apiVersion: lumbridge/v1\nadmission: {admission}\nmodels:\n  embed:\n    name: e\n    footprint_gb: 3\n    serve: {{kind: exec, port: 1}}\n  brain:\n    name: b\n    footprint_gb: 40\n    serve: {{kind: exec, port: 2}}\n"
        ))
        .unwrap()
    }

    /// `MIN_VERSION_SAMPLES` samples of `id` under its current definition.
    fn record_current(samples: &Samples, registry: &Registry, now: i64, id: &str, gpu: f64) {
        let version = model_version(&registry.models[id]);
        for i in 0..MIN_VERSION_SAMPLES as i64 {
            samples
                .record_versioned(now - 500 - i, id, Some(&version), &measurement(gpu))
                .unwrap();
        }
    }

    fn measurement(gpu: f64) -> Measurement {
        Measurement {
            source: "scope",
            cgroup: None,
            host_gb: 0.5,
            file_gb: 0.0,
            gpu_gb: Some(gpu),
            swap_gb: 2.0,
            peak_host_gb: None,
            pids: 1,
        }
    }

    #[test]
    fn measured_admission_only_ever_raises_the_declared_number() {
        let root = temp_dir("admission");
        let now = crate::telemetry::unix_time_ms().unwrap();
        let samples = Samples::open(&root).unwrap();
        let measured = registry("measured");
        record_current(&samples, &measured, now, "embed", 8.0);
        record_current(&samples, &measured, now, "brain", 31.0);

        let declared = registry("declared");
        assert_eq!(admission_gb(&root, &declared, "embed"), 3.0);
        // 0.5 host + 8.0 GPU. The 2 GB it has in swap is not RAM and is not budgeted.
        assert!((admission_gb(&root, &measured, "embed") - 8.5).abs() < 1e-9);
        // brain measured 31.5 < 40 declared: the declared number stands.
        assert_eq!(admission_gb(&root, &measured, "brain"), 40.0);
        // Never sampled: declared.
        assert_eq!(admission_gb(&temp_dir("empty"), &measured, "embed"), 3.0);
    }

    #[test]
    fn the_observed_check_asks_for_what_the_model_really_takes() {
        let root = temp_dir("observed");
        let now = crate::telemetry::unix_time_ms().unwrap();
        let samples = Samples::open(&root).unwrap();
        // 0.5 host + GPU, and 2 GB of cold swap that a fresh start does not need.
        let version = model_version(&registry("declared").models["brain"]);
        samples
            .record_versioned(now - 90_000, "brain", Some(&version), &measurement(2.0))
            .unwrap();
        record_current(&samples, &registry("declared"), now, "brain", 30.0);

        for mode in ["declared", "measured"] {
            let reg = registry(mode);
            // Peak resident, below the 40 GB the author declared, swap left out.
            assert!((observed_gb(&root, &reg, "brain") - 30.5).abs() < 1e-9);
            // Never sampled: fall back to the admission figure.
            assert_eq!(observed_gb(&root, &reg, "embed"), 3.0);
            // The declared half is untouched.
            assert_eq!(admission_gb(&root, &reg, "brain"), 40.0);
        }
    }

    #[test]
    fn admission_learns_only_from_the_models_current_definition() {
        let root = temp_dir("versions");
        let now = crate::telemetry::unix_time_ms().unwrap();
        let samples = Samples::open(&root).unwrap();
        let reg = registry("measured");
        // The id `embed` used to be a 20 GB model; its samples carry another version.
        for i in 0..MIN_VERSION_SAMPLES as i64 {
            samples
                .record_versioned(
                    now - 100_000 - i,
                    "embed",
                    Some("0000000000000000"),
                    &measurement(20.0),
                )
                .unwrap();
            // And some rows predate versions altogether.
            samples
                .record(now - 200_000 - i, "embed", &measurement(15.0))
                .unwrap();
        }
        // Neither counts for today's definition: declared stands.
        assert_eq!(admission_gb(&root, &reg, "embed"), 3.0);
        assert_eq!(observed_gb(&root, &reg, "embed"), 3.0);

        // A few samples of the current one, taken while it was still loading, are
        // not trusted yet...
        let version = model_version(&reg.models["embed"]);
        for i in 0..(MIN_VERSION_SAMPLES as i64 - 1) {
            samples
                .record_versioned(now - 1000 - i, "embed", Some(&version), &measurement(0.1))
                .unwrap();
        }
        assert_eq!(observed_gb(&root, &reg, "embed"), 3.0);
        // ...and once there are enough, they are what admission uses.
        record_current(&samples, &reg, now, "embed", 8.0);
        assert!((admission_gb(&root, &reg, "embed") - 8.5).abs() < 1e-9);
        assert!((observed_gb(&root, &reg, "embed") - 8.5).abs() < 1e-9);
        // `footprints` still reports everything recorded under the id.
        assert_eq!(
            samples
                .learned("embed", 0)
                .unwrap()
                .unwrap()
                .peak_resident_gb,
            20.5
        );
    }

    #[test]
    fn a_version_follows_the_launch_definition_and_not_the_typed_footprint() {
        let reg = registry("declared");
        let embed = &reg.models["embed"];
        assert_eq!(model_version(embed), model_version(embed));
        assert_ne!(model_version(embed), model_version(&reg.models["brain"]));
        let mut retyped = embed.clone();
        retyped.footprint_gb = 8.0;
        assert_eq!(model_version(embed), model_version(&retyped));
        let mut relaunched = embed.clone();
        relaunched.serve.port = Some(9);
        assert_ne!(model_version(embed), model_version(&relaunched));
    }

    #[test]
    fn a_database_from_before_versions_gains_the_column_and_keeps_its_rows() {
        let root = temp_dir("migrate");
        let path = crate::telemetry::db_path(&root);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE footprint_samples (
               timestamp_ms INTEGER NOT NULL, model TEXT NOT NULL, source TEXT NOT NULL,
               host_gb REAL NOT NULL, file_gb REAL NOT NULL, gpu_gb REAL, swap_gb REAL NOT NULL,
               peak_host_gb REAL, PRIMARY KEY (timestamp_ms, model));
             INSERT INTO footprint_samples VALUES (1, 'embed', 'scope', 0.5, 0.0, 8.0, 2.0, NULL);",
        )
        .unwrap();
        drop(conn);
        let samples = Samples::open(&root).unwrap();
        let reopened = Samples::open(&root).unwrap();
        assert_eq!(reopened.learned("embed", 0).unwrap().unwrap().samples, 1);
        samples
            .record_versioned(2, "embed", Some("v"), &measurement(8.0))
            .unwrap();
        assert_eq!(samples.learned("embed", 0).unwrap().unwrap().samples, 2);
    }

    #[test]
    fn old_samples_are_pruned_and_outside_the_window_they_do_not_count() {
        let root = temp_dir("prune");
        let now = crate::telemetry::unix_time_ms().unwrap();
        let samples = Samples::open(&root).unwrap();
        samples
            .record(now - RETENTION_MS - 1, "embed", &measurement(50.0))
            .unwrap();
        samples.record(now, "embed", &measurement(8.0)).unwrap();
        assert_eq!(
            samples.learned("embed", 0).unwrap().unwrap().peak_demand_gb,
            52.5
        );
        samples.prune(now).unwrap();
        let l = samples.learned("embed", 0).unwrap().unwrap();
        assert_eq!((l.samples, l.peak_demand_gb), (1, 10.5));
        assert_eq!(samples.models(now + 1).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn overage_is_demand_above_the_declared_footprint_and_never_negative() {
        let reg = registry("declared");
        let measured = BTreeMap::from([
            ("embed".to_string(), measurement(8.0)),
            ("brain".to_string(), measurement(30.0)),
        ]);
        let over = overage_gb(&reg, &measured);
        assert!((over["embed"] - 7.5).abs() < 1e-9);
        assert_eq!(over["brain"], 0.0);
    }
}
