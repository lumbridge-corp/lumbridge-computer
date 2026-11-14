//! Privacy-safe, durable model-serving telemetry.
//!
//! The model runtime remains the authority for scheduling, token, latency, and
//! multimodal encoder metrics. Lumbridge Computer samples each running model's Prometheus
//! endpoint and stores only numeric measurements plus bounded labels. Prompts,
//! outputs, image data, URLs, and arbitrary HTTP headers never enter this
//! database. Which metric means what, per engine, is `engines`' job.

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::config::{find_scene, Model, Registry};
use crate::engines::{self, http::parse_http_url, prom, Engine};
use crate::proc::State;

const DB_RELATIVE_PATH: &str = ".compute/usage/telemetry.sqlite3";
const COUNTER_SNAPSHOT_INTERVAL_MS: i64 = 60_000;
/// While a target keeps failing the same way, say so at most this often.
const FAILURE_REMINDER: Duration = Duration::from_secs(60 * 60);
const ALLOWED_CUSTOM_LABELS: &[&str] = &["client", "agent", "workload"];

const COUNTERS: &[&str] = &[
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:cached_tokens_total",
    "sglang:num_requests_total",
    "sglang:num_aborted_requests_total",
    "sglang:encoder_requests_received_total",
    "sglang:encoder_cache_total_tokens_total",
    "sglang:encoder_cache_hit_tokens_total",
    "sglang:encoder_mm_items_per_request_sum",
    "sglang:encoder_mm_items_per_request_count",
    "sglang:time_to_first_token_seconds_sum",
    "sglang:time_to_first_token_seconds_count",
    "sglang:e2e_request_latency_seconds_sum",
    "sglang:e2e_request_latency_seconds_count",
];

type Metric = prom::Sample;

#[derive(Debug, Clone)]
pub struct Target {
    pub model: String,
    pub scene: String,
    pub max_concurrency: u32,
    pub config_fingerprint: String,
    /// From the registry; `Generic` is resolved from the scraped metric names.
    pub engine: Engine,
    pub port: Option<u16>,
}

#[derive(Debug, Serialize)]
pub struct CollectResult {
    pub timestamp_ms: i64,
    pub model: String,
    pub scene: String,
    pub engine: Engine,
    pub max_concurrency: u32,
    pub metrics_recorded: usize,
    pub running: u32,
    pub queued: u32,
    pub kv_usage: Option<f64>,
}

/// One collection pass: what was recorded, and which targets could not be.
#[derive(Debug, Serialize, Default)]
pub struct CollectReport {
    pub results: Vec<CollectResult>,
    pub failures: Vec<CollectFailure>,
}

#[derive(Debug, Serialize)]
pub struct CollectFailure {
    pub model: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct UsageTotals {
    pub requests: f64,
    pub aborted_requests: f64,
    pub prompt_tokens: f64,
    pub generation_tokens: f64,
    pub cached_tokens: f64,
    pub vision_requests: f64,
    pub vision_request_share: f64,
    pub vision_request_source: String,
    pub vision_prompt_tokens: f64,
    pub vision_items: f64,
    pub vision_encoder_tokens: f64,
    pub vision_encoder_cached_tokens: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConcurrencyBucket {
    pub concurrency: String,
    pub samples: u64,
    pub share: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConcurrencySummary {
    pub samples: u64,
    pub queued_samples: u64,
    pub peak_running: u32,
    pub peak_queued: u32,
    /// A time-sampled distribution, not an admission-count distribution.
    pub distribution: Vec<ConcurrencyBucket>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelConcurrency {
    pub model: String,
    pub engine: Option<String>,
    pub port: Option<u16>,
    pub peak_kv_usage: Option<f64>,
    pub concurrency: ConcurrencySummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentUsage {
    pub client: String,
    pub agent: String,
    pub workload: String,
    pub requests: f64,
    pub prompt_tokens: f64,
    pub generation_tokens: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct UsageSummary {
    pub since_ms: i64,
    pub until_ms: i64,
    pub database: String,
    pub totals: UsageTotals,
    /// The C0-C4 distribution of the busiest serving port in the window (usually
    /// the brain's port, whichever brain variant held it). Mixing
    /// idle embed/rerank samples in would bury the brain's distribution in C0.
    pub concurrency: ConcurrencySummary,
    /// What `concurrency` covers, e.g. `port 8001`; `model <id>` for samples
    /// recorded before ports were stored.
    pub concurrency_scope: Option<String>,
    /// Every model's own distribution.
    pub models: Vec<ModelConcurrency>,
    pub agents: Vec<AgentUsage>,
}

/// The telemetry database, shared with `footprint` (which adds its own table).
pub(crate) fn db_path(root: &Path) -> PathBuf {
    root.join(DB_RELATIVE_PATH)
}

pub struct Store {
    path: PathBuf,
    conn: Connection,
}

impl Store {
    pub fn open(root: &Path) -> Result<Self> {
        let path = root.join(DB_RELATIVE_PATH);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating telemetry directory {}", parent.display()))?;
        }
        let conn = Connection::open(&path)
            .with_context(|| format!("opening telemetry database {}", path.display()))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS telemetry_schema (
               version INTEGER PRIMARY KEY,
               applied_at_ms INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS config_snapshots (
               fingerprint TEXT PRIMARY KEY,
               first_seen_ms INTEGER NOT NULL,
               model TEXT NOT NULL,
               scene TEXT NOT NULL,
               max_concurrency INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS engine_samples (
               timestamp_ms INTEGER NOT NULL,
               model TEXT NOT NULL,
               scene TEXT NOT NULL,
               config_fingerprint TEXT NOT NULL,
               max_concurrency INTEGER NOT NULL,
               running REAL NOT NULL,
               queued REAL NOT NULL,
               generation_tps REAL,
               cache_hit_rate REAL,
               PRIMARY KEY (timestamp_ms, model)
             );
             CREATE INDEX IF NOT EXISTS engine_samples_time
               ON engine_samples(timestamp_ms);
             CREATE TABLE IF NOT EXISTS counter_samples (
               timestamp_ms INTEGER NOT NULL,
               model TEXT NOT NULL,
               metric TEXT NOT NULL,
               labels_json TEXT NOT NULL,
               value REAL NOT NULL,
               PRIMARY KEY (timestamp_ms, model, metric, labels_json)
             );
             CREATE INDEX IF NOT EXISTS counter_samples_metric_time
               ON counter_samples(metric, timestamp_ms);
             INSERT OR IGNORE INTO telemetry_schema(version, applied_at_ms)
               VALUES (1, CAST(strftime('%s','now') AS INTEGER) * 1000);",
        )?;
        migrate_v2(&conn)?;
        Ok(Self { path, conn })
    }

    fn record(&mut self, target: &Target, metrics: &[Metric]) -> Result<CollectResult> {
        self.record_with_counter_interval(target, metrics, COUNTER_SNAPSHOT_INTERVAL_MS)
    }

    fn record_with_counter_interval(
        &mut self,
        target: &Target,
        metrics: &[Metric],
        counter_interval_ms: i64,
    ) -> Result<CollectResult> {
        let timestamp_ms = unix_time_ms()?;
        let engine = target.engine.resolve(metrics);
        // Unknown load is not recorded as an idle sample.
        let load = engines::load_from_samples(engine, metrics)
            .with_context(|| format!("no {engine} load metrics for '{}'", target.model))?;
        let counters = normalize_runtime_metrics(engine, metrics);
        // SGLang-only gauges; absent elsewhere and stored as NULL.
        let generation_tps = default_gauge(metrics, "sglang:gen_throughput");
        let cache_hit_rate = default_gauge(metrics, "sglang:cache_hit_rate");

        let last_counter_ms: Option<i64> = self.conn.query_row(
            "SELECT MAX(timestamp_ms) FROM counter_samples WHERE model = ?1",
            params![target.model],
            |row| row.get(0),
        )?;
        let counter_due =
            last_counter_ms.is_none_or(|previous| timestamp_ms - previous >= counter_interval_ms);

        let transaction = self.conn.transaction()?;
        transaction.execute(
            "INSERT OR IGNORE INTO config_snapshots
             (fingerprint, first_seen_ms, model, scene, max_concurrency)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                target.config_fingerprint,
                timestamp_ms,
                target.model,
                target.scene,
                target.max_concurrency
            ],
        )?;
        transaction.execute(
            "INSERT INTO engine_samples
             (timestamp_ms, model, scene, config_fingerprint, max_concurrency,
              running, queued, generation_tps, cache_hit_rate, kv_usage, engine, port)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                timestamp_ms,
                target.model,
                target.scene,
                target.config_fingerprint,
                target.max_concurrency,
                f64::from(load.running),
                f64::from(load.waiting),
                generation_tps,
                cache_hit_rate,
                load.kv_usage,
                engine.as_str(),
                target.port
            ],
        )?;

        let mut recorded = 0usize;
        if counter_due {
            for metric in counters.iter().filter(|metric| {
                COUNTERS.contains(&metric.name.as_str()) && metric.value.is_finite()
            }) {
                let labels = retained_labels(&metric.labels);
                transaction.execute(
                    "INSERT INTO counter_samples
                     (timestamp_ms, model, metric, labels_json, value)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        timestamp_ms,
                        target.model,
                        metric.name,
                        serde_json::to_string(&labels)?,
                        metric.value
                    ],
                )?;
                recorded += 1;
            }
        }
        transaction.commit()?;

        Ok(CollectResult {
            timestamp_ms,
            model: target.model.clone(),
            scene: target.scene.clone(),
            engine,
            max_concurrency: target.max_concurrency,
            metrics_recorded: recorded,
            running: load.running,
            queued: load.waiting,
            kv_usage: load.kv_usage,
        })
    }

    pub fn summary(&self, since_ms: i64, until_ms: i64) -> Result<UsageSummary> {
        let image = |labels: &BTreeMap<String, String>| {
            labels.get("modality").is_some_and(|value| value == "image")
        };
        let vision_workload = |labels: &BTreeMap<String, String>| {
            labels
                .get("workload")
                .is_some_and(|value| value == "vision")
        };
        let has_encoder_request_metrics =
            self.has_metric_samples("sglang:encoder_requests_received_total", since_ms, until_ms)?;
        let vision_requests = if has_encoder_request_metrics {
            self.metric_delta(
                "sglang:encoder_requests_received_total",
                since_ms,
                until_ms,
                image,
            )?
        } else {
            self.metric_delta(
                "sglang:num_requests_total",
                since_ms,
                until_ms,
                vision_workload,
            )?
        };
        let mut totals = UsageTotals {
            requests: self
                .metric_delta("sglang:num_requests_total", since_ms, until_ms, |_| true)?,
            aborted_requests: self.metric_delta(
                "sglang:num_aborted_requests_total",
                since_ms,
                until_ms,
                |_| true,
            )?,
            prompt_tokens: self.metric_delta(
                "sglang:prompt_tokens_total",
                since_ms,
                until_ms,
                |_| true,
            )?,
            generation_tokens: self.metric_delta(
                "sglang:generation_tokens_total",
                since_ms,
                until_ms,
                |_| true,
            )?,
            cached_tokens: self.metric_delta(
                "sglang:cached_tokens_total",
                since_ms,
                until_ms,
                |_| true,
            )?,
            vision_requests,
            vision_request_share: 0.0,
            vision_request_source: if has_encoder_request_metrics {
                "encoder_metrics".to_string()
            } else {
                "workload_label".to_string()
            },
            vision_prompt_tokens: self.metric_delta(
                "sglang:prompt_tokens_total",
                since_ms,
                until_ms,
                vision_workload,
            )?,
            vision_items: self.metric_delta(
                "sglang:encoder_mm_items_per_request_sum",
                since_ms,
                until_ms,
                image,
            )?,
            vision_encoder_tokens: self.metric_delta(
                "sglang:encoder_cache_total_tokens_total",
                since_ms,
                until_ms,
                image,
            )?,
            vision_encoder_cached_tokens: self.metric_delta(
                "sglang:encoder_cache_hit_tokens_total",
                since_ms,
                until_ms,
                image,
            )?,
        };
        totals.vision_request_share = if totals.requests > 0.0 {
            (totals.vision_requests / totals.requests).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let (concurrency, concurrency_scope, models) = self.concurrency(since_ms, until_ms)?;
        Ok(UsageSummary {
            since_ms,
            until_ms,
            database: self.path.display().to_string(),
            totals,
            concurrency,
            concurrency_scope,
            models,
            agents: self.agent_usage(since_ms, until_ms)?,
        })
    }

    fn metric_delta<F>(&self, metric: &str, since_ms: i64, until_ms: i64, keep: F) -> Result<f64>
    where
        F: Fn(&BTreeMap<String, String>) -> bool,
    {
        let total: f64 = self
            .series_deltas(metric, since_ms, until_ms)?
            .into_iter()
            .filter(|(labels, _)| keep(labels))
            .map(|(_, delta)| delta)
            .sum();
        Ok(if total == 0.0 { 0.0 } else { total })
    }

    fn has_metric_samples(&self, metric: &str, since_ms: i64, until_ms: i64) -> Result<bool> {
        self.conn
            .query_row(
                "SELECT EXISTS(
                   SELECT 1 FROM counter_samples
                   WHERE metric = ?1 AND timestamp_ms >= ?2 AND timestamp_ms <= ?3
                 )",
                params![metric, since_ms, until_ms],
                |row| row.get(0),
            )
            .context("checking telemetry metric availability")
    }

    fn series_deltas(
        &self,
        metric: &str,
        since_ms: i64,
        until_ms: i64,
    ) -> Result<Vec<(BTreeMap<String, String>, f64)>> {
        let mut baseline_statement = self.conn.prepare(
            "SELECT model, labels_json, MIN(timestamp_ms)
             FROM counter_samples
             WHERE metric = ?1 AND timestamp_ms <= ?2
             GROUP BY model, labels_json",
        )?;
        let baseline_rows = baseline_statement.query_map(params![metric, until_ms], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;
        let mut model_baselines: HashMap<Vec<String>, i64> = HashMap::new();
        for row in baseline_rows {
            let (model, labels_json, timestamp_ms) = row?;
            let scope = counter_scope(&model, &labels_json)?;
            let baseline = model_baselines.entry(scope).or_insert(timestamp_ms);
            *baseline = (*baseline).min(timestamp_ms);
        }

        let mut statement = self.conn.prepare(
            "SELECT model, timestamp_ms, labels_json, value
             FROM counter_samples
             WHERE metric = ?1 AND timestamp_ms <= ?2
             ORDER BY model, labels_json, timestamp_ms",
        )?;
        let rows = statement.query_map(params![metric, until_ms], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
            ))
        })?;

        let mut state: HashMap<(String, String), (Option<f64>, f64)> = HashMap::new();
        for row in rows {
            let (model, timestamp_ms, labels_json, value) = row?;
            let model_baseline = model_baselines
                .get(&counter_scope(&model, &labels_json)?)
                .copied()
                .unwrap_or(timestamp_ms);
            let entry = state.entry((model, labels_json)).or_insert((None, 0.0));
            if timestamp_ms >= since_ms {
                if let Some(previous) = entry.0 {
                    entry.1 += if value >= previous {
                        value - previous
                    } else {
                        value
                    };
                } else if timestamp_ms > model_baseline {
                    // Prometheus creates a new labeled series on first use. The
                    // model-wide baseline predates it, so its initial value is
                    // usage in this window rather than unknown prehistory.
                    entry.1 += value;
                }
            }
            entry.0 = Some(value);
        }

        state
            .into_iter()
            .map(|((_model, labels), (_previous, delta))| {
                let labels = serde_json::from_str(&labels)
                    .context("parsing labels stored in telemetry database")?;
                Ok((labels, delta))
            })
            .collect()
    }

    fn concurrency(
        &self,
        since_ms: i64,
        until_ms: i64,
    ) -> Result<(ConcurrencySummary, Option<String>, Vec<ModelConcurrency>)> {
        struct PerModel {
            engine: Option<String>,
            port: Option<u16>,
            peak_kv_usage: Option<f64>,
            rows: Vec<(f64, f64)>,
        }
        let mut statement = self.conn.prepare(
            "SELECT model, port, engine, running, queued, kv_usage FROM engine_samples
             WHERE timestamp_ms >= ?1 AND timestamp_ms <= ?2
             ORDER BY timestamp_ms",
        )?;
        let rows = statement.query_map(params![since_ms, until_ms], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<u16>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, f64>(4)?,
                row.get::<_, Option<f64>>(5)?,
            ))
        })?;
        let mut models: BTreeMap<String, PerModel> = BTreeMap::new();
        let mut groups: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        for row in rows {
            let (model, port, engine, running, queued, kv_usage) = row?;
            let group = match port {
                Some(port) => format!("port {port}"),
                None => format!("model {model}"),
            };
            groups.entry(group).or_default().push((running, queued));
            let entry = models.entry(model).or_insert(PerModel {
                engine: None,
                port: None,
                peak_kv_usage: None,
                rows: Vec::new(),
            });
            entry.engine = engine.or(entry.engine.take());
            entry.port = port.or(entry.port);
            entry.peak_kv_usage = match (entry.peak_kv_usage, kv_usage) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            entry.rows.push((running, queued));
        }
        // Busiest = most running requests summed over its samples; ties go to
        // the group with more samples, then the first by name, so the choice
        // is deterministic.
        let busiest = groups
            .iter()
            .map(|(name, rows)| {
                let load: f64 = rows.iter().map(|(running, _)| running.max(0.0)).sum();
                (name, rows, load)
            })
            .reduce(|best, next| {
                if next.2 > best.2 || (next.2 == best.2 && next.1.len() > best.1.len()) {
                    next
                } else {
                    best
                }
            });
        let (overall, scope) = match busiest {
            Some((name, rows, _)) => (concurrency_distribution(rows), Some(name.clone())),
            None => (concurrency_distribution(&[]), None),
        };
        let per_model = models
            .into_iter()
            .map(|(model, entry)| ModelConcurrency {
                model,
                engine: entry.engine,
                port: entry.port,
                peak_kv_usage: entry.peak_kv_usage,
                concurrency: concurrency_distribution(&entry.rows),
            })
            .collect();
        Ok((overall, scope, per_model))
    }

    fn agent_usage(&self, since_ms: i64, until_ms: i64) -> Result<Vec<AgentUsage>> {
        #[derive(Default)]
        struct Row {
            requests: f64,
            prompt_tokens: f64,
            generation_tokens: f64,
        }
        let mut grouped: BTreeMap<(String, String, String), Row> = BTreeMap::new();
        for (metric, field) in [
            ("sglang:num_requests_total", 0),
            ("sglang:prompt_tokens_total", 1),
            ("sglang:generation_tokens_total", 2),
        ] {
            for (labels, delta) in self.series_deltas(metric, since_ms, until_ms)? {
                let key = (
                    label_or_unknown(&labels, "client"),
                    label_or_unknown(&labels, "agent"),
                    label_or_unknown(&labels, "workload"),
                );
                let row = grouped.entry(key).or_default();
                match field {
                    0 => row.requests += delta,
                    1 => row.prompt_tokens += delta,
                    _ => row.generation_tokens += delta,
                }
            }
        }
        let mut rows: Vec<_> = grouped
            .into_iter()
            .filter(|(_, row)| {
                row.requests != 0.0 || row.prompt_tokens != 0.0 || row.generation_tokens != 0.0
            })
            .map(|((client, agent, workload), row)| AgentUsage {
                client,
                agent,
                workload,
                requests: row.requests,
                prompt_tokens: row.prompt_tokens,
                generation_tokens: row.generation_tokens,
            })
            .collect();
        rows.sort_by(|a, b| b.requests.total_cmp(&a.requests));
        Ok(rows)
    }
}

/// Schema v2 (2026-11-08) adds, never drops: `engine_samples.kv_usage`
/// (0..1 fraction of KV cache in use), `engine` (which normaliser read the
/// sample) and `port` (so a report can follow "the brain on :8001" across
/// model swaps). Rows written before v2 keep NULL in all three.
fn migrate_v2(conn: &Connection) -> Result<()> {
    let mut statement = conn.prepare("SELECT name FROM pragma_table_info('engine_samples')")?;
    let existing: Vec<String> = statement
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for (column, kind) in [
        ("kv_usage", "REAL"),
        ("engine", "TEXT"),
        ("port", "INTEGER"),
    ] {
        if !existing.iter().any(|name| name == column) {
            conn.execute_batch(&format!(
                "ALTER TABLE engine_samples ADD COLUMN {column} {kind};"
            ))
            .with_context(|| format!("adding engine_samples.{column}"))?;
        }
    }
    conn.execute(
        "INSERT OR IGNORE INTO telemetry_schema(version, applied_at_ms)
         VALUES (2, CAST(strftime('%s','now') AS INTEGER) * 1000)",
        [],
    )?;
    Ok(())
}

/// Time-sampled C0-C4 distribution of `(running, queued)` samples.
fn concurrency_distribution(rows: &[(f64, f64)]) -> ConcurrencySummary {
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut samples = 0u64;
    let mut queued_samples = 0u64;
    let mut peak_running = 0u32;
    let mut peak_queued = 0u32;
    for &(running, queued) in rows {
        let running = running.max(0.0).round() as u32;
        let queued = queued.max(0.0).round() as u32;
        let bucket = if running >= 4 {
            "C4+".to_string()
        } else {
            format!("C{running}")
        };
        *counts.entry(bucket).or_default() += 1;
        samples += 1;
        queued_samples += u64::from(queued > 0);
        peak_running = peak_running.max(running);
        peak_queued = peak_queued.max(queued);
    }
    let distribution = counts
        .into_iter()
        .map(|(concurrency, count)| ConcurrencyBucket {
            concurrency,
            samples: count,
            share: if samples == 0 {
                0.0
            } else {
                count as f64 / samples as f64
            },
        })
        .collect();
    ConcurrencySummary {
        samples,
        queued_samples,
        peak_running,
        peak_queued,
        distribution,
    }
}

/// Scrape and record once. With `metrics_url`, scrape only that endpoint
/// (the pre-`engines` behaviour, kept as an override); otherwise scrape every
/// running model with a known engine in the active Scene.
pub fn collect_once(
    root: &Path,
    metrics_url: Option<&str>,
    model_hint: &str,
    timeout: Duration,
) -> Result<CollectReport> {
    let mut store = Store::open(root)?;
    let outcomes = collect_pass(root, &mut store, metrics_url, model_hint, timeout)?;
    let mut report = CollectReport::default();
    for (model, outcome) in outcomes {
        match outcome {
            Ok(result) => report.results.push(result),
            Err(error) => report.failures.push(CollectFailure {
                model,
                error: format!("{error:#}"),
            }),
        }
    }
    if report.results.is_empty() {
        match report.failures.as_slice() {
            [] => bail!("no running model with a known engine to scrape"),
            [only] => bail!("{}: {}", only.model, only.error),
            _ => {}
        }
    }
    Ok(report)
}

type Outcome = (String, Result<CollectResult>);

fn collect_pass(
    root: &Path,
    store: &mut Store,
    metrics_url: Option<&str>,
    model_hint: &str,
    timeout: Duration,
) -> Result<Vec<Outcome>> {
    if let Some(url) = metrics_url {
        let target = resolve_target(root, url, model_hint)?;
        let outcome = engines::fetch_url(url, timeout)
            .and_then(|body| store.record(&target, &prom::parse(&body)));
        return Ok(vec![(target.model, outcome)]);
    }
    let mut outcomes = Vec::new();
    for (target, model) in scrape_targets(root)? {
        if model_hint != "auto" && target.model != model_hint {
            continue;
        }
        let outcome = engines::fetch_metrics(&model, timeout)
            .and_then(|body| store.record(&target, &prom::parse(&body)));
        outcomes.push((target.model, outcome));
    }
    Ok(outcomes)
}

/// The models worth scraping now: those in the active (else desired) Scene,
/// or the whole registry when no Scene is known; with an engine `engines`
/// can read; and actually serving. Mutually exclusive models may share a port
/// (every brain variant uses :8001), so only one model per port is kept, the
/// one whose health marker answers when several could be there.
fn scrape_targets(root: &Path) -> Result<Vec<(Target, Model)>> {
    let registry = Registry::load(root)?;
    let state = State::load_checked(root)?;
    let scene_name = state.active_scene.or(state.desired_scene);
    let ids: Vec<String> = match &scene_name {
        Some(name) => find_scene(root, name)?.models,
        None => registry.models.keys().cloned().collect(),
    };
    let scene = scene_name.unwrap_or_else(|| "unknown".to_string());
    let mut by_port: BTreeMap<u16, Vec<(String, Model)>> = BTreeMap::new();
    for id in ids {
        let Some(model) = registry.models.get(&id) else {
            continue;
        };
        let Some(port) = model.serve.port else {
            continue;
        };
        if engines::detect(model) == Engine::Generic {
            continue;
        }
        by_port.entry(port).or_default().push((id, model.clone()));
    }
    let mut targets = Vec::new();
    for (port, candidates) in by_port {
        let chosen = if candidates.len() == 1 {
            candidates
                .into_iter()
                .find(|_| crate::governor::port_open(port))
        } else {
            candidates
                .into_iter()
                .find(|(_, model)| crate::governor::is_running(model))
        };
        if let Some((id, model)) = chosen {
            targets.push((target_for(&id, &model, &scene), model));
        }
    }
    Ok(targets)
}

fn target_for(id: &str, model: &Model, scene: &str) -> Target {
    let max_concurrency = engines::configured_concurrency(model);
    // Unchanged input, so fingerprints stay stable across this upgrade.
    let fingerprint_input = format!(
        "{id}\n{scene}\n{max_concurrency}\n{:?}\n{:?}",
        model.serve.command, model.serve.args
    );
    Target {
        model: id.to_string(),
        scene: scene.to_string(),
        max_concurrency,
        config_fingerprint: format!("fnv1a64:{:016x}", fnv1a64(fingerprint_input.as_bytes())),
        engine: engines::detect(model),
        port: model.serve.port,
    }
}

/// The resident collector. Failures degrade telemetry, never the agent, and
/// are logged on change: once when a target starts failing, once when it
/// recovers, and at most hourly while it keeps failing (the old collector
/// logged every 60th failure, ~20 lines an hour per dead endpoint).
pub fn run_collector(
    root: PathBuf,
    metrics_url: Option<String>,
    model_hint: String,
    interval: Duration,
    timeout: Duration,
) {
    let mut log = ScrapeLog::default();
    loop {
        let pass = Store::open(&root).and_then(|mut store| {
            collect_pass(
                &root,
                &mut store,
                metrics_url.as_deref(),
                &model_hint,
                timeout,
            )
        });
        let now = Instant::now();
        let messages = match pass {
            Ok(outcomes) => {
                let observed: Vec<(String, Result<String, String>)> = outcomes
                    .into_iter()
                    .map(|(model, outcome)| {
                        let outcome = outcome
                            .map(|result| format!("{}, scene {}", result.engine, result.scene))
                            .map_err(|error| format!("{error:#}"));
                        (model, outcome)
                    })
                    .collect();
                log.observe(now, &observed, true)
            }
            Err(error) => log.observe(
                now,
                &[("collector".to_string(), Err(format!("{error:#}")))],
                false,
            ),
        };
        for message in messages {
            eprintln!("telemetry: {message}");
        }
        thread::sleep(interval.max(Duration::from_secs(1)));
    }
}

/// Per-target scrape state, so the collector logs transitions, not ticks.
#[derive(Debug, Default)]
struct ScrapeLog {
    targets: BTreeMap<String, TargetLog>,
}

#[derive(Debug)]
struct TargetLog {
    failures: u64,
    last_logged: Instant,
}

impl ScrapeLog {
    /// Fold one pass into the state and return the lines worth printing.
    /// `complete` is false when the pass itself failed (registry unreadable):
    /// targets missing from it have not stopped, we just could not look.
    fn observe(
        &mut self,
        now: Instant,
        outcomes: &[(String, Result<String, String>)],
        complete: bool,
    ) -> Vec<String> {
        let mut messages = Vec::new();
        for (key, outcome) in outcomes {
            let previous = self.targets.get_mut(key);
            match (previous, outcome) {
                (None, Ok(description)) => {
                    messages.push(format!("scraping {key} ({description})"));
                    self.targets.insert(
                        key.clone(),
                        TargetLog {
                            failures: 0,
                            last_logged: now,
                        },
                    );
                }
                (None, Err(error)) => {
                    messages.push(format!("{key}: scrape failed: {error}"));
                    self.targets.insert(
                        key.clone(),
                        TargetLog {
                            failures: 1,
                            last_logged: now,
                        },
                    );
                }
                (Some(state), Ok(_)) => {
                    if state.failures > 0 {
                        messages.push(format!(
                            "{key}: recovered after {} failed scrape(s)",
                            state.failures
                        ));
                        state.failures = 0;
                        state.last_logged = now;
                    }
                }
                (Some(state), Err(error)) => {
                    state.failures += 1;
                    if state.failures == 1 {
                        messages.push(format!("{key}: scrape failed: {error}"));
                        state.last_logged = now;
                    } else if now.duration_since(state.last_logged) >= FAILURE_REMINDER {
                        messages.push(format!(
                            "{key}: still failing ({} consecutive): {error}",
                            state.failures
                        ));
                        state.last_logged = now;
                    }
                }
            }
        }
        if complete {
            if self
                .targets
                .remove("collector")
                .is_some_and(|s| s.failures > 0)
            {
                messages.push("collector: recovered".to_string());
            }
            let seen: Vec<&String> = outcomes.iter().map(|(key, _)| key).collect();
            let gone: Vec<String> = self
                .targets
                .keys()
                .filter(|key| !seen.contains(key))
                .cloned()
                .collect();
            for key in gone {
                self.targets.remove(&key);
                messages.push(format!("{key}: not serving; stopped scraping"));
            }
        }
        messages
    }
}

pub fn report(root: &Path, since: &str) -> Result<UsageSummary> {
    let duration = parse_duration(since)?;
    let until_ms = unix_time_ms()?;
    let since_ms = until_ms.saturating_sub(duration.as_millis().min(i64::MAX as u128) as i64);
    Store::open(root)?.summary(since_ms, until_ms)
}

pub fn render_text(summary: &UsageSummary) -> String {
    let totals = &summary.totals;
    let mut lines = vec![
        "Lumbridge Computer · usage".to_string(),
        format!("  requests     {:>12.0}", totals.requests),
        format!("  prompt tok   {:>12.0}", totals.prompt_tokens),
        format!("  output tok   {:>12.0}", totals.generation_tokens),
        format!("  cached tok   {:>12.0}", totals.cached_tokens),
        format!("  aborted      {:>12.0}", totals.aborted_requests),
        format!("  vision calls {:>12.0}", totals.vision_requests),
        format!(
            "  vision share {:>11.1}%",
            totals.vision_request_share * 100.0
        ),
        format!("  vision tok   {:>12.0}", totals.vision_prompt_tokens),
        format!("  image items  {:>12.0}", totals.vision_items),
        String::new(),
        match &summary.concurrency_scope {
            Some(scope) => format!("  time-sampled concurrency ({scope}):"),
            None => "  time-sampled concurrency:".to_string(),
        },
    ];
    if summary.concurrency.samples == 0 {
        lines.push("    (no samples in this window)".to_string());
    } else {
        for bucket in &summary.concurrency.distribution {
            lines.push(format!(
                "    {:>3}  {:>8} samples  {:>6.1}%",
                bucket.concurrency,
                bucket.samples,
                bucket.share * 100.0
            ));
        }
        lines.push(format!(
            "    peak running {} · peak queued {} · queue observed in {} sample(s)",
            summary.concurrency.peak_running,
            summary.concurrency.peak_queued,
            summary.concurrency.queued_samples
        ));
    }
    if summary.models.len() > 1 {
        lines.push(String::new());
        lines.push("  models:".to_string());
        for row in &summary.models {
            let kv = row
                .peak_kv_usage
                .map(|kv| format!(" · peak KV {:.0}%", kv * 100.0))
                .unwrap_or_default();
            lines.push(format!(
                "    {} ({}){}  {} samples · peak running {} · peak queued {}{kv}",
                row.model,
                row.engine.as_deref().unwrap_or("unknown engine"),
                row.port.map(|p| format!(" :{p}")).unwrap_or_default(),
                row.concurrency.samples,
                row.concurrency.peak_running,
                row.concurrency.peak_queued,
            ));
        }
    }
    if !summary.agents.is_empty() {
        lines.push(String::new());
        lines.push("  agents:".to_string());
        for row in &summary.agents {
            lines.push(format!(
                "    {}/{}/{}  {:.0} req · {:.0} in · {:.0} out tok",
                row.client,
                row.agent,
                row.workload,
                row.requests,
                row.prompt_tokens,
                row.generation_tokens
            ));
        }
    }
    lines.join("\n")
}

fn resolve_target(root: &Path, metrics_url: &str, model_hint: &str) -> Result<Target> {
    let registry = Registry::load(root)?;
    let state = State::load_checked(root)?;
    let scene_name = state
        .active_scene
        .or(state.desired_scene)
        .unwrap_or_else(|| "unknown".to_string());
    let port = parse_http_url(metrics_url)?.1;
    let model = if model_hint != "auto" {
        model_hint.to_string()
    } else if scene_name != "unknown" {
        find_scene(root, &scene_name)?
            .models
            .into_iter()
            .find(|id| registry.models.get(id).and_then(|m| m.serve.port) == Some(port))
            .with_context(|| {
                format!("active Scene '{scene_name}' has no model on metrics port {port}")
            })?
    } else {
        registry
            .models
            .iter()
            .find(|(_, model)| model.serve.port == Some(port))
            .map(|(id, _)| id.clone())
            .with_context(|| format!("registry has no model on metrics port {port}"))?
    };
    let model_config = registry
        .models
        .get(&model)
        .with_context(|| format!("no registry model named '{model}'"))?;
    Ok(target_for(&model, model_config, &scene_name))
}

/// A new runtime/worker starts a fresh baseline, not invented historical usage.
fn counter_scope(model: &str, labels_json: &str) -> Result<Vec<String>> {
    let labels: BTreeMap<String, String> = serde_json::from_str(labels_json)?;
    let mut scope = vec![model.to_string()];
    for key in ["runtime", "engine", "replica", "stage"] {
        scope.push(labels.get(key).cloned().unwrap_or_default());
    }
    Ok(scope)
}

fn retained_labels(labels: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    labels
        .iter()
        .filter(|(key, _)| {
            matches!(
                key.as_str(),
                "model_name"
                    | "stream"
                    | "reason"
                    | "modality"
                    | "status"
                    | "priority"
                    | "runtime"
                    | "engine"
                    | "replica"
                    | "stage"
            ) || ALLOWED_CUSTOM_LABELS.contains(&key.as_str())
        })
        .map(|(key, value)| {
            let value = if ALLOWED_CUSTOM_LABELS.contains(&key.as_str()) {
                sanitize_custom_label(value)
            } else {
                value.chars().take(128).collect()
            };
            (key.clone(), value)
        })
        .collect()
}

fn sanitize_custom_label(value: &str) -> String {
    if !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        value.to_string()
    } else {
        "invalid".to_string()
    }
}

fn label_or_unknown(labels: &BTreeMap<String, String>, key: &str) -> String {
    labels
        .get(key)
        // SGLang emits empty custom labels on unlabeled traffic. They are
        // deliberately collapsed to the same bounded sentinel as malformed
        // values at ingestion, and both render as unattributed usage.
        .filter(|value| !value.is_empty() && value.as_str() != "invalid")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string())
}

/// Map an engine's counters onto the durable (SGLang-named) schema; never
/// synthesize absent metrics. Runtime/engine labels preserve independent
/// counter resets and worker series.
fn normalize_runtime_metrics(engine: Engine, metrics: &[Metric]) -> Vec<Metric> {
    match engine {
        Engine::Sglang => return metrics.to_vec(),
        Engine::Vllm | Engine::LlamaCpp => {}
        // No token counters to map: load gauges only.
        Engine::NemoSpeech | Engine::Generic => return Vec::new(),
    }
    let has_cached_tokens = metrics
        .iter()
        .any(|m| m.name == "vllm:prompt_tokens_cached_total");
    let mut normalized = Vec::new();
    for metric in metrics {
        let name = match metric.name.as_str() {
            "vllm:num_requests_running" => "sglang:num_running_reqs",
            "vllm:num_requests_waiting" => "sglang:num_queue_reqs",
            "vllm:request_success_total" => "sglang:num_requests_total",
            "vllm:prompt_tokens_total" => "sglang:prompt_tokens_total",
            "vllm:generation_tokens_total" => "sglang:generation_tokens_total",
            "vllm:prompt_tokens_cached_total" => "sglang:cached_tokens_total",
            "vllm:prefix_cache_hits_total" if !has_cached_tokens => "sglang:cached_tokens_total",
            "vllm:time_to_first_token_seconds_sum" => "sglang:time_to_first_token_seconds_sum",
            "vllm:time_to_first_token_seconds_count" => "sglang:time_to_first_token_seconds_count",
            "vllm:e2e_request_latency_seconds_sum" => "sglang:e2e_request_latency_seconds_sum",
            "vllm:e2e_request_latency_seconds_count" => "sglang:e2e_request_latency_seconds_count",
            "llamacpp:requests_processing" => "sglang:num_running_reqs",
            "llamacpp:requests_deferred" => "sglang:num_queue_reqs",
            "llamacpp:prompt_tokens_total" => "sglang:prompt_tokens_total",
            "llamacpp:tokens_predicted_total" => "sglang:generation_tokens_total",
            _ => continue,
        };
        let mut metric = metric.clone();
        metric.name = name.to_string();
        metric
            .labels
            .insert("runtime".to_string(), engine.as_str().to_string());
        if let Some(reason) = metric.labels.remove("finished_reason") {
            metric.labels.insert("reason".to_string(), reason);
        }
        if name == "sglang:num_requests_total"
            && metric.labels.get("reason").is_some_and(|r| r == "abort")
        {
            let mut aborted = metric.clone();
            aborted.name = "sglang:num_aborted_requests_total".to_string();
            normalized.push(aborted);
        }
        normalized.push(metric);
    }
    normalized
}

fn default_gauge(metrics: &[Metric], name: &str) -> Option<f64> {
    let values: Vec<f64> = metrics
        .iter()
        .filter(|metric| metric.name == name)
        .filter(|metric| metric.labels.get("priority").is_none_or(String::is_empty))
        .filter(|metric| metric.value.is_finite())
        .map(|metric| metric.value)
        .collect();
    (!values.is_empty()).then(|| values.iter().sum())
}

pub(crate) fn parse_duration(value: &str) -> Result<Duration> {
    let split = value
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(value.len());
    let amount = value[..split]
        .parse::<u64>()
        .with_context(|| format!("invalid duration '{value}'"))?;
    let unit = &value[split..];
    let seconds = match unit {
        "s" => amount,
        "m" => amount.saturating_mul(60),
        "h" => amount.saturating_mul(60 * 60),
        "d" => amount.saturating_mul(24 * 60 * 60),
        "w" => amount.saturating_mul(7 * 24 * 60 * 60),
        _ => bail!("duration must end in s, m, h, d, or w (for example 24h or 7d)"),
    };
    Ok(Duration::from_secs(seconds))
}

pub(crate) fn unix_time_ms() -> Result<i64> {
    let milliseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")?
        .as_millis();
    Ok(milliseconds.min(i64::MAX as u128) as i64)
}

fn fnv1a64(input: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in input {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("lumbridge-computer-telemetry-{suffix}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn counter_deltas_survive_runtime_resets() {
        let root = temp_root();
        let mut store = Store::open(&root).unwrap();
        let target = Target {
            model: "brain".to_string(),
            scene: "solo".to_string(),
            max_concurrency: 4,
            config_fingerprint: "test".to_string(),
            engine: Engine::Sglang,
            port: Some(8001),
        };
        let labels = BTreeMap::from([
            ("agent".to_string(), "a".to_string()),
            ("client".to_string(), "c".to_string()),
        ]);
        let mut record = |value| {
            store
                .record_with_counter_interval(
                    &target,
                    &[
                        Metric {
                            name: "sglang:num_running_reqs".to_string(),
                            labels: BTreeMap::new(),
                            value: 1.0,
                        },
                        Metric {
                            name: "sglang:num_requests_total".to_string(),
                            labels: labels.clone(),
                            value,
                        },
                    ],
                    0,
                )
                .unwrap();
            thread::sleep(Duration::from_millis(2));
        };
        record(10.0);
        let since = unix_time_ms().unwrap() - 1;
        record(14.0);
        record(2.0); // runtime restarted
        let summary = store.summary(since, unix_time_ms().unwrap()).unwrap();
        assert_eq!(summary.totals.requests, 6.0);
        assert_eq!(summary.agents[0].requests, 6.0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn new_labeled_series_count_from_first_use_and_identify_vision() {
        let root = temp_root();
        let mut store = Store::open(&root).unwrap();
        let target = Target {
            model: "brain".to_string(),
            scene: "solo".to_string(),
            max_concurrency: 4,
            config_fingerprint: "test".to_string(),
            engine: Engine::Sglang,
            port: Some(8001),
        };
        let gauge = Metric {
            name: "sglang:num_running_reqs".to_string(),
            labels: BTreeMap::new(),
            value: 0.0,
        };
        let baseline_labels = BTreeMap::from([
            ("agent".to_string(), "".to_string()),
            ("client".to_string(), "".to_string()),
            ("workload".to_string(), "".to_string()),
        ]);
        store
            .record_with_counter_interval(
                &target,
                &[
                    gauge.clone(),
                    Metric {
                        name: "sglang:num_requests_total".to_string(),
                        labels: baseline_labels.clone(),
                        value: 10.0,
                    },
                    Metric {
                        name: "sglang:prompt_tokens_total".to_string(),
                        labels: baseline_labels,
                        value: 100.0,
                    },
                ],
                0,
            )
            .unwrap();
        thread::sleep(Duration::from_millis(2));
        let since = unix_time_ms().unwrap() - 1;
        let vision_labels = BTreeMap::from([
            ("agent".to_string(), "eye-1".to_string()),
            ("client".to_string(), "cloud-agent".to_string()),
            ("workload".to_string(), "vision".to_string()),
        ]);
        store
            .record_with_counter_interval(
                &target,
                &[
                    gauge,
                    Metric {
                        name: "sglang:num_requests_total".to_string(),
                        labels: vision_labels.clone(),
                        value: 1.0,
                    },
                    Metric {
                        name: "sglang:prompt_tokens_total".to_string(),
                        labels: vision_labels,
                        value: 334.0,
                    },
                ],
                0,
            )
            .unwrap();

        let summary = store.summary(since, unix_time_ms().unwrap()).unwrap();
        assert_eq!(summary.totals.requests, 1.0);
        assert_eq!(summary.totals.vision_requests, 1.0);
        assert_eq!(summary.totals.vision_prompt_tokens, 334.0);
        assert_eq!(summary.totals.vision_request_source, "workload_label");
        assert_eq!(summary.agents[0].agent, "eye-1");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn vllm_workers_and_finish_reasons_are_durable_without_double_counting() {
        let root = temp_root();
        let mut store = Store::open(&root).unwrap();
        let target = Target {
            model: "brain".into(),
            scene: "everything".into(),
            max_concurrency: 4,
            config_fingerprint: "vllm-test".into(),
            engine: Engine::Vllm,
            port: Some(8001),
        };
        let scrape = |value: f64| {
            prom::parse(&format!(
            "vllm:num_requests_running{{engine=\"0\"}} 1\nvllm:num_requests_running{{engine=\"1\"}} 2\n\
             vllm:num_requests_waiting{{engine=\"0\"}} 1\n\
             vllm:request_success_total{{engine=\"0\",finished_reason=\"stop\"}} {value}\n\
             vllm:request_success_total{{engine=\"1\",finished_reason=\"stop\"}} {value}\n\
             vllm:request_success_total{{engine=\"0\",finished_reason=\"abort\"}} 1\n\
             vllm:prompt_tokens_cached_total{{engine=\"0\"}} 12\n\
             vllm:prefix_cache_hits_total{{engine=\"0\"}} 12\n"
        ))
        };
        let raw_baseline = scrape(10.0);
        let baseline = normalize_runtime_metrics(Engine::Vllm, &raw_baseline);
        assert_eq!(
            default_gauge(&baseline, "sglang:num_running_reqs"),
            Some(3.0)
        );
        assert_eq!(default_gauge(&baseline, "sglang:num_queue_reqs"), Some(1.0));
        assert_eq!(
            baseline
                .iter()
                .filter(|m| m.name == "sglang:cached_tokens_total")
                .count(),
            1
        );
        assert_eq!(
            baseline
                .iter()
                .filter(|m| m.name == "sglang:num_aborted_requests_total")
                .count(),
            1
        );
        store
            .record_with_counter_interval(&target, &raw_baseline, 0)
            .unwrap();
        thread::sleep(Duration::from_millis(2));
        store
            .record_with_counter_interval(&target, &scrape(12.0), 0)
            .unwrap();
        let summary = store.summary(0, unix_time_ms().unwrap()).unwrap();
        assert_eq!(summary.totals.requests, 4.0);
        assert_eq!(summary.concurrency.peak_running, 3);
        assert_eq!(summary.concurrency.peak_queued, 1);
        assert_eq!(summary.agents[0].agent, "unknown");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn changing_runtime_starts_a_baseline_without_backfilling_lifetime_counters() {
        let root = temp_root();
        let mut store = Store::open(&root).unwrap();
        let target = Target {
            model: "brain".into(),
            scene: "everything".into(),
            max_concurrency: 4,
            config_fingerprint: "test".into(),
            // Resolved per scrape from the metric names: SGLang, then vLLM.
            engine: Engine::Generic,
            port: Some(8001),
        };
        store
            .record_with_counter_interval(
                &target,
                &prom::parse("sglang:num_running_reqs 0\nsglang:num_requests_total 100\n"),
                0,
            )
            .unwrap();
        thread::sleep(Duration::from_millis(2));
        let vllm = |value| {
            prom::parse(&format!("vllm:num_requests_running{{engine=\"0\"}} 0\nvllm:request_success_total{{engine=\"0\",finished_reason=\"stop\"}} {value}\n"))
        };
        store
            .record_with_counter_interval(&target, &vllm(1500), 0)
            .unwrap();
        assert_eq!(
            store
                .summary(0, unix_time_ms().unwrap())
                .unwrap()
                .totals
                .requests,
            0.0
        );
        thread::sleep(Duration::from_millis(2));
        store
            .record_with_counter_interval(&target, &vllm(1501), 0)
            .unwrap();
        assert_eq!(
            store
                .summary(0, unix_time_ms().unwrap())
                .unwrap()
                .totals
                .requests,
            1.0
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sglang_adapter_is_identity_and_vllm_missing_values_stay_absent() {
        let sglang = prom::parse("sglang:num_running_reqs 2\nsglang:gen_throughput 40\n");
        assert_eq!(normalize_runtime_metrics(Engine::Sglang, &sglang), sglang);
        let vllm =
            normalize_runtime_metrics(Engine::Vllm, &prom::parse("vllm:num_requests_running 0\n"));
        assert_eq!(default_gauge(&vllm, "sglang:gen_throughput"), None);
        assert!(!vllm.iter().any(|m| m.name.contains("encoder")));
    }

    fn target(model: &str, engine: Engine, port: u16) -> Target {
        Target {
            model: model.to_string(),
            scene: "everything".to_string(),
            max_concurrency: 4,
            config_fingerprint: "test".to_string(),
            engine,
            port: Some(port),
        }
    }

    #[test]
    fn real_vllm_capture_records_kv_usage_engine_and_port() {
        let root = temp_root();
        let mut store = Store::open(&root).unwrap();
        let samples = prom::parse(include_str!("testdata/vllm-omni-tts.prom"));
        let result = store
            .record(&target("voice", Engine::Vllm, 8097), &samples)
            .unwrap();
        assert_eq!((result.running, result.queued), (1, 0));
        assert_eq!(result.engine, Engine::Vllm);
        assert!(result.metrics_recorded > 0);
        let (kv, engine, port): (Option<f64>, String, u16) = store
            .conn
            .query_row(
                "SELECT kv_usage, engine, port FROM engine_samples",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert!(kv.unwrap() > 0.0);
        assert_eq!((engine.as_str(), port), ("vllm", 8097));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn synthetic_llama_cpp_tokens_reach_the_durable_counters() {
        let root = temp_root();
        let mut store = Store::open(&root).unwrap();
        let samples = prom::parse(include_str!("testdata/llamacpp-synthetic.prom"));
        let result = store
            .record_with_counter_interval(&target("brain", Engine::LlamaCpp, 8001), &samples, 0)
            .unwrap();
        assert_eq!((result.running, result.queued), (2, 1));
        assert_eq!(result.kv_usage, Some(0.42));
        let names: Vec<String> = store
            .conn
            .prepare("SELECT metric FROM counter_samples ORDER BY metric")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(
            names,
            [
                "sglang:generation_tokens_total",
                "sglang:prompt_tokens_total"
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unknown_load_is_refused_rather_than_recorded_as_idle() {
        let root = temp_root();
        let mut store = Store::open(&root).unwrap();
        let samples = prom::parse(include_str!("testdata/nemo-speech-asr.prom"));
        assert!(store
            .record(&target("asr", Engine::NemoSpeech, 8791), &samples)
            .is_err());
        assert!(store
            .record(&target("brain", Engine::Vllm, 8001), &[])
            .is_err());
        let rows: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM engine_samples", [], |row| row.get(0))
            .unwrap();
        assert_eq!(rows, 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_v1_database_gains_the_new_columns_and_keeps_its_rows() {
        let root = temp_root();
        let path = root.join(DB_RELATIVE_PATH);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE engine_samples (
               timestamp_ms INTEGER NOT NULL, model TEXT NOT NULL, scene TEXT NOT NULL,
               config_fingerprint TEXT NOT NULL, max_concurrency INTEGER NOT NULL,
               running REAL NOT NULL, queued REAL NOT NULL,
               generation_tps REAL, cache_hit_rate REAL,
               PRIMARY KEY (timestamp_ms, model));
             INSERT INTO engine_samples VALUES (1000, 'brain', 'solo', 'fp', 4, 2, 0, NULL, NULL);",
        )
        .unwrap();
        drop(conn);
        let store = Store::open(&root).unwrap();
        Store::open(&root).unwrap(); // idempotent
        let summary = store.summary(0, 2000).unwrap();
        assert_eq!(summary.concurrency.samples, 1);
        assert_eq!(summary.concurrency.peak_running, 2);
        assert_eq!(summary.concurrency_scope.as_deref(), Some("model brain"));
        let version: i64 = store
            .conn
            .query_row("SELECT MAX(version) FROM telemetry_schema", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(version, 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn top_level_concurrency_follows_the_busiest_port_and_models_are_listed() {
        let root = temp_root();
        let mut store = Store::open(&root).unwrap();
        let brain = |running| prom::parse(&format!("vllm:num_requests_running {running}\n"));
        let idle = prom::parse("vllm:num_requests_running 0\n");
        for running in [2, 3, 0] {
            store
                .record(&target("brain-osprey", Engine::Vllm, 8001), &brain(running))
                .unwrap();
            store
                .record(&target("embed", Engine::Vllm, 8012), &idle)
                .unwrap();
            store
                .record(&target("rerank", Engine::Vllm, 8015), &idle)
                .unwrap();
            thread::sleep(Duration::from_millis(2));
        }
        // A brain swap on the same port stays in the same distribution.
        store
            .record(&target("brain-q27", Engine::Vllm, 8001), &brain(1))
            .unwrap();
        let summary = store.summary(0, unix_time_ms().unwrap()).unwrap();
        assert_eq!(summary.concurrency_scope.as_deref(), Some("port 8001"));
        assert_eq!(summary.concurrency.samples, 4);
        assert_eq!(summary.concurrency.peak_running, 3);
        let models: Vec<&str> = summary.models.iter().map(|m| m.model.as_str()).collect();
        assert_eq!(models, ["brain-osprey", "brain-q27", "embed", "rerank"]);
        assert_eq!(summary.models[2].concurrency.samples, 3);
        assert_eq!(summary.models[2].engine.as_deref(), Some("vllm"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scrape_failures_are_logged_on_change_then_at_most_hourly() {
        let mut log = ScrapeLog::default();
        let start = Instant::now();
        let fail = || vec![("brain".to_string(), Err("connection refused".to_string()))];
        let ok = || {
            vec![(
                "brain".to_string(),
                Ok("vllm, scene everything".to_string()),
            )]
        };

        assert_eq!(
            log.observe(start, &ok(), true),
            ["scraping brain (vllm, scene everything)"]
        );
        assert!(log.observe(start, &ok(), true).is_empty());
        assert_eq!(
            log.observe(start, &fail(), true),
            ["brain: scrape failed: connection refused"]
        );
        // 1320 more failures over the next 110 minutes: one reminder an hour.
        let mut printed = 0;
        for tick in 1..=1320u64 {
            let now = start + Duration::from_secs(tick * 5);
            printed += log.observe(now, &fail(), true).len();
        }
        assert_eq!(printed, 1);
        let later = start + Duration::from_secs(1321 * 5);
        assert_eq!(
            log.observe(later, &ok(), true),
            ["brain: recovered after 1321 failed scrape(s)"]
        );
        assert_eq!(
            log.observe(later, &[], true),
            ["brain: not serving; stopped scraping"]
        );
    }

    #[test]
    fn a_failed_pass_does_not_forget_targets_it_could_not_see() {
        let mut log = ScrapeLog::default();
        let now = Instant::now();
        log.observe(now, &[("brain".to_string(), Ok("vllm".to_string()))], true);
        let error = vec![("collector".to_string(), Err("parsing registry".to_string()))];
        assert_eq!(
            log.observe(now, &error, false),
            ["collector: scrape failed: parsing registry"]
        );
        assert!(log.observe(now, &error, false).is_empty());
        assert_eq!(
            log.observe(now, &[("brain".to_string(), Ok("vllm".to_string()))], true),
            ["collector: recovered"]
        );
    }

    /// Answers one `GET /metrics`; port probes (connect, no request) are ignored.
    fn fake_engine(body: &'static str) -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let head = crate::engines::http::tests::read_request_head(&mut stream);
                if head.starts_with("GET /metrics ") {
                    use std::io::Write;
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                        body.len()
                    );
                    return;
                }
            }
        });
        port
    }

    #[test]
    fn collector_scrapes_each_running_model_by_engine_without_a_metrics_url() {
        let root = temp_root();
        let vllm = fake_engine(include_str!("testdata/vllm-pooling-embed.prom"));
        let llama = fake_engine(include_str!("testdata/llamacpp-synthetic.prom"));
        fs::create_dir_all(root.join("registry")).unwrap();
        fs::write(
            root.join("registry/models.yaml"),
            format!(
                "apiVersion: lumbridge/v1\nmodels:\n\
                 \x20 embed:\n    name: e\n    footprint_gb: 1\n    serve:\n      kind: vllm\n      port: {vllm}\n\
                 \x20 brain:\n    name: b\n    footprint_gb: 1\n    serve:\n      kind: exec\n      port: {llama}\n      command: [llama-server, --metrics, -np, '2']\n\
                 \x20 image:\n    name: i\n    footprint_gb: 1\n    serve:\n      kind: diffusers\n      port: {vllm}\n\
                 \x20 stopped:\n    name: s\n    footprint_gb: 1\n    serve:\n      kind: vllm\n      port: 1\n"
            ),
        )
        .unwrap();
        let report = collect_once(&root, None, "auto", Duration::from_secs(1)).unwrap();
        assert!(report.failures.is_empty(), "{:?}", report.failures);
        let mut got: Vec<(String, Engine, u32, u32)> = report
            .results
            .iter()
            .map(|r| (r.model.clone(), r.engine, r.running, r.max_concurrency))
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            got,
            [
                ("brain".to_string(), Engine::LlamaCpp, 2, 2),
                ("embed".to_string(), Engine::Vllm, 0, 0),
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn metrics_url_override_still_scrapes_one_endpoint_and_sniffs_the_engine() {
        let root = temp_root();
        let port = fake_engine(include_str!("testdata/vllm-0.27-generate.prom"));
        fs::create_dir_all(root.join("registry")).unwrap();
        fs::write(
            root.join("registry/models.yaml"),
            format!(
                "apiVersion: lumbridge/v1\nmodels:\n  brain:\n    name: b\n    footprint_gb: 1\n    serve:\n      kind: docker\n      port: {port}\n      command: [docker, run, opaque/image]\n"
            ),
        )
        .unwrap();
        let url = format!("http://127.0.0.1:{port}/metrics");
        let report = collect_once(&root, Some(&url), "auto", Duration::from_secs(1)).unwrap();
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].model, "brain");
        assert_eq!(report.results[0].engine, Engine::Vllm);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn duration_parser_is_explicit() {
        assert_eq!(parse_duration("24h").unwrap(), Duration::from_secs(86_400));
        assert_eq!(parse_duration("7d").unwrap(), Duration::from_secs(604_800));
        assert!(parse_duration("24").is_err());
    }

    #[test]
    fn caller_labels_are_bounded_and_cannot_carry_structured_data() {
        assert_eq!(sanitize_custom_label(""), "invalid");
        assert_eq!(sanitize_custom_label("worker-1"), "worker-1");
        assert_eq!(sanitize_custom_label("person@example.com"), "invalid");
        assert_eq!(sanitize_custom_label(&"x".repeat(65)), "invalid");
        assert_eq!(
            label_or_unknown(
                &BTreeMap::from([("agent".to_string(), "invalid".to_string())]),
                "agent"
            ),
            "unknown"
        );
    }
}
