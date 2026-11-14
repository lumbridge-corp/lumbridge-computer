//! Reproducible evaluations for any OpenAI-compatible model server.
//!
//! Suites are declarative YAML. Results are append-only JSON artifacts suitable
//! for CI, regression comparisons, and future publication to an eval registry.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Deserialize)]
struct Suite {
    #[serde(rename = "apiVersion")]
    api_version: String,
    kind: String,
    metadata: Metadata,
    #[serde(default)]
    defaults: Defaults,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    name: String,
    #[serde(default)]
    version: u32,
    #[serde(default)]
    description: String,
    /// Part of the published suite manifest surface; parsed so a suite
    /// carrying tags loads, not read by the runner itself.
    #[serde(default)]
    #[allow(dead_code)]
    tags: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
struct Defaults {
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_repeat")]
    repeat: u32,
    #[serde(default)]
    system: String,
}

fn default_max_tokens() -> u32 {
    128
}
fn default_temperature() -> f64 {
    0.0
}
fn default_repeat() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    #[serde(default)]
    category: String,
    prompt: String,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    assertions: Vec<Assertion>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Assertion {
    Exact { value: String },
    Contains { value: String },
    ContainsAny { values: Vec<String> },
    NotContains { value: String },
    MaxWords { value: usize },
}

#[derive(Debug, Serialize)]
struct Artifact {
    schema_version: u32,
    run_id: String,
    suite: String,
    suite_version: u32,
    model: String,
    base_url: String,
    started_unix_ms: u128,
    summary: Summary,
    samples: Vec<Sample>,
}

#[derive(Debug, Serialize)]
struct Summary {
    passed: usize,
    total: usize,
    score: f64,
    mean_ttft_ms: f64,
    p50_ttft_ms: f64,
    p95_ttft_ms: f64,
    mean_prefill_tps: f64,
    mean_decode_tps: f64,
}

#[derive(Debug, Serialize)]
struct Sample {
    case_id: String,
    category: String,
    repetition: u32,
    passed: bool,
    assertion_results: Vec<bool>,
    output: String,
    reasoning: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    ttft_ms: f64,
    total_ms: f64,
    prefill_tps: f64,
    decode_tps: f64,
}

struct Completion {
    output: String,
    reasoning: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    ttft_ms: f64,
    total_ms: f64,
}

/// One line of `eval ls`. Suites themselves stay private — a caller has no
/// business reaching into cases and assertions — but the catalogue is the
/// useful part and is shared by the CLI and the MCP server.
#[derive(Debug, Clone)]
pub struct SuiteSummary {
    pub name: String,
    pub version: u32,
    pub cases: usize,
    pub description: String,
}

pub fn catalogue(root: &Path) -> Result<Vec<SuiteSummary>> {
    Ok(load_all(root)?
        .into_iter()
        .map(|s| SuiteSummary {
            name: s.metadata.name,
            version: s.metadata.version,
            cases: s.cases.len(),
            description: s.metadata.description,
        })
        .collect())
}

pub fn list(root: &Path) -> Result<()> {
    println!("{:<20} {:>4}  {:>5}  DESCRIPTION", "SUITE", "VER", "CASES");
    for s in catalogue(root)? {
        println!(
            "{:<20} {:>4}  {:>5}  {}",
            s.name, s.version, s.cases, s.description
        );
    }
    Ok(())
}

pub fn run(
    root: &Path,
    name: &str,
    base_url: &str,
    model: &str,
    repeat: Option<u32>,
) -> Result<()> {
    let suite = load(root, name)?;
    let known_contract = matches!(
        suite.api_version.as_str(),
        "lumbridge-computer/v1" | "lumbridge/v1"
    );
    if !known_contract || suite.kind != "EvalSuite" {
        bail!(
            "unsupported eval contract: {}/{}",
            suite.api_version,
            suite.kind
        );
    }
    let reps = repeat.unwrap_or(suite.defaults.repeat).max(1);
    println!(
        "eval '{}' v{} → {} ({})",
        suite.metadata.name, suite.metadata.version, model, base_url
    );
    let started = now_ms();
    let mut samples = Vec::new();
    for case in &suite.cases {
        for repetition in 1..=reps {
            print!("  {:<24} [{}/{}] ", case.id, repetition, reps);
            std::io::stdout().flush().ok();
            let c = stream_completion(base_url, model, &suite.defaults, case)?;
            let assertion_results: Vec<bool> = case
                .assertions
                .iter()
                .map(|a| evaluate(a, &c.output))
                .collect();
            let passed = assertion_results.iter().all(|v| *v);
            let decode_seconds = ((c.total_ms - c.ttft_ms) / 1000.0).max(0.001);
            let prefill_seconds = (c.ttft_ms / 1000.0).max(0.001);
            let sample = Sample {
                case_id: case.id.clone(),
                category: case.category.clone(),
                repetition,
                passed,
                assertion_results,
                output: c.output,
                reasoning: c.reasoning,
                prompt_tokens: c.prompt_tokens,
                completion_tokens: c.completion_tokens,
                ttft_ms: c.ttft_ms,
                total_ms: c.total_ms,
                prefill_tps: c.prompt_tokens as f64 / prefill_seconds,
                decode_tps: c.completion_tokens as f64 / decode_seconds,
            };
            println!(
                "{}  ttft={:.0}ms decode={:.1}tok/s",
                if passed { "PASS" } else { "FAIL" },
                sample.ttft_ms,
                sample.decode_tps
            );
            samples.push(sample);
        }
    }
    let summary = summarize(&samples);
    let run_id = format!("{}-{}-{}", started, slug(name), slug(model));
    let artifact = Artifact {
        schema_version: 1,
        run_id: run_id.clone(),
        suite: suite.metadata.name,
        suite_version: suite.metadata.version,
        model: model.to_string(),
        base_url: base_url.to_string(),
        started_unix_ms: started,
        summary,
        samples,
    };
    let out_dir = root.join("eval-results");
    fs::create_dir_all(&out_dir)?;
    let out = out_dir.join(format!("{run_id}.json"));
    fs::write(&out, serde_json::to_string_pretty(&artifact)?)?;
    println!(
        "\nscore {:.1}% ({}/{}) · mean TTFT {:.0}ms · prefill≈{:.1}tok/s · decode {:.1}tok/s",
        artifact.summary.score * 100.0,
        artifact.summary.passed,
        artifact.summary.total,
        artifact.summary.mean_ttft_ms,
        artifact.summary.mean_prefill_tps,
        artifact.summary.mean_decode_tps
    );
    println!("result {}", out.display());
    Ok(())
}

fn stream_completion(
    base_url: &str,
    model: &str,
    defaults: &Defaults,
    case: &Case,
) -> Result<Completion> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let mut messages = Vec::new();
    if !defaults.system.is_empty() {
        messages.push(json!({"role":"system","content":defaults.system}));
    }
    messages.push(json!({"role":"user","content":case.prompt}));
    let request = json!({
        "model": model, "messages": messages, "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": case.max_tokens.unwrap_or(defaults.max_tokens),
        "temperature": defaults.temperature,
        "chat_template_kwargs": {"enable_thinking": false}
    });
    let mut child = Command::new("curl")
        .args([
            "-sS",
            "-N",
            "-X",
            "POST",
            &url,
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("starting curl (required for eval HTTP streaming)")?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())?;
    let start = Instant::now();
    let mut first = None;
    let mut output = String::new();
    let mut reasoning = String::new();
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0;
    for line in BufReader::new(child.stdout.take().unwrap()).lines() {
        let line = line?;
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let v: Value = serde_json::from_str(data).context("parsing streamed completion")?;
        if let Some(usage) = v.get("usage") {
            prompt_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(prompt_tokens);
            completion_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(completion_tokens);
        }
        let delta = &v["choices"][0]["delta"];
        let content = delta.get("content").and_then(Value::as_str).unwrap_or("");
        let thought = delta
            .get("reasoning")
            .or_else(|| delta.get("reasoning_content"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if first.is_none() && (!content.is_empty() || !thought.is_empty()) {
            first = Some(start.elapsed());
        }
        output.push_str(content);
        reasoning.push_str(thought);
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("completion request failed with {status}");
    }
    let total = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Completion {
        output,
        reasoning,
        prompt_tokens,
        completion_tokens,
        ttft_ms: first.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(total),
        total_ms: total,
    })
}

fn evaluate(a: &Assertion, output: &str) -> bool {
    let normalized = output.trim();
    match a {
        Assertion::Exact { value } => normalized.eq_ignore_ascii_case(value.trim()),
        Assertion::Contains { value } => normalized.to_lowercase().contains(&value.to_lowercase()),
        Assertion::ContainsAny { values } => values
            .iter()
            .any(|v| normalized.to_lowercase().contains(&v.to_lowercase())),
        Assertion::NotContains { value } => {
            !normalized.to_lowercase().contains(&value.to_lowercase())
        }
        Assertion::MaxWords { value } => normalized.split_whitespace().count() <= *value,
    }
}

fn summarize(samples: &[Sample]) -> Summary {
    let mut ttft: Vec<f64> = samples.iter().map(|s| s.ttft_ms).collect();
    ttft.sort_by(f64::total_cmp);
    let mean = |f: fn(&Sample) -> f64| {
        if samples.is_empty() {
            0.0
        } else {
            samples.iter().map(f).sum::<f64>() / samples.len() as f64
        }
    };
    let percentile = |p: f64| {
        if ttft.is_empty() {
            0.0
        } else {
            ttft[((ttft.len() - 1) as f64 * p).round() as usize]
        }
    };
    let passed = samples.iter().filter(|s| s.passed).count();
    Summary {
        passed,
        total: samples.len(),
        score: if samples.is_empty() {
            0.0
        } else {
            passed as f64 / samples.len() as f64
        },
        mean_ttft_ms: mean(|s| s.ttft_ms),
        p50_ttft_ms: percentile(0.50),
        p95_ttft_ms: percentile(0.95),
        mean_prefill_tps: mean(|s| s.prefill_tps),
        mean_decode_tps: mean(|s| s.decode_tps),
    }
}

fn load(root: &Path, name: &str) -> Result<Suite> {
    load_all(root)?
        .into_iter()
        .find(|s| s.metadata.name == name)
        .with_context(|| format!("no eval suite named '{name}' in {}/evals", root.display()))
}

fn load_all(root: &Path) -> Result<Vec<Suite>> {
    let dir = root.join("evals");
    let mut suites = Vec::new();
    if !dir.exists() {
        return Ok(suites);
    }
    for entry in fs::read_dir(&dir)? {
        let path: PathBuf = entry?.path();
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.ends_with(".eval.yaml"))
            .unwrap_or(false)
        {
            let suite: Suite = serde_yaml::from_str(&fs::read_to_string(&path)?)
                .with_context(|| format!("parsing {}", path.display()))?;
            suites.push(suite);
        }
    }
    suites.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));
    Ok(suites)
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
fn slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assertions_are_case_insensitive_and_composable() {
        let output = "Market risk remains, while company-specific risk falls.";
        assert!(evaluate(
            &Assertion::Contains {
                value: "MARKET RISK".into()
            },
            output
        ));
        assert!(evaluate(
            &Assertion::ContainsAny {
                values: vec!["idiosyncratic".into(), "company-specific".into()],
            },
            output
        ));
        assert!(evaluate(
            &Assertion::NotContains {
                value: "markdown".into()
            },
            output
        ));
    }

    #[test]
    fn exact_trims_but_does_not_accept_explanation() {
        assert!(evaluate(&Assertion::Exact { value: "25".into() }, " 25\n"));
        assert!(!evaluate(&Assertion::Exact { value: "25".into() }, "25 GB"));
    }

    #[test]
    fn word_limit_counts_whitespace_tokens() {
        assert!(evaluate(
            &Assertion::MaxWords { value: 4 },
            "one two three four"
        ));
        assert!(!evaluate(
            &Assertion::MaxWords { value: 3 },
            "one two three four"
        ));
    }
}
