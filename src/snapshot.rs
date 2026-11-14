//! `lumbridge-computer snapshot`: the sanitized, read-only JSON a status
//! publisher can POST somewhere every 30 s (one line: compact JSON, keys
//! sorted, trailing newline; a golden test pins the format).
//!
//! - `models` are the *active* Scene's models in Scene order, else the default
//!   Scene's (`desired_scene` is reported but never chooses the models).
//! - `health` is the full readiness probe: port plus `health_contains` marker.
//! - `scene_budget_gb` is the Scene's declared `budget_gb`, `null` if it has none.
//! - `error_code` is `runtime_error` when the state records a `last_error`.
//!
//! A load failure (registry, Scene, state) never fails the command: it always
//! prints a valid snapshot and reports the failure as `runtime_error`, with the
//! models it could not resolve left out. `runtime_error` is the only error code;
//! receivers may reject any other value.
//!
//! "Sanitized" means nothing from the registry's `serve`/`health` sections, no
//! error text, PIDs, paths or hostnames. Model names are registry-authored
//! display labels and pass through verbatim.

use serde::Serialize;
use std::path::Path;
use std::time::Duration;

use crate::config::{find_scene, Model, Registry, DEFAULT_SCENE};
use crate::governor::{self, Readiness};
use crate::mem::MemInfo;
use crate::pool::Pool;
use crate::proc::State;

/// The RPC rejects any other value.
pub const SCHEMA_VERSION: u32 = 1;

/// The publisher kills the command after 25 s. Probes run in parallel, so the
/// slowest single probe bounds the whole snapshot; a registry probe timeout
/// may be up to 60 s, which is a supervision setting, not a status one.
const PROBE_CAP: Duration = Duration::from_secs(3);

/// The RPC rejects a model whose name is longer than this (Postgres `length`,
/// i.e. characters), which would drop the whole snapshot, not just the model.
const NAME_MAX_CHARS: usize = 200;

/// Field order is alphabetical: serde writes struct fields in declaration
/// order, and the old binary's output has sorted keys.
#[derive(Debug, Serialize)]
pub struct Snapshot {
    active_scene: Option<String>,
    declared_allocation_gb: f64,
    default_scene: &'static str,
    desired_scene: Option<String>,
    error_code: Option<ErrorCode>,
    global_budget_gb: f64,
    memory_available_gb: f64,
    memory_total_gb: f64,
    models: Vec<ModelStatus>,
    scene_budget_gb: Option<f64>,
    schema_version: u32,
}

#[derive(Debug, Serialize)]
struct ModelStatus {
    footprint_gb: f64,
    health: Health,
    id: String,
    name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Health {
    Ready,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ErrorCode {
    RuntimeError,
}

impl Snapshot {
    /// The exact bytes the publisher reads, without the trailing newline.
    pub fn to_line(&self) -> String {
        // Serializing plain structs, strings and finite floats cannot fail; if it
        // somehow did, an empty line makes the publisher retry rather than crash us.
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Read the live node. Never fails: every problem becomes `runtime_error`.
pub fn collect(root: &Path) -> Snapshot {
    let memory = Pool::unified().sense().ok();
    let probe =
        |m: &Model| governor::readiness(m, m.probe_timeout().min(PROBE_CAP)) == Readiness::Ready;
    gather(root, State::load_checked(root), memory, &probe)
}

/// `collect` with the state, memory and readiness probe supplied, for tests.
fn gather(
    root: &Path,
    state: anyhow::Result<State>,
    memory: Option<MemInfo>,
    probe: &(dyn Fn(&Model) -> bool + Sync),
) -> Snapshot {
    let mut failed = memory.is_none();
    let state = state.unwrap_or_else(|_| {
        failed = true;
        State::default()
    });
    failed |= state.last_error.is_some();

    let scene_name = state.active_scene.as_deref().unwrap_or(DEFAULT_SCENE);
    let scene = find_scene(root, scene_name).ok();
    let registry = Registry::load(root).ok();
    failed |= scene.is_none() || registry.is_none();

    let models = match (&registry, &scene) {
        (Some(registry), Some(scene)) => {
            let (models, all_known) = scene_models(registry, &scene.models, probe);
            failed |= !all_known;
            models
        }
        _ => Vec::new(),
    };
    let memory = memory.unwrap_or(MemInfo {
        total_gb: 0.0,
        available_gb: 0.0,
    });

    Snapshot {
        active_scene: state.active_scene,
        declared_allocation_gb: models.iter().map(|m| m.footprint_gb).sum(),
        default_scene: DEFAULT_SCENE,
        desired_scene: state.desired_scene,
        error_code: failed.then_some(ErrorCode::RuntimeError),
        global_budget_gb: Pool::unified().budget_gb,
        memory_available_gb: memory.available_gb,
        memory_total_gb: memory.total_gb,
        models,
        scene_budget_gb: scene.and_then(|s| s.budget_gb),
        schema_version: SCHEMA_VERSION,
    }
}

/// The Scene's models in Scene order, probed in parallel. The flag is false if
/// the Scene names a model the registry does not have; that model is skipped.
fn scene_models(
    registry: &Registry,
    ids: &[String],
    probe: &(dyn Fn(&Model) -> bool + Sync),
) -> (Vec<ModelStatus>, bool) {
    let known: Vec<(&String, &Model)> = ids
        .iter()
        .filter_map(|id| registry.models.get(id).map(|m| (id, m)))
        .collect();
    let all_known = known.len() == ids.len();
    let ready: Vec<bool> = std::thread::scope(|scope| {
        let handles: Vec<_> = known
            .iter()
            .map(|(_, model)| scope.spawn(move || probe(model)))
            .collect();
        // A panicking probe is a model we could not confirm, not a failed snapshot.
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or(false))
            .collect()
    });
    let models = known
        .into_iter()
        .zip(ready)
        .map(|((id, model), ready)| ModelStatus {
            footprint_gb: model.footprint_gb,
            health: if ready {
                Health::Ready
            } else {
                Health::Unavailable
            },
            id: id.clone(),
            name: display_name(id, &model.name),
        })
        .collect();
    (models, all_known)
}

/// The RPC wants 1..=200 characters; trimming keeps one long label from
/// rejecting the whole node's status.
fn display_name(id: &str, name: &str) -> String {
    if name.is_empty() {
        id.to_string()
    } else {
        name.chars().take(NAME_MAX_CHARS).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/box")
    }

    fn state(active: Option<&str>, desired: Option<&str>) -> State {
        State {
            active_scene: active.map(str::to_string),
            desired_scene: desired.map(str::to_string),
            ..State::default()
        }
    }

    fn memory(total_gb: f64, available_gb: f64) -> Option<MemInfo> {
        Some(MemInfo {
            total_gb,
            available_gb,
        })
    }

    /// Everything a sanitized snapshot must never carry, with model names (the
    /// registry's display labels, passed through verbatim) blanked first.
    fn assert_sanitized(line: &str) {
        let mut value: serde_json::Value = serde_json::from_str(line).unwrap();
        for model in value["models"].as_array_mut().unwrap() {
            model["name"] = serde_json::Value::String(String::new());
        }
        let rest = value.to_string();
        assert!(!rest.contains('/'), "path in {rest}");
        assert!(!rest.contains("127.0.0.1"), "address in {rest}");
        assert!(!rest.contains("localhost"), "host in {rest}");
        // Every remaining value is a number, null, or an id-like string, so any
        // colon outside a `"key":` separator is a leaked `host:port` or `:port`.
        assert!(!rest.replace("\":", "\"=").contains(':'), "port in {rest}");
    }

    #[test]
    fn the_fixture_snapshot_is_byte_identical_to_its_golden_line() {
        // Every model ready, and memory as a busy 128 GB box reads it.
        let legacy = include_str!("../tests/fixtures/box/snapshot.golden.json");
        let snapshot = gather(
            &fixture(),
            Ok(state(Some("everything"), Some("everything"))),
            memory(121.68962097167969, 25.903392791748047),
            &|_| true,
        );
        assert_eq!(format!("{}\n", snapshot.to_line()), legacy);
    }

    #[test]
    fn the_golden_snapshot_reports_down_models_as_unavailable_and_still_counts_them() {
        let snapshot = gather(
            &fixture(),
            Ok(state(Some("everything"), None)),
            memory(121.5, 20.25),
            &|m| m.footprint_gb >= 26.0,
        );
        assert_eq!(
            snapshot.to_line(),
            concat!(
                r#"{"active_scene":"everything","declared_allocation_gb":93.0,"default_scene":"everything","#,
                r#""desired_scene":null,"error_code":null,"global_budget_gb":100.0,"memory_available_gb":20.25,"#,
                r#""memory_total_gb":121.5,"models":["#,
                r#"{"footprint_gb":40.0,"health":"ready","id":"brain","name":"General brain 35B-A3B (NVFP4, vision, MTP)"},"#,
                r#"{"footprint_gb":7.0,"health":"unavailable","id":"asr","name":"Streaming speech recognition 0.6B (English)"},"#,
                r#"{"footprint_gb":26.0,"health":"ready","id":"transcribe","name":"Transcription 30B-A4B (GGUF) with diarization and word timestamps"},"#,
                r#"{"footprint_gb":0.0,"health":"unavailable","id":"asr-compat","name":"Older transcription API on :8792, served by transcribe"},"#,
                r#"{"footprint_gb":12.0,"health":"unavailable","id":"tts","name":"Voice-clone TTS 2B (48 kHz)"},"#,
                r#"{"footprint_gb":0.0,"health":"unavailable","id":"tts-compat","name":"OpenAI-compatible speech endpoint -> tts (24 kHz)"},"#,
                r#"{"footprint_gb":3.0,"health":"unavailable","id":"embed","name":"Embedding model 1B (NVFP4)"},"#,
                r#"{"footprint_gb":5.0,"health":"unavailable","id":"rerank","name":"Reranker 1B (BF16, cross-encoder)"}"#,
                r#"],"scene_budget_gb":111.0,"schema_version":1}"#,
            )
        );
    }

    #[test]
    fn models_come_from_the_active_scene_and_never_from_the_desired_one() {
        let active = gather(
            &fixture(),
            Ok(state(Some("stock"), Some("stock"))),
            memory(1.0, 1.0),
            &|_| false,
        );
        let ids: Vec<_> = active.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["brain-long", "asr", "tts", "vision", "tts-fast"]);
        assert_eq!(active.scene_budget_gb, Some(100.0));

        // Desired but not active: the old binary showed the default Scene.
        let desired = gather(
            &fixture(),
            Ok(state(None, Some("stock"))),
            memory(1.0, 1.0),
            &|_| false,
        );
        assert_eq!(desired.models[0].id, "brain");
        assert_eq!(desired.scene_budget_gb, Some(111.0));
        assert_eq!(desired.error_code, None);
    }

    #[test]
    fn a_recorded_last_error_becomes_runtime_error_without_its_text() {
        let mut failing = state(Some("everything"), Some("everything"));
        failing.last_error = Some("spawn failed: /home/sam/x on 127.0.0.1:8001".into());
        let line = gather(&fixture(), Ok(failing), memory(1.0, 1.0), &|_| true).to_line();
        assert!(line.contains(r#""error_code":"runtime_error""#), "{line}");
        assert!(!line.contains("spawn failed"), "{line}");
        assert_sanitized(&line);
    }

    fn temp_root(name: &str, registry: &str, scene: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "lumbridge-computer-snapshot-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("registry")).unwrap();
        std::fs::create_dir_all(root.join("scenes")).unwrap();
        std::fs::write(root.join("registry/models.yaml"), registry).unwrap();
        std::fs::write(root.join("scenes/everything.scene.yaml"), scene).unwrap();
        root
    }

    const SCENE: &str = "apiVersion: lumbridge/v1\nkind: Scene\nmetadata: {name: everything}\nmodels: [base, child, ghost]\n";
    const REGISTRY: &str = "apiVersion: lumbridge/v1\nkind: Registry\nmodels:\n  base:\n    name: Base\n    footprint_gb: 4\n    serve: {kind: exec, port: 9001, command: [/bin/true]}\n  child:\n    extends: base\n    name: Child\n";

    #[test]
    fn a_broken_registry_still_prints_a_valid_snapshot_with_runtime_error_and_no_models() {
        let root = temp_root("broken", "models: [not: valid", SCENE);
        let snapshot = gather(&root, Ok(State::default()), memory(1.0, 1.0), &|_| true);
        std::fs::remove_dir_all(&root).unwrap();
        let value: serde_json::Value = serde_json::from_str(&snapshot.to_line()).unwrap();
        assert_eq!(value["error_code"], "runtime_error");
        assert_eq!(value["models"], serde_json::json!([]));
        assert_eq!(value["declared_allocation_gb"], 0.0);
        assert_eq!(value["schema_version"], 1);
        assert_sanitized(&snapshot.to_line());
    }

    #[test]
    fn an_extends_model_resolves_and_an_unknown_scene_model_is_skipped_as_runtime_error() {
        let root = temp_root("extends", REGISTRY, SCENE);
        let snapshot = gather(&root, Ok(State::default()), memory(1.0, 1.0), &|_| true);
        std::fs::remove_dir_all(&root).unwrap();
        let ids: Vec<_> = snapshot
            .models
            .iter()
            .map(|m| (m.id.as_str(), m.footprint_gb))
            .collect();
        assert_eq!(ids, [("base", 4.0), ("child", 4.0)]);
        assert_eq!(snapshot.declared_allocation_gb, 8.0);
        assert_eq!(snapshot.error_code, Some(ErrorCode::RuntimeError));
        assert_eq!(snapshot.scene_budget_gb, None);
    }

    #[test]
    fn unreadable_state_or_memory_is_runtime_error_not_a_failed_command() {
        let snapshot = gather(&fixture(), Err(anyhow::anyhow!("bad state")), None, &|_| {
            true
        });
        assert_eq!(snapshot.error_code, Some(ErrorCode::RuntimeError));
        assert_eq!(snapshot.active_scene, None);
        assert_eq!(snapshot.memory_total_gb, 0.0);
        assert_eq!(snapshot.models.len(), 8);
    }

    #[test]
    fn every_fixture_scene_snapshot_is_sanitized_and_publishable() {
        for scene in crate::config::load_scenes(&fixture()).unwrap() {
            let line = gather(
                &fixture(),
                Ok(state(
                    Some(&scene.metadata.name),
                    Some(&scene.metadata.name),
                )),
                memory(121.68962097167969, 25.9),
                &|m| m.footprint_gb > 5.0,
            )
            .to_line();
            assert_sanitized(&line);
            let value: serde_json::Value = serde_json::from_str(&line).unwrap();
            for model in value["models"].as_array().unwrap() {
                let name = model["name"].as_str().unwrap();
                assert!(
                    (1..=NAME_MAX_CHARS).contains(&name.chars().count()),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn the_whole_fixture_registry_in_one_snapshot_stays_under_64_kib() {
        let registry = Registry::load(&fixture()).unwrap();
        let ids: Vec<String> = registry.models.keys().cloned().collect();
        let (models, all_known) = scene_models(&registry, &ids, &|_| true);
        assert!(all_known);
        let mut snapshot = gather(
            &fixture(),
            Ok(State::default()),
            memory(121.7, 25.9),
            &|_| true,
        );
        snapshot.models = models;
        let line = snapshot.to_line();
        assert!(line.len() + 1 < 64 * 1024, "{} bytes", line.len());
        assert_eq!(snapshot.models.len(), registry.models.len());
    }

    #[test]
    fn long_or_empty_names_are_made_acceptable_to_the_publish_rpc() {
        assert_eq!(display_name("embed", ""), "embed");
        assert_eq!(
            display_name("x", &"é".repeat(250)).chars().count(),
            NAME_MAX_CHARS
        );
        assert_eq!(display_name("x", "Short"), "Short");
    }
}
