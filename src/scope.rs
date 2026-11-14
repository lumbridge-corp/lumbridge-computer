//! Per-model transient systemd user scopes (M2).
//!
//! Each model is launched as `systemd-run --user --scope … -- <argv>`. Why a
//! scope through systemd rather than writing cgroupfs directly:
//!
//! - **pid semantics are free.** `systemd-run --scope` asks the user manager to
//!   create the scope around its *own* pid, waits for the job, then `execvp`s the
//!   model. The recorded pid therefore *is* the model process: same pid, same
//!   process group (set by `spawn` before exec), same start time — so `Proc`
//!   identity, `stop_owned` group signals and adoption are unchanged. (Measured
//!   on a dev box: the pid reads `systemd-run` for <50 ms, then the model, with an
//!   identical `/proc/<pid>/stat` start time and pgrp.)
//! - **stdio, env and cwd are inherited**, unlike `systemd-run` service mode,
//!   so the per-model log file and registry `env:` keep working.
//! - **no delegation needed.** Writing cgroupfs ourselves only works inside a
//!   delegated subtree; the agent runs in `lumbridge-computer-agent.service`
//!   with `Delegate=no`, and moving processes out of it from within is exactly
//!   what systemd forbids. The user manager already owns `memory` on its
//!   subtree (typically `cpu memory pids`).
//! - **models leave the agent's cgroup.** Without scopes every model lives in
//!   the agent's service cgroup, so its `memory.current` is one number for all
//!   models. A scope gives each model its own `memory.*` files, and an agent
//!   restart with any `KillMode` can no longer take models with it.
//!
//! Scopes are best-effort: without a reachable user manager the model is
//! spawned exactly as in M1 and reported as `unscoped`.

use std::path::{Path, PathBuf};

use crate::config::{Model, ScopesMode};

/// Parent slice for every model scope. systemd nests slices by their dashes, so in
/// the user manager it lives at `lumbridge.slice/lumbridge-computer.slice/lumbridge-computer-models.slice`.
pub const SLICE: &str = "lumbridge-computer-models.slice";
/// Unit name prefix; `footprint` recognises scoped models by it.
pub const UNIT_PREFIX: &str = "lumbridge-computer-model-";
/// Default `MemoryHigh` headroom over the declared footprint.
pub const MEMORY_HIGH_FACTOR: f64 = 1.15;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// cgroup limits in bytes, after defaults are applied. `None` = leave unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Resolved {
    pub memory_high: Option<u64>,
    pub memory_max: Option<u64>,
    pub swap_max: Option<u64>,
}

fn bytes(gb: f64) -> Option<u64> {
    (gb.is_finite() && gb >= 0.0).then(|| (gb * GIB).round() as u64)
}

/// Apply the defaults to a model's optional `limits:` block.
///
/// - `MemoryHigh` = footprint × 1.15. It throttles and reclaims, never kills.
///   A 0 GB shim gets none (a 0-byte high would stall it forever).
/// - `MemoryMax` unset: an OOM kill of the brain is worse than the watchdog
///   shedding something else.
/// - `MemorySwapMax` = 0 for anything that holds memory: a model paging its
///   host side to a 16 GB swap file is a latency cliff on the first request.
///   Shims (0 GB) may swap.
pub fn resolve(m: &Model) -> Resolved {
    let limits = m.limits.clone().unwrap_or_default();
    let holds_memory = m.footprint_gb > 0.0;
    let memory_high = match limits.memory_high_gb {
        Some(gb) => bytes(gb),
        None if holds_memory => bytes(m.footprint_gb * MEMORY_HIGH_FACTOR),
        None => None,
    };
    let swap_max = match limits.swap_max_gb {
        Some(gb) => bytes(gb),
        None if holds_memory => Some(0),
        None => None,
    };
    Resolved {
        memory_high,
        memory_max: limits.memory_max_gb.and_then(bytes),
        swap_max,
    }
}

/// A systemd unit name for this model. Registry ids are already `[a-z0-9-]`;
/// anything else is replaced so a hostile id cannot inject unit syntax.
pub fn unit_name(id: &str, nonce: u64) -> String {
    let safe: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    // The nonce keeps a leftover scope (a daemonised grandchild outliving its
    // group) from blocking the next start with "unit already exists".
    format!("{UNIT_PREFIX}{safe}-{nonce:x}")
}

/// How a model gets wrapped. A trait so tests never need systemd.
pub trait Launcher {
    /// True when a user manager is reachable to create the scope.
    fn available(&self) -> bool;
    /// The argv that runs `argv` inside the scope `unit`.
    fn command(&self, unit: &str, limits: &Resolved, argv: &[String]) -> Vec<String>;
}

/// The production launcher: `systemd-run --user --scope`.
pub struct SystemdRun;

impl Launcher for SystemdRun {
    fn available(&self) -> bool {
        // `systemd-run --user` connects to `$XDG_RUNTIME_DIR/systemd/private`
        // first; no socket means no user manager, and the scope would fail.
        let Some(runtime) = std::env::var_os("XDG_RUNTIME_DIR") else {
            return false;
        };
        PathBuf::from(runtime).join("systemd/private").exists() && which("systemd-run")
    }

    fn command(&self, unit: &str, limits: &Resolved, argv: &[String]) -> Vec<String> {
        systemd_run_argv(unit, limits, argv)
    }
}

/// `systemd-run` argv, split out so it is testable without systemd.
pub fn systemd_run_argv(unit: &str, limits: &Resolved, argv: &[String]) -> Vec<String> {
    let mut out: Vec<String> = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        // A scope OOM-killed by MemoryMax ends "failed" and would linger.
        "--collect",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    out.push(format!("--unit={unit}"));
    out.push(format!("--slice={SLICE}"));
    let mut property = |name: &str, value: Option<u64>| {
        if let Some(value) = value {
            out.push("-p".into());
            out.push(format!("{name}={value}"));
        }
    };
    property("MemoryHigh", limits.memory_high);
    property("MemoryMax", limits.memory_max);
    property("MemorySwapMax", limits.swap_max);
    out.push("--".into());
    out.extend(argv.iter().cloned());
    out
}

fn which(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|dir| is_executable(&dir.join(program)))
    })
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata()
        .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// True when the registry command already wraps itself in `systemd-run`
/// (some TTS servers' launch scripts do). Wrapping twice would just move the
/// process from our scope into theirs, so we leave those alone.
pub fn already_scoped(argv: &[String]) -> bool {
    argv.first()
        .and_then(|program| Path::new(program).file_name())
        .is_some_and(|name| name == "systemd-run")
}

/// What `spawn` should execute, and the scope unit if one is used.
pub fn plan(
    launcher: &dyn Launcher,
    mode: ScopesMode,
    id: &str,
    model: &Model,
    argv: Vec<String>,
    nonce: u64,
) -> (Vec<String>, Option<String>) {
    if mode == ScopesMode::Off || already_scoped(&argv) || !launcher.available() {
        return (argv, None);
    }
    let unit = unit_name(id, nonce);
    let wrapped = launcher.command(&unit, &resolve(model), &argv);
    (wrapped, Some(format!("{unit}.scope")))
}

/// `LUMBRIDGE_COMPUTER_SCOPES=off|on` overrides the registry for one agent (rollback
/// without editing the registry); anything else is ignored.
pub fn mode(registry: ScopesMode) -> ScopesMode {
    match crate::env_var("SCOPES").as_deref() {
        Some("off") => ScopesMode::Off,
        Some("on") => ScopesMode::On,
        _ => registry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(yaml: &str) -> Model {
        serde_yaml::from_str(yaml).unwrap()
    }

    struct Stub(bool);
    impl Launcher for Stub {
        fn available(&self) -> bool {
            self.0
        }
        fn command(&self, unit: &str, limits: &Resolved, argv: &[String]) -> Vec<String> {
            systemd_run_argv(unit, limits, argv)
        }
    }

    const GPU: &str =
        "name: brain\nfootprint_gb: 40\nserve:\n  kind: exec\n  command: [vllm, serve]\n";

    #[test]
    fn defaults_throttle_above_the_footprint_never_kill_and_never_swap() {
        let r = resolve(&model(GPU));
        assert_eq!(r.memory_high, Some((40.0 * 1.15 * GIB).round() as u64));
        assert_eq!(r.memory_max, None);
        assert_eq!(r.swap_max, Some(0));
    }

    #[test]
    fn a_zero_gb_shim_gets_no_limits_at_all() {
        let r = resolve(&model(
            "name: shim\nfootprint_gb: 0\nserve:\n  kind: exec\n  command: [python3]\n",
        ));
        assert_eq!(r, Resolved::default());
    }

    #[test]
    fn explicit_limits_override_every_default() {
        let r = resolve(&model(
            "name: b\nfootprint_gb: 40\nlimits:\n  memory_high_gb: 10\n  memory_max_gb: 12\n  swap_max_gb: 2\nserve:\n  kind: exec\n  command: [x]\n",
        ));
        assert_eq!(r.memory_high, Some(10 * 1024 * 1024 * 1024));
        assert_eq!(r.memory_max, Some(12 * 1024 * 1024 * 1024));
        assert_eq!(r.swap_max, Some(2 * 1024 * 1024 * 1024));
    }

    #[test]
    fn unknown_limit_keys_are_rejected() {
        let bad: Result<Model, _> = serde_yaml::from_str(
            "name: b\nfootprint_gb: 1\nlimits:\n  memory_hgih_gb: 1\nserve:\n  kind: exec\n",
        );
        assert!(bad.is_err());
    }

    #[test]
    fn the_wrapped_command_keeps_the_original_argv_after_a_separator() {
        let (argv, unit) = plan(
            &Stub(true),
            ScopesMode::On,
            "brain",
            &model(GPU),
            vec!["vllm".into(), "serve".into()],
            0xabc,
        );
        assert_eq!(
            unit.as_deref(),
            Some("lumbridge-computer-model-brain-abc.scope")
        );
        assert_eq!(&argv[..3], ["systemd-run", "--user", "--scope"]);
        assert!(argv.contains(&"--unit=lumbridge-computer-model-brain-abc".to_string()));
        assert!(argv.contains(&"--slice=lumbridge-computer-models.slice".to_string()));
        assert!(argv.contains(&"MemorySwapMax=0".to_string()));
        assert!(!argv.iter().any(|a| a.starts_with("MemoryMax=")));
        let sep = argv.iter().position(|a| a == "--").unwrap();
        assert_eq!(&argv[sep + 1..], ["vllm", "serve"]);
    }

    #[test]
    fn scopes_off_unavailable_or_self_wrapped_commands_run_unchanged() {
        let argv = vec!["vllm".to_string()];
        let m = model(GPU);
        for (launcher, mode) in [(Stub(true), ScopesMode::Off), (Stub(false), ScopesMode::On)] {
            assert_eq!(
                plan(&launcher, mode, "b", &m, argv.clone(), 1),
                (argv.clone(), None)
            );
        }
        let wrapped = vec!["/usr/bin/systemd-run".to_string(), "--scope".into()];
        assert_eq!(
            plan(&Stub(true), ScopesMode::On, "b", &m, wrapped.clone(), 1),
            (wrapped, None)
        );
    }

    #[test]
    fn unit_names_cannot_carry_unit_syntax() {
        assert_eq!(
            unit_name("a/b c.service", 1),
            "lumbridge-computer-model-a_b_c_service-1"
        );
    }

    #[test]
    fn registry_scopes_and_admission_default_on_and_declared() {
        let r = crate::config::Registry::parse("apiVersion: lumbridge/v1\nmodels: {}\n").unwrap();
        assert_eq!(r.scopes, ScopesMode::On);
        assert_eq!(r.admission, crate::config::AdmissionMode::Declared);
        let r = crate::config::Registry::parse(
            "apiVersion: lumbridge/v1\nscopes: off\nadmission: measured\nmodels: {}\n",
        )
        .unwrap();
        assert_eq!(r.scopes, ScopesMode::Off);
        assert_eq!(r.admission, crate::config::AdmissionMode::Measured);
    }
}
