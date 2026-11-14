//! The box: `lumbridge-computer.toml` desired state for the Spark, its Supabase and its
//! Tailscale exposure, with read-only `doctor` and `apply --dry-run`.
//!
//! Doctor and the dry run never mutate the host: every command goes through
//! [`cmd::System`], which refuses anything outside a read-only allowlist; see
//! docs/box.md. The one real apply, `apply --only supabase-floor`, lives in
//! [`floor`] and writes systemd drop-ins plus a `daemon-reload`, nothing else.

pub mod cmd;
pub mod config;
pub mod doctor;
pub mod floor;
pub mod observe;
pub mod plan;

use anyhow::Result;
use std::path::{Path, PathBuf};

/// The box config's file name.
pub const CONFIG_FILE: &str = "lumbridge-computer.toml";

/// `--config`, then `$LUMBRIDGE_COMPUTER_CONFIG`, then `./lumbridge-computer.toml`.
pub fn config_path(flag: Option<&Path>) -> PathBuf {
    flag.map(Path::to_path_buf)
        .or_else(|| crate::env_var("CONFIG").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(CONFIG_FILE))
}

/// Exit code: 0 when nothing failed (warnings allowed), 1 when any check failed.
pub fn doctor(config: Option<&Path>, json: bool) -> Result<i32> {
    let config = config::BoxConfig::load(&config_path(config))?;
    let observed = observe::gather(&cmd::System::default(), &config);
    let checks = doctor::run(&config, &observed);
    if json {
        println!("{}", serde_json::to_string_pretty(&checks)?);
    } else {
        print!("{}", doctor::render_text(&checks));
    }
    Ok(i32::from(
        checks.iter().any(|c| c.status == doctor::Status::Fail),
    ))
}

/// The items `lumbridge-computer apply --only` can apply for real.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ApplyOnly {
    /// cgroup `memory.min` on the Supabase slice (systemd drop-ins + daemon-reload).
    SupabaseFloor,
}

/// Exit code: 0 no changes, 2 changes pending, 1 when some area couldn't be read.
/// For a real `--only supabase-floor`: 0 done or nothing to do, 3 needs root (the
/// exact commands were printed), 1 on error.
pub fn apply(
    config: Option<&Path>,
    dry_run: bool,
    json: bool,
    only: Option<ApplyOnly>,
    host_root: Option<&Path>,
) -> Result<i32> {
    if !dry_run && only.is_none() {
        eprintln!(
            "lumbridge-computer apply applies one item at a time so far: \
             `lumbridge-computer apply --only supabase-floor`. \
             Run `lumbridge-computer apply --dry-run` to see the whole diff."
        );
        return Ok(1);
    }
    let config = config::BoxConfig::load(&config_path(config))?;
    if let (false, Some(ApplyOnly::SupabaseFloor)) = (dry_run, only) {
        return apply_floor(&config, host_root.unwrap_or(Path::new("/")));
    }
    let observed = observe::gather(&cmd::System::default(), &config);
    let mut result = plan::dry_run(&config, &observed);
    if only == Some(ApplyOnly::SupabaseFloor) {
        result = plan::DryRun {
            changes: plan::floor_change(&config, &observed).into_iter().collect(),
            blockers: Vec::new(),
        };
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print!("{}", plan::render_text(&result));
    }
    Ok(if !result.blockers.is_empty() {
        1
    } else if result.changes.is_empty() {
        0
    } else {
        2
    })
}

fn apply_floor(config: &config::BoxConfig, host: &Path) -> Result<i32> {
    let real = host == Path::new("/");
    let writable = floor::dir_writable(&host.join("etc/systemd/system"));
    let mut reload = || -> Result<bool> {
        if real {
            floor::daemon_reload().map(|()| true)
        } else {
            println!(
                "supabase-floor: --host-root {} is not the live system; not reloading systemd",
                host.display()
            );
            Ok(false)
        }
    };
    let applied = floor::apply(
        config,
        &cmd::System::default(),
        host,
        observe::effective_uid(),
        writable,
        &mut reload,
        &mut std::io::stdout(),
    )?;
    Ok(match applied {
        floor::Applied::NeedsRoot => floor::EXIT_NEEDS_ROOT,
        floor::Applied::Unchanged | floor::Applied::Written(_) => 0,
    })
}

#[cfg(test)]
mod asset_tests {
    //! The shipped Supabase overlay is what doctor checks for; keep them in step.

    const OVERLAY: &str = include_str!("../../assets/supabase/lumbridge-computer.override.yml");
    const UPSTREAM: &str = include_str!("../../assets/supabase/UPSTREAM");

    fn services() -> serde_yaml::Mapping {
        let mut doc: serde_yaml::Value = serde_yaml::from_str(OVERLAY).unwrap();
        doc.apply_merge().unwrap();
        doc["services"].as_mapping().unwrap().clone()
    }

    fn bytes(limit: &str) -> f64 {
        let (n, unit) = limit.split_at(limit.len() - 1);
        let n: f64 = n.parse().unwrap();
        match unit {
            "g" => n * 1024.0 * 1024.0 * 1024.0,
            "m" => n * 1024.0 * 1024.0,
            other => panic!("unit {other}"),
        }
    }

    #[test]
    fn overlay_puts_every_service_in_the_configurable_slice_with_a_limit() {
        let mut total = 0.0;
        for (name, svc) in services() {
            assert_eq!(
                svc["cgroup_parent"].as_str(),
                Some("${LUMBRIDGE_COMPUTER_SLICE:-lumbridge-computer.slice}"),
                "{name:?}"
            );
            total += bytes(svc["mem_limit"].as_str().unwrap());
        }
        // The reservation the example lumbridge-computer.toml and registry declare.
        assert_eq!(total / super::observe::GIB, 8.5);
    }

    #[test]
    fn overlay_publishes_ports_on_loopback_only_and_protects_postgres() {
        let services = services();
        for (name, svc) in &services {
            if let Some(ports) = svc.get("ports") {
                let serde_yaml::Value::Tagged(tagged) = ports else {
                    panic!("{name:?} ports must use !override so upstream bindings are replaced");
                };
                for p in tagged.value.as_sequence().unwrap() {
                    assert!(p.as_str().unwrap().starts_with("127.0.0.1:"), "{p:?}");
                }
            }
        }
        assert_eq!(services["db"]["oom_score_adj"].as_i64(), Some(-900));
    }

    #[test]
    fn upstream_is_pinned_to_a_full_commit_of_v0_8_4() {
        let get = |k: &str| {
            UPSTREAM
                .lines()
                .find_map(|l| l.strip_prefix(k))
                .unwrap()
                .to_string()
        };
        assert_eq!(get("ref="), "self-hosted/v0.8.4");
        let commit = get("commit=");
        assert_eq!(commit.len(), 40);
        assert!(commit.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
