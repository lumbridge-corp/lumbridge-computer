//! The Supabase memory floor: cgroup v2 `memory.min` on the stack's slice, so that
//! under pressure the kernel reclaims from models before it touches Postgres.
//!
//! The overlay caps each container and lowers Postgres's `oom_score_adj`, and the
//! governor holds the reservation back from admission. None of that stops the
//! kernel from *reclaiming* the database's pages (swapping it, dropping its cache)
//! when a model overshoots its footprint. `memory.min` does: memory under it is
//! never reclaimed while anything unprotected can be.
//!
//! A cgroup's protection is capped by its parent's, so the floor goes on every
//! slice from the top of the tree down to the Supabase slice: by default both
//! `lumbridge.slice` and `lumbridge-server.slice`. The Docker containers join a
//! *system* slice (dockerd is a system service), so the drop-ins live under
//! /etc/systemd/system and applying them needs root.

use super::cmd::Cmd;
use super::config::BoxConfig;
use super::observe::GIB;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Drop-in file name; `50-` sorts after distribution defaults and before local overrides.
pub const DROP_IN: &str = "50-lumbridge-computer-memory-min.conf";

/// Which systemd manager owns the Supabase slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// `/sys/fs/cgroup/<slice path>`: PID 1's slice. Docker's `cgroup_parent` lands here.
    System,
    /// Under `user@<uid>.service`: a user manager's slice. Its protection is capped by
    /// `user.slice`, which Lumbridge Computer does not manage, so a floor here protects nothing.
    User,
    /// No such cgroup: the stack is not running (systemd removes empty implicit slices).
    Missing,
}

/// One slice on the path from the cgroup root to the Supabase slice.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SliceFloor {
    /// systemd unit name, e.g. `lumbridge.slice`.
    pub slice: String,
    /// Path under /sys/fs/cgroup.
    pub cgroup: String,
    /// `memory.min` / `memory.low` in bytes; `None` when unreadable.
    pub min_bytes: Option<u64>,
    pub low_bytes: Option<u64>,
    /// Current contents of Lumbridge Computer's drop-in for this slice, if present.
    pub drop_in: Option<String>,
    /// The drop-in's file name, [`DROP_IN`]. Not serialised: `--json` output
    /// keeps its shape.
    #[serde(skip)]
    pub drop_in_name: &'static str,
}

/// What the host says about the floor today.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FloorFacts {
    pub level: Level,
    /// Outermost slice first.
    pub chain: Vec<SliceFloor>,
}

impl FloorFacts {
    /// The guarantee the kernel actually gives the Supabase slice: the smallest
    /// `memory.min` on its path (an unreadable one counts as none).
    pub fn effective_min_bytes(&self) -> u64 {
        self.chain
            .iter()
            .map(|s| s.min_bytes.unwrap_or(0))
            .min()
            .unwrap_or(0)
    }

    /// Best-effort protection: `memory.low` (or a higher `memory.min`) on every slice.
    pub fn effective_low_bytes(&self) -> u64 {
        self.chain
            .iter()
            .map(|s| s.low_bytes.unwrap_or(0).max(s.min_bytes.unwrap_or(0)))
            .min()
            .unwrap_or(0)
    }
}

/// systemd slice names from the root down: `a-b.slice` -> [`a.slice`, `a-b.slice`].
pub fn slice_chain(slice: &str) -> Vec<String> {
    let stem = slice.trim_end_matches(".slice");
    let parts: Vec<&str> = stem.split('-').collect();
    (1..=parts.len())
        .map(|n| format!("{}.slice", parts[..n].join("-")))
        .collect()
}

pub fn target_bytes(c: &BoxConfig) -> u64 {
    (c.supabase.memory_min_gb * GIB).round() as u64
}

/// systemd spells whole GiB as `4G` (base 1024); anything else as plain bytes.
pub fn systemd_size(bytes: u64) -> String {
    let gib = 1u64 << 30;
    if bytes > 0 && bytes.is_multiple_of(gib) {
        format!("{}G", bytes / gib)
    } else {
        bytes.to_string()
    }
}

#[cfg(test)]
pub fn drop_in_path(etc: &Path, slice: &str) -> PathBuf {
    drop_in_path_named(etc, slice, DROP_IN)
}

fn drop_in_path_named(etc: &Path, slice: &str, name: &str) -> PathBuf {
    etc.join("systemd/system")
        .join(format!("{slice}.d"))
        .join(name)
}

/// The drop-in for `slice` under `etc`, with its contents if present.
fn read_drop_in(cmd: &dyn Cmd, etc: &Path, slice: &str) -> (Option<String>, &'static str) {
    let text = cmd.read(&drop_in_path_named(etc, slice, DROP_IN)).ok();
    (text, DROP_IN)
}

pub fn drop_in_contents(c: &BoxConfig) -> String {
    let size = systemd_size(target_bytes(c));
    format!(
        "# Written by `lumbridge-computer apply --only supabase-floor` from lumbridge-computer.toml\n\
         # ([supabase] memory_min_gb = {gb}). cgroup v2 memory.min: under memory pressure\n\
         # the kernel reclaims from unprotected cgroups (the models) before it touches\n\
         # the first {size} of {slice}. A slice's protection is capped by its\n\
         # parent's, so every slice above it carries the same floor.\n\
         [Slice]\n\
         MemoryMin={size}\n",
        gb = c.supabase.memory_min_gb,
        slice = c.supabase.slice,
    )
}

/// Read the floor through `cmd` (read-only). `host` prefixes every path so tests and
/// `--host-root` can point at a scratch tree; production passes `/`.
pub fn observe(cmd: &dyn Cmd, host: &Path, c: &BoxConfig, uid: u32) -> FloorFacts {
    let cgroup_root = host.join("sys/fs/cgroup");
    let own = c.slice_cgroup();
    let user_prefix = format!("user.slice/user-{uid}.slice/user@{uid}.service");
    let level = if cmd.exists(&cgroup_root.join(&own).join("memory.min")) {
        Level::System
    } else if cmd.exists(&cgroup_root.join(&user_prefix).join(&own).join("memory.min")) {
        Level::User
    } else {
        Level::Missing
    };
    let base = match level {
        Level::User => format!("{user_prefix}/"),
        _ => String::new(),
    };
    let read_bytes = |path: PathBuf| -> Option<u64> {
        let text = cmd.read(&path).ok()?;
        match text.trim() {
            "max" => Some(u64::MAX),
            n => n.parse().ok(),
        }
    };
    let mut cgroup = String::new();
    let chain = slice_chain(&c.supabase.slice)
        .into_iter()
        .map(|slice| {
            if !cgroup.is_empty() {
                cgroup.push('/');
            }
            cgroup.push_str(&slice);
            let dir = cgroup_root.join(format!("{base}{cgroup}"));
            let (drop_in, drop_in_name) = read_drop_in(cmd, &host.join("etc"), &slice);
            SliceFloor {
                min_bytes: read_bytes(dir.join("memory.min")),
                low_bytes: read_bytes(dir.join("memory.low")),
                drop_in,
                drop_in_name,
                cgroup: format!("{base}{cgroup}"),
                slice,
            }
        })
        .collect();
    FloorFacts { level, chain }
}

/// One file `apply` would write.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlannedFile {
    pub path: PathBuf,
    pub contents: String,
}

/// What reaching the floor takes, given the facts.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FloorPlan {
    /// Drop-ins that are missing or differ, with their full contents.
    pub files: Vec<PlannedFile>,
    /// The kernel already guarantees the target (whoever set it).
    pub satisfied: bool,
}

pub fn plan(c: &BoxConfig, facts: &FloorFacts) -> FloorPlan {
    let target = target_bytes(c);
    let contents = drop_in_contents(c);
    let satisfied = target == 0 || facts.effective_min_bytes() >= target;
    let files = if satisfied {
        Vec::new()
    } else {
        facts
            .chain
            .iter()
            .filter(|s| s.drop_in.as_deref() != Some(contents.as_str()))
            .map(|s| PlannedFile {
                path: drop_in_path_named(Path::new("/etc"), &s.slice, s.drop_in_name),
                contents: contents.clone(),
            })
            .collect()
    };
    FloorPlan { files, satisfied }
}

/// One line describing the floor as it is.
pub fn describe(facts: &FloorFacts) -> String {
    facts
        .chain
        .iter()
        .map(|s| {
            let gb = |b: Option<u64>| match b {
                Some(u64::MAX) => "max".to_string(),
                Some(b) => format!("{:.1}", b as f64 / GIB),
                None => "?".to_string(),
            };
            format!(
                "{} min {} / low {} GB",
                s.slice,
                gb(s.min_bytes),
                gb(s.low_bytes)
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The commands a human runs to apply `files` as root, exactly.
pub fn sudo_commands(files: &[PlannedFile]) -> String {
    let mut out = String::new();
    for f in files {
        if let Some(dir) = f.path.parent() {
            out.push_str(&format!("sudo install -d -m 0755 {}\n", dir.display()));
        }
        out.push_str(&format!(
            "sudo tee {} >/dev/null <<'LUMBRIDGE_COMPUTER_EOF'\n{}LUMBRIDGE_COMPUTER_EOF\n",
            f.path.display(),
            f.contents
        ));
    }
    out.push_str("sudo systemctl daemon-reload\n");
    out
}

/// Result of `lumbridge-computer apply --only supabase-floor`.
#[derive(Debug, PartialEq)]
pub enum Applied {
    /// Already guaranteed; nothing written.
    Unchanged,
    /// Drop-ins written (paths) and systemd reloaded.
    Written(Vec<PathBuf>),
    /// Needs root: the exact commands were printed instead.
    NeedsRoot,
}

/// Exit code for `apply --only supabase-floor` when a human must run the printed commands.
pub const EXIT_NEEDS_ROOT: i32 = 3;

/// Apply the floor under `host` (`/` in production). `reload` runs
/// `systemctl daemon-reload` and says whether it did (a scratch `host` skips it);
/// it is a parameter so tests never reach the real systemd.
/// `writable` says whether this process may write `host/etc/systemd/system`.
pub fn apply(
    c: &BoxConfig,
    cmd: &dyn Cmd,
    host: &Path,
    uid: u32,
    writable: bool,
    reload: &mut dyn FnMut() -> Result<bool>,
    out: &mut dyn std::io::Write,
) -> Result<Applied> {
    let facts = observe(cmd, host, c, uid);
    if facts.level == Level::User {
        bail!(
            "{} is a user slice (under user@{uid}.service); its memory.min is capped by \
             user.slice, so a floor there protects nothing. Put the Supabase containers in a \
             system slice (the overlay's cgroup_parent) and re-run.",
            c.supabase.slice
        );
    }
    let p = plan(c, &facts);
    if p.satisfied {
        writeln!(
            out,
            "supabase-floor: nothing to do; the kernel already guarantees {} GiB ({})",
            c.supabase.memory_min_gb,
            describe(&facts)
        )?;
        return Ok(Applied::Unchanged);
    }
    if !writable {
        writeln!(
            out,
            "supabase-floor: {} is a system slice; writing its drop-ins needs root.\n\
             Nothing was changed. Run these as an administrator:\n",
            c.supabase.slice
        )?;
        write!(out, "{}", sudo_commands(&p.files))?;
        writeln!(
            out,
            "\nThen check: lumbridge-computer doctor (supabase.memory_floor) or\n  cat /sys/fs/cgroup/{}/memory.min",
            c.slice_cgroup()
        )?;
        return Ok(Applied::NeedsRoot);
    }
    let mut written = Vec::new();
    for f in &p.files {
        let rel = f.path.strip_prefix("/").unwrap_or(&f.path);
        let path = host.join(rel);
        write_atomic(&path, &f.contents)?;
        writeln!(out, "supabase-floor: wrote {}", f.path.display())?;
        written.push(f.path.clone());
    }
    // Files may already be in place from a run whose reload never happened; the
    // kernel value is still short, so reloading is the remaining step either way.
    if reload().context("systemctl daemon-reload")? {
        writeln!(out, "supabase-floor: ran systemctl daemon-reload")?;
    }
    let after = observe(cmd, host, c, uid);
    if after.effective_min_bytes() >= target_bytes(c) {
        writeln!(out, "supabase-floor: now {}", describe(&after))?;
    } else {
        writeln!(
            out,
            "supabase-floor: the kernel does not show the floor yet ({}). If the stack is \
             stopped it applies when the slice starts; otherwise check `systemctl show {} -p MemoryMin`.",
            describe(&after),
            c.supabase.slice
        )?;
    }
    Ok(Applied::Written(written))
}

/// `systemctl daemon-reload`: the only command Lumbridge Computer runs that changes the host,
/// spelled out here instead of going through the read-only [`super::cmd::System`].
pub fn daemon_reload() -> Result<()> {
    let status = std::process::Command::new("systemctl")
        .arg("daemon-reload")
        .stdin(std::process::Stdio::null())
        .status()
        .context("running systemctl daemon-reload")?;
    if !status.success() {
        bail!(
            "systemctl daemon-reload exited {}",
            status.code().unwrap_or(-1)
        );
    }
    Ok(())
}

/// Write via a temp file and rename, so systemd never reads a half-written drop-in.
fn write_atomic(path: &Path, contents: &str) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let dir = path
        .parent()
        .with_context(|| format!("{} has no parent", path.display()))?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let tmp = dir.join(format!(".{DROP_IN}.tmp"));
    std::fs::write(&tmp, contents).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644))?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))?;
    Ok(())
}

/// Can this process create files in `dir` (or the nearest existing ancestor)?
pub fn dir_writable(dir: &Path) -> bool {
    let mut probe = dir;
    while !probe.exists() {
        match probe.parent() {
            Some(parent) => probe = parent,
            None => return false,
        }
    }
    let Ok(c) = std::ffi::CString::new(probe.as_os_str().as_encoded_bytes()) else {
        return false;
    };
    // access(2) answers for the real uid, which is what sudo sets; no file is created.
    unsafe { libc::access(c.as_ptr(), libc::W_OK) == 0 }
}

#[cfg(test)]
mod tests {
    use super::super::cmd::{Fake, System};
    use super::super::config::{BoxConfig, EXAMPLE};
    use super::*;

    fn config() -> BoxConfig {
        BoxConfig::parse(EXAMPLE, Some(Path::new("/home/sam"))).unwrap()
    }

    const SERVER: &str = "/sys/fs/cgroup/lumbridge.slice/lumbridge-server.slice";
    const PARENT: &str = "/sys/fs/cgroup/lumbridge.slice";

    /// The fixture box: system slice, no protection anywhere.
    fn box1() -> Fake {
        let mut f = Fake::default();
        for dir in [PARENT, SERVER] {
            f.file(&format!("{dir}/memory.min"), "0\n")
                .file(&format!("{dir}/memory.low"), "0\n");
        }
        f
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lumbridge-computer-floor-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_floor_defaults_to_the_postgres_container_limit() {
        assert_eq!(config().supabase.memory_min_gb, 4.0);
        assert_eq!(target_bytes(&config()), 4 << 30);
    }

    #[test]
    fn nested_slices_put_the_floor_on_every_ancestor() {
        assert_eq!(
            slice_chain("lumbridge-server.slice"),
            vec!["lumbridge.slice", "lumbridge-server.slice"]
        );
        assert_eq!(
            slice_chain("lumbridge-computer.slice"),
            vec!["lumbridge.slice", "lumbridge-computer.slice"]
        );
    }

    #[test]
    fn box1_is_a_system_slice_with_no_floor() {
        let facts = observe(&box1(), Path::new("/"), &config(), 1000);
        assert_eq!(facts.level, Level::System);
        assert_eq!(facts.chain.len(), 2);
        assert_eq!(
            facts.chain[1].cgroup,
            "lumbridge.slice/lumbridge-server.slice"
        );
        assert_eq!(facts.effective_min_bytes(), 0);
    }

    #[test]
    fn protection_is_capped_by_the_least_protected_ancestor() {
        let mut h = box1();
        h.file(&format!("{SERVER}/memory.min"), "4294967296\n");
        let facts = observe(&h, Path::new("/"), &config(), 1000);
        assert_eq!(facts.effective_min_bytes(), 0, "parent still has none");
        h.file(&format!("{PARENT}/memory.min"), "4294967296\n");
        let facts = observe(&h, Path::new("/"), &config(), 1000);
        assert_eq!(facts.effective_min_bytes(), 4 << 30);
        assert!(plan(&config(), &facts).satisfied);
    }

    #[test]
    fn the_plan_names_each_drop_in_with_its_exact_contents() {
        let facts = observe(&box1(), Path::new("/"), &config(), 1000);
        let p = plan(&config(), &facts);
        assert!(!p.satisfied);
        let paths: Vec<String> = p
            .files
            .iter()
            .map(|f| f.path.display().to_string())
            .collect();
        assert_eq!(
            paths,
            vec![
                "/etc/systemd/system/lumbridge.slice.d/50-lumbridge-computer-memory-min.conf",
                "/etc/systemd/system/lumbridge-server.slice.d/50-lumbridge-computer-memory-min.conf",
            ]
        );
        assert!(p.files[0].contents.ends_with("[Slice]\nMemoryMin=4G\n"));
    }

    #[test]
    fn fractional_floors_are_written_in_bytes() {
        assert_eq!(systemd_size(4 << 30), "4G");
        assert_eq!(systemd_size(3 << 29), "1610612736");
    }

    #[test]
    fn a_user_slice_is_detected_and_refused() {
        let mut h = Fake::default();
        h.file(
            "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/lumbridge.slice/lumbridge-server.slice/memory.min",
            "0\n",
        );
        let facts = observe(&h, Path::new("/"), &config(), 1000);
        assert_eq!(facts.level, Level::User);
        let err = apply(
            &config(),
            &h,
            Path::new("/"),
            1000,
            true,
            &mut || Ok(true),
            &mut Vec::new(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("user slice"));
    }

    #[test]
    fn without_root_it_prints_the_exact_commands_and_changes_nothing() {
        let mut out = Vec::new();
        let mut reloads = 0;
        let got = apply(
            &config(),
            &box1(),
            Path::new("/"),
            1000,
            false,
            &mut || {
                reloads += 1;
                Ok(true)
            },
            &mut out,
        )
        .unwrap();
        assert_eq!(got, Applied::NeedsRoot);
        assert_eq!(reloads, 0);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains(
            "sudo tee /etc/systemd/system/lumbridge-server.slice.d/50-lumbridge-computer-memory-min.conf"
        ));
        assert!(text.contains("sudo install -d -m 0755 /etc/systemd/system/lumbridge.slice.d"));
        assert!(text.contains("MemoryMin=4G\nLUMBRIDGE_COMPUTER_EOF"));
        assert!(text.contains("sudo systemctl daemon-reload"));
    }

    /// A temp root standing in for `/`: the cgroup files are real files, and the
    /// fake reload plays systemd by writing memory.min the way PID 1 would.
    #[test]
    fn apply_writes_the_drop_ins_reloads_once_and_is_idempotent() {
        let root = scratch("apply");
        for dir in [PARENT, SERVER] {
            let d = root.join(dir.trim_start_matches('/'));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("memory.min"), "0\n").unwrap();
            std::fs::write(d.join("memory.low"), "0\n").unwrap();
        }
        let c = config();
        let mut reloads = 0;
        let mut reload = || -> Result<bool> {
            reloads += 1;
            for dir in [PARENT, SERVER] {
                let d = root.join(dir.trim_start_matches('/'));
                std::fs::write(d.join("memory.min"), "4294967296\n").unwrap();
            }
            Ok(true)
        };
        let mut out = Vec::new();
        let got = apply(
            &c,
            &System::default(),
            &root,
            1000,
            true,
            &mut reload,
            &mut out,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(
            got,
            Applied::Written(vec![
                PathBuf::from("/etc/systemd/system/lumbridge.slice.d/50-lumbridge-computer-memory-min.conf"),
                PathBuf::from(
                    "/etc/systemd/system/lumbridge-server.slice.d/50-lumbridge-computer-memory-min.conf"
                ),
            ]),
            "{text}"
        );
        assert!(text.contains("ran systemctl daemon-reload"));
        assert!(text.contains("now lumbridge.slice min 4.0"), "{text}");
        let written = std::fs::read_to_string(root.join(
            "etc/systemd/system/lumbridge-server.slice.d/50-lumbridge-computer-memory-min.conf",
        ))
        .unwrap();
        assert_eq!(written, drop_in_contents(&c));

        // Second run: the kernel shows the floor, so nothing is written or reloaded.
        let mut out = Vec::new();
        let mut second = || -> Result<bool> { panic!("must not reload") };
        let again = apply(
            &c,
            &System::default(),
            &root,
            1000,
            true,
            &mut second,
            &mut out,
        )
        .unwrap();
        assert_eq!(again, Applied::Unchanged);
        assert!(String::from_utf8(out).unwrap().contains("nothing to do"));
        assert_eq!(reloads, 1);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn drop_ins_already_in_place_only_need_the_reload() {
        let c = config();
        let mut h = box1();
        for slice in ["lumbridge.slice", "lumbridge-server.slice"] {
            h.file(
                &drop_in_path(Path::new("/etc"), slice).display().to_string(),
                &drop_in_contents(&c),
            );
        }
        let facts = observe(&h, Path::new("/"), &c, 1000);
        let p = plan(&c, &facts);
        assert!(!p.satisfied && p.files.is_empty());
    }

    #[test]
    fn a_zero_floor_is_satisfied_and_writes_nothing() {
        let mut c = config();
        c.supabase.memory_min_gb = 0.0;
        let facts = observe(&box1(), Path::new("/"), &c, 1000);
        assert!(plan(&c, &facts).satisfied);
    }

    #[test]
    fn writability_is_judged_on_the_nearest_existing_directory() {
        let root = scratch("writable");
        assert!(dir_writable(&root.join("etc/systemd/system")));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
