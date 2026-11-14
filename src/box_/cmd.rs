//! The one seam between the box checks and the host: commands and file reads.
//!
//! Doctor and the dry-run planner never call `std::process::Command` or `std::fs`
//! directly. Tests inject [`Fake`] with outputs in the shape a real box gives; production
//! uses [`System`], which refuses any command not on a read-only allowlist, so a
//! bug in this module can't turn a check into a mutation of a production box.

use anyhow::{bail, Context, Result};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub trait Cmd {
    /// Run `program args…`; stdout on exit 0, an error carrying stderr otherwise.
    fn run(&self, program: &str, args: &[&str]) -> Result<String>;
    fn read(&self, path: &Path) -> Result<String>;
    fn exists(&self, path: &Path) -> bool;
    /// Names of the entries in a directory (not recursive), e.g. the PIDs under /proc.
    fn list_dir(&self, path: &Path) -> Result<Vec<String>>;
    /// Permission bits (e.g. 0o600), `None` when the file can't be stat'ed.
    fn mode(&self, path: &Path) -> Option<u32>;
}

/// Real host access, read-only by construction.
pub struct System {
    pub timeout: Duration,
}

impl Default for System {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(20),
        }
    }
}

/// Is this invocation one of the read-only commands the box checks are allowed to run?
/// Anything not matched here is refused before a process is spawned.
pub fn is_read_only(program: &str, args: &[&str]) -> bool {
    match (program, args) {
        ("uname", ["-m"]) => true,
        ("tailscale", ["status", "--json"])
        | ("tailscale", ["serve", "status", "--json"])
        | ("tailscale", ["version"]) => true,
        ("docker", ["ps", rest @ ..]) => {
            rest.iter().all(|a| a.starts_with("--format") || *a == "-q")
        }
        ("docker", ["inspect", "--format", _, ids @ ..]) => {
            !ids.is_empty() && ids.iter().all(|id| !id.starts_with('-'))
        }
        ("docker", ["compose", "ls", "--format", "json"]) => true,
        ("docker", ["compose", "-p", _, "ps", "--format", "json"]) => true,
        ("docker", ["exec", container, "pg_isready"]) => !container.starts_with('-'),
        // `nvidia-smi` alone or with queries only: `-r`, `-pm`, `-pl` etc. change the GPU.
        ("nvidia-smi", rest) => rest
            .iter()
            .all(|a| a.starts_with("--query-gpu=") || a.starts_with("--format=")),
        ("df", ["-P", "-B1", path]) => !path.starts_with('-'),
        _ => false,
    }
}

impl Cmd for System {
    fn run(&self, program: &str, args: &[&str]) -> Result<String> {
        if !is_read_only(program, args) {
            bail!(
                "refusing to run non-allowlisted command: {program} {}",
                args.join(" ")
            );
        }
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("running {program}"))?;
        // Read pipes on threads so a chatty command can't fill a pipe and deadlock the wait.
        let mut stdout = child.stdout.take().context("stdout pipe")?;
        let mut stderr = child.stderr.take().context("stderr pipe")?;
        let out = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = stdout.read_to_string(&mut s);
            s
        });
        let err = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = stderr.read_to_string(&mut s);
            s
        });
        let deadline = Instant::now() + self.timeout;
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if Instant::now() > deadline {
                let _ = child.kill();
                let _ = child.wait();
                bail!("{program} timed out after {:?}", self.timeout);
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let stdout = out.join().unwrap_or_default();
        let stderr = err.join().unwrap_or_default();
        if !status.success() {
            bail!(
                "{program} {} exited {}: {}",
                args.first().unwrap_or(&""),
                status.code().unwrap_or(-1),
                stderr.trim()
            );
        }
        Ok(stdout)
    }

    fn read(&self, path: &Path) -> Result<String> {
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))
    }

    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn list_dir(&self, path: &Path) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in
            std::fs::read_dir(path).with_context(|| format!("listing {}", path.display()))?
        {
            names.push(entry?.file_name().to_string_lossy().into_owned());
        }
        Ok(names)
    }

    fn mode(&self, path: &Path) -> Option<u32> {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .ok()
            .map(|m| m.permissions().mode() & 0o777)
    }
}

/// Canned host for tests: commands match by prefix of the joined command line.
#[cfg(test)]
#[derive(Default, Clone)]
pub struct Fake {
    pub commands: Vec<(String, Result<String, String>)>,
    pub files: std::collections::BTreeMap<std::path::PathBuf, String>,
    pub modes: std::collections::BTreeMap<std::path::PathBuf, u32>,
}

#[cfg(test)]
impl Fake {
    /// Answer commands starting with `prefix`; the longest matching prefix wins and
    /// re-registering a prefix replaces its answer.
    pub fn on(&mut self, prefix: &str, out: &str) -> &mut Self {
        self.set(prefix, Ok(out.to_string()))
    }
    /// Make every command starting with `prefix` fail, overriding longer prefixes too.
    pub fn fail(&mut self, prefix: &str, err: &str) -> &mut Self {
        self.commands.retain(|(p, _)| !p.starts_with(prefix));
        self.set(prefix, Err(err.to_string()))
    }
    fn set(&mut self, prefix: &str, answer: Result<String, String>) -> &mut Self {
        self.commands.retain(|(p, _)| p != prefix);
        self.commands.push((prefix.to_string(), answer));
        self
    }
    pub fn file(&mut self, path: &str, content: &str) -> &mut Self {
        self.files.insert(path.into(), content.to_string());
        self
    }
    pub fn remove_file(&mut self, path: &str) -> &mut Self {
        self.files.remove(Path::new(path));
        self
    }
}

#[cfg(test)]
impl Cmd for Fake {
    fn run(&self, program: &str, args: &[&str]) -> Result<String> {
        // The fake enforces the same allowlist, so tests prove the checks only ask for
        // commands production would permit.
        assert!(
            is_read_only(program, args),
            "box check asked for a non-allowlisted command: {program} {args:?}"
        );
        let line = std::iter::once(program)
            .chain(args.iter().copied())
            .collect::<Vec<_>>()
            .join(" ");
        let hit = self
            .commands
            .iter()
            .filter(|(p, _)| line.starts_with(p.as_str()))
            .max_by_key(|(p, _)| p.len());
        match hit {
            Some((_, Ok(out))) => Ok(out.clone()),
            Some((_, Err(e))) => bail!("{e}"),
            None => bail!("{program}: command not found (fake)"),
        }
    }
    fn read(&self, path: &Path) -> Result<String> {
        self.files
            .get(path)
            .cloned()
            .with_context(|| format!("reading {}: no such file (fake)", path.display()))
    }
    fn exists(&self, path: &Path) -> bool {
        self.files.contains_key(path)
    }
    fn list_dir(&self, path: &Path) -> Result<Vec<String>> {
        let names: std::collections::BTreeSet<String> = self
            .files
            .keys()
            .filter_map(|f| f.strip_prefix(path).ok())
            .filter_map(|rest| rest.components().next())
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        if names.is_empty() {
            bail!("listing {}: no such directory (fake)", path.display());
        }
        Ok(names.into_iter().collect())
    }
    fn mode(&self, path: &Path) -> Option<u32> {
        self.modes
            .get(path)
            .copied()
            .or_else(|| self.files.contains_key(path).then_some(0o600))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_permits_the_documented_read_only_commands() {
        for (p, a) in [
            ("tailscale", vec!["status", "--json"]),
            ("tailscale", vec!["serve", "status", "--json"]),
            ("docker", vec!["ps", "-q"]),
            ("docker", vec!["compose", "ls", "--format", "json"]),
            (
                "docker",
                vec!["compose", "-p", "x", "ps", "--format", "json"],
            ),
            (
                "docker",
                vec!["inspect", "--format", "{{json .Name}}", "abc"],
            ),
            ("docker", vec!["exec", "supabase-db", "pg_isready"]),
            (
                "nvidia-smi",
                vec!["--query-gpu=name", "--format=csv,noheader"],
            ),
            ("nvidia-smi", vec![]),
            ("df", vec!["-P", "-B1", "/home"]),
        ] {
            assert!(is_read_only(p, &a), "{p} {a:?}");
        }
    }

    #[test]
    fn allowlist_refuses_every_mutating_command() {
        for (p, a) in [
            ("tailscale", vec!["serve", "--bg", "--https=443", "8000"]),
            ("tailscale", vec!["up"]),
            ("tailscale", vec!["set", "--hostname=x"]),
            ("tailscale", vec!["funnel", "443"]),
            ("docker", vec!["compose", "-p", "x", "up", "-d"]),
            ("docker", vec!["compose", "-p", "x", "down"]),
            ("docker", vec!["compose", "-p", "x", "restart"]),
            ("docker", vec!["exec", "supabase-db", "psql"]),
            (
                "docker",
                vec!["exec", "supabase-db", "pg_isready", "&&", "rm"],
            ),
            ("docker", vec!["rm", "-f", "x"]),
            ("nvidia-smi", vec!["-r"]),
            ("nvidia-smi", vec!["-pm", "1"]),
            ("systemctl", vec!["--user", "restart", "x"]),
            ("sh", vec!["install.sh"]),
        ] {
            assert!(!is_read_only(p, &a), "{p} {a:?}");
        }
    }

    #[test]
    fn system_runner_refuses_before_spawning() {
        let err = System::default().run("tailscale", &["up"]).unwrap_err();
        assert!(err.to_string().contains("refusing"));
    }
}
