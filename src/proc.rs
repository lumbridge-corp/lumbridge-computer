//! Process lifecycle: spawn a model in its own process group, track it in a state
//! file, and stop it (SIGTERM → SIGKILL to the whole group). The Governor decides
//! *whether* to start; this module *how*.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::time::Duration;

use crate::config::{Model, ScopesMode};
use crate::scope::Launcher;
use crate::states::ModelState;

/// One Lumbridge Computer-managed process. `seq` is a monotonic launch counter so the watchdog
/// can always find the *newest* model to sacrifice first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proc {
    pub pid: i32,
    pub seq: u64,
    pub port: Option<u16>,
    /// Linux boot id plus process start ticks bind ownership to one process
    /// incarnation, preventing a reused PID from ever being signalled.
    #[serde(default)]
    pub boot_id: Option<String>,
    #[serde(default)]
    pub start_time_ticks: Option<u64>,
    /// Lifecycle state (M3). Absent in state files written before M3, and for
    /// every process a Scene transition starts: both mean `Ready`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<ModelState>,
    /// When `state` last changed, Unix ms. The idle clock falls back to it when
    /// the gateway has seen no request, and `stop_after_sec` counts from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_since_ms: Option<u64>,
    /// For a `docker run` model (M4): the container the CLI at `pid` started,
    /// read from its `--cidfile`. Absent in older state files and when the
    /// cidfile never appeared; such a record is resolved by the `--name` in
    /// the CLI's argv instead (see `docker`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_id: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct State {
    #[serde(default)]
    pub seq: u64,
    #[serde(default)]
    pub procs: BTreeMap<String, Proc>,
    #[serde(default)]
    pub desired_scene: Option<String>,
    #[serde(default)]
    pub active_scene: Option<String>,
    /// Previous proven scene used if the desired scene cannot resume.
    #[serde(default)]
    pub last_known_good_scene: Option<String>,
    #[serde(default)]
    pub transition_scene: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
}

fn state_dir(root: &Path) -> PathBuf {
    root.join(".compute")
}

impl State {
    pub fn load_checked(root: &Path) -> Result<State> {
        let p = state_dir(root).join("state.yaml");
        match fs::read_to_string(&p) {
            Ok(s) => serde_yaml::from_str(&s)
                .with_context(|| format!("parsing persisted state {}", p.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(State::default()),
            Err(error) => Err(error).with_context(|| format!("reading {}", p.display())),
        }
    }

    pub fn save(&self, root: &Path) -> Result<()> {
        let dir = state_dir(root);
        fs::create_dir_all(&dir)?;
        let p = dir.join("state.yaml");
        let tmp = dir.join(format!(".state.yaml.{}.tmp", process::id()));
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        file.write_all(serde_yaml::to_string(self)?.as_bytes())?;
        file.sync_all()?;
        fs::rename(&tmp, &p).with_context(|| format!("atomically replacing {}", p.display()))?;
        File::open(&dir)
            .with_context(|| format!("opening state directory {}", dir.display()))?
            .sync_all()
            .with_context(|| format!("syncing state directory {}", dir.display()))?;
        Ok(())
    }

    /// The model the memory watchdog should stop first: highest `shed_priority`, models
    /// that actually hold memory before 0 GB shims, newest among equals. Without a
    /// registry this is `newest_alive` (the original behaviour).
    #[cfg(test)]
    pub fn shed_candidate(
        &self,
        registry: Option<&crate::config::Registry>,
    ) -> Option<(String, Proc)> {
        self.shed_candidate_by(registry, &BTreeMap::new())
    }

    /// `shed_candidate`, but among equal `shed_priority` the model furthest over
    /// its declared footprint goes first (`overage_gb` from `footprint`): it is
    /// the one whose paperwork let the box get here. Unmeasured models count as
    /// 0 GB over, which keeps the M1 order when nothing is measured.
    pub fn shed_candidate_by(
        &self,
        registry: Option<&crate::config::Registry>,
        overage_gb: &BTreeMap<String, f64>,
    ) -> Option<(String, Proc)> {
        let Some(registry) = registry else {
            return self.newest_alive();
        };
        self.procs
            .iter()
            .filter(|(_, p)| p.owned_alive())
            .max_by_key(|(id, p)| {
                let model = registry.models.get(*id);
                let holds_memory = model.map(|m| m.footprint_gb > 0.0).unwrap_or(true);
                let priority = model.and_then(|m| m.shed_priority).unwrap_or(0);
                // Whole 0.1 GB steps: measurement noise must not reorder equals.
                let over = overage_gb
                    .get(*id)
                    .map_or(0, |gb| (gb.max(0.0) * 10.0) as i64);
                (holds_memory, priority, over, p.seq)
            })
            .map(|(id, p)| (id.clone(), p.clone()))
    }

    pub fn newest_alive(&self) -> Option<(String, Proc)> {
        self.procs
            .iter()
            .filter(|(_, p)| p.owned_alive())
            .max_by_key(|(_, p)| p.seq)
            .map(|(id, p)| (id.clone(), p.clone()))
    }
}

#[derive(Debug)]
struct ProcessIdentity {
    boot_id: String,
    start_time_ticks: u64,
    process_group_id: i32,
    state: char,
}

impl Proc {
    /// Capture the identity of a process that Lumbridge Computer just spawned or
    /// that an operator explicitly adopted from legacy state.
    pub fn capture(pid: i32, seq: u64, port: Option<u16>) -> Result<Proc> {
        let identity = process_identity(pid)?;
        if identity.process_group_id != pid {
            bail!(
                "refusing process {pid}: process group {} does not equal pid",
                identity.process_group_id
            );
        }
        Ok(Proc {
            pid,
            seq,
            port,
            boot_id: Some(identity.boot_id),
            start_time_ticks: Some(identity.start_time_ticks),
            state: None,
            state_since_ms: None,
            container_id: None,
        })
    }

    /// The persisted lifecycle state; a record without one is `Ready`.
    pub fn model_state(&self) -> ModelState {
        self.state.unwrap_or(ModelState::Ready)
    }

    pub fn set_state(&mut self, state: ModelState, now_ms: u64) {
        self.state = Some(state);
        self.state_since_ms = Some(now_ms);
    }

    pub fn owned_alive(&self) -> bool {
        self.validate_owned().is_ok()
    }

    /// True only on positive evidence that the recorded process is gone: no
    /// `/proc` entry, a different incarnation behind the pid, or a zombie.
    /// A legacy record without identity is never reported as exited, because
    /// supervision treats an exit as fatal and must not guess.
    ///
    /// For a `docker run` model the container counts too: a container docker
    /// reports finished (or, by stored id, gone) is an exit even while the CLI
    /// lingers. A docker error is not evidence either way.
    pub fn exited(&self) -> bool {
        let (Some(boot_id), Some(start)) = (self.boot_id.as_deref(), self.start_time_ticks) else {
            return false;
        };
        let cli_exited = match process_identity(self.pid) {
            Ok(current) => {
                current.boot_id != boot_id
                    || current.start_time_ticks != start
                    || current.state == 'Z'
            }
            Err(_) => !Path::new(&format!("/proc/{}", self.pid)).exists(),
        };
        cli_exited || self.container_finished(crate::docker::system(), Path::new("/proc"))
    }

    /// The full id of this record's container: the stored one, else (only
    /// while the CLI is still the exact process Lumbridge Computer started, so a reused
    /// pid can never point us at someone else's container) the one holding
    /// the `--name` in the CLI's argv. `None` for a non-docker model.
    fn container(&self, docker: &dyn crate::docker::Docker, proc_root: &Path) -> Option<String> {
        if let Some(id) = &self.container_id {
            return Some(id.clone());
        }
        if !self.owned_alive() {
            return None;
        }
        let handle = crate::docker::handle(proc_root, None, self.pid)?;
        match crate::docker::resolve(docker, &handle) {
            Ok(id) => id,
            Err(error) => {
                log_docker_once(&format!(
                    "resolving {handle:?} for pid {}: {error:#}",
                    self.pid
                ));
                None
            }
        }
    }

    /// Positive evidence that this record's container is finished. A name
    /// matching no container is *not* evidence: the CLI may still be pulling.
    fn container_finished(&self, docker: &dyn crate::docker::Docker, proc_root: &Path) -> bool {
        let Some(id) = self.container(docker, proc_root) else {
            return false;
        };
        match crate::docker::status(docker, &id) {
            Ok(status) => crate::docker::finished(status.as_deref()),
            Err(error) => {
                log_docker_once(&format!("inspecting container {id}: {error:#}"));
                false
            }
        }
    }

    fn validate_owned(&self) -> Result<()> {
        let expected_boot = self
            .boot_id
            .as_deref()
            .context("legacy process record has no boot identity; adopt the active scene first")?;
        let expected_start = self
            .start_time_ticks
            .context("legacy process record has no start identity; adopt the active scene first")?;
        let current = process_identity(self.pid)?;
        if current.boot_id != expected_boot || current.start_time_ticks != expected_start {
            bail!(
                "pid {} no longer identifies the process Lumbridge Computer started",
                self.pid
            );
        }
        if current.state == 'Z' {
            bail!("pid {} is a zombie awaiting reap", self.pid);
        }
        if current.process_group_id != self.pid {
            bail!(
                "pid {} now belongs to process group {}; refusing group signal",
                self.pid,
                current.process_group_id
            );
        }
        Ok(())
    }
}

/// Log a docker failure once per distinct message: liveness is checked every
/// supervision tick and a down daemon must not flood the journal.
fn log_docker_once(message: &str) {
    static LAST: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    let Ok(mut last) = LAST.lock() else {
        return;
    };
    if last.as_deref() != Some(message) {
        eprintln!("docker: {message}; treating the container as alive");
        *last = Some(message.to_string());
    }
}

fn process_identity(pid: i32) -> Result<ProcessIdentity> {
    let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .context("reading Linux boot id")?
        .trim()
        .to_string();
    let stat_path = format!("/proc/{pid}/stat");
    let stat = fs::read_to_string(&stat_path)
        .with_context(|| format!("reading process identity {stat_path}"))?;
    let (state, process_group_id, start_time_ticks) = parse_stat_identity(&stat)?;
    Ok(ProcessIdentity {
        boot_id,
        start_time_ticks,
        process_group_id,
        state,
    })
}

fn parse_stat_identity(stat: &str) -> Result<(char, i32, u64)> {
    // `comm` is parenthesized and may contain spaces, so locate its final `)`
    // before indexing the fields that follow it. The remainder starts at field 3.
    let close = stat
        .rfind(')')
        .context("malformed /proc stat: missing process name terminator")?;
    let fields: Vec<&str> = stat[close + 1..].split_whitespace().collect();
    let state = fields
        .first()
        .and_then(|value| value.chars().next())
        .context("malformed /proc stat: missing process state")?;
    let process_group_id = fields
        .get(2)
        .context("malformed /proc stat: missing process group")?
        .parse::<i32>()
        .context("parsing process group id")?;
    let start_time_ticks = fields
        .get(19)
        .context("malformed /proc stat: missing start time")?
        .parse::<u64>()
        .context("parsing process start time")?;
    Ok((state, process_group_id, start_time_ticks))
}

/// Expand a leading `~/` to `$HOME`.
fn expand(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    s.to_string()
}

/// The fraction of the pool a vLLM model may reserve, derived from its declared
/// footprint rather than typed separately.
///
/// `footprint_gb` is what the Governor budgets against; `--gpu-memory-utilization` is
/// what actually reserves the memory. Keeping them as two hand-copied numbers means
/// admission control can be arithmetically correct and still wrong about the machine —
/// the registry says a model takes 66 GB while the flag lets it reserve 0.55 of a
/// 121.6 GB pool, which is 67. Deriving one from the other makes the declared footprint
/// the single source of truth.
///
/// An explicit `gpu-memory-utilization` in `serve.args` always wins; this only fills in
/// the gap. Returns `None` when the pool size is unknown, in which case vLLM's own
/// default applies exactly as before.
fn derived_gpu_memory_utilization(m: &Model, mem_total_gb: Option<f64>) -> Option<f64> {
    if m.serve.args.contains_key("gpu-memory-utilization") {
        return None;
    }
    let total = mem_total_gb?;
    if total <= 0.0 || m.footprint_gb <= 0.0 {
        return None;
    }
    let fraction = m.footprint_gb / total;
    // Never hand vLLM a fraction that would reserve the whole box.
    (fraction > 0.0 && fraction < 0.95).then_some((fraction * 1000.0).round() / 1000.0)
}

/// `kind`-based builder. argv[0] is the program.
pub fn build_argv(m: &Model, mem_total_gb: Option<f64>) -> Result<Vec<String>> {
    if !m.serve.command.is_empty() {
        return Ok(m.serve.command.iter().map(|s| expand(s)).collect());
    }
    match m.serve.kind.as_str() {
        "vllm" => {
            let executable = m
                .serve
                .executable
                .as_deref()
                .map(expand)
                .unwrap_or_else(|| "vllm".to_string());
            let mut a = vec![executable, "serve".to_string()];
            if let Some(w) = &m.serve.weights {
                a.push(expand(w));
            }
            if !m.serve.served_name.is_empty() {
                a.push("--served-model-name".into());
                a.extend(m.serve.served_name.iter().cloned());
            }
            // A sleepable engine runs in vLLM's dev mode, which serves
            // `/collective_rpc` (any worker method) to whoever reaches the port:
            // bind it to loopback, so the gateway is the only way in.
            let loopback_only = m
                .serve
                .sleep
                .as_ref()
                .is_some_and(|sleep| !sleep.allow_public_dev_mode);
            a.push("--host".into());
            a.push(
                if loopback_only {
                    "127.0.0.1"
                } else {
                    "0.0.0.0"
                }
                .into(),
            );
            if let Some(p) = m.serve.port {
                a.push("--port".into());
                a.push(p.to_string());
            }
            if let Some(fraction) = derived_gpu_memory_utilization(m, mem_total_gb) {
                a.push("--gpu-memory-utilization".into());
                a.push(format!("{fraction}"));
            }
            // `serve.sleep` is the single switch: the flag follows from it, unless
            // the registry already spelled it out in `args`, and so does the
            // worker extension that reloads target and drafter after a level-2
            // sleep (assets/vllm/lumbridge_computer_sleep_ext.py, on PYTHONPATH via
            // `launch_env`).
            if m.serve.sleep.is_some() {
                if !m.serve.args.contains_key("enable-sleep-mode") {
                    a.push("--enable-sleep-mode".into());
                }
                a.push("--worker-extension-cls".into());
                a.push(crate::config::SLEEP_EXTENSION_CLS.into());
            }
            for (k, v) in &m.serve.args {
                if loopback_only && k == "host" {
                    continue; // the loopback bind above wins over the recipe
                }
                render_arg(&mut a, k, v);
            }
            Ok(a)
        }
        other => bail!(
            "model '{}' has no explicit `command` and kind '{}' has no builder yet — \
             add a `command: [...]` to the registry entry",
            m.name,
            other
        ),
    }
}

fn render_arg(out: &mut Vec<String>, k: &str, v: &serde_yaml::Value) {
    use serde_yaml::Value;
    let flag = format!("--{k}");
    match v {
        Value::Bool(true) => out.push(flag),
        Value::Bool(false) => {}
        Value::String(s) => {
            out.push(flag);
            out.push(s.clone());
        }
        Value::Number(n) => {
            out.push(flag);
            out.push(n.to_string());
        }
        _ => {
            out.push(flag);
            out.push(
                serde_yaml::to_string(v)
                    .unwrap_or_default()
                    .trim()
                    .to_string(),
            );
        }
    }
}

/// The environment a model launches with: `serve.env`, plus, for a model with
/// `serve.sleep`, `VLLM_SERVER_DEV_MODE=1` (what makes vLLM expose `/sleep`,
/// `/wake_up`, `/collective_rpc` and `/is_sleeping`; an explicit value in
/// `serve.env` wins) and the worker extension's directory first on
/// `PYTHONPATH` (before `serve.env`'s, else the inherited one).
pub fn launch_env(m: &Model, root: &Path) -> Result<BTreeMap<String, String>> {
    let mut env = m.serve.env.clone();
    if let Some(sleep) = &m.serve.sleep {
        env.entry("VLLM_SERVER_DEV_MODE".to_string())
            .or_insert_with(|| "1".to_string());
        let dir = sleep_extension_dir(root, sleep)
            .with_context(|| format!("model '{}': serve.sleep", m.name))?;
        let dir = dir.display().to_string();
        let rest = env
            .get("PYTHONPATH")
            .cloned()
            .or_else(|| std::env::var("PYTHONPATH").ok())
            .filter(|rest| !rest.is_empty());
        let path = match rest {
            Some(rest) => format!("{dir}:{rest}"),
            None => dir,
        };
        env.insert("PYTHONPATH".to_string(), path);
    }
    Ok(env)
}

/// The worker extension a launch uses: the directory to put first on
/// `PYTHONPATH` and the class to pass as `--worker-extension-cls`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SleepExtension {
    pub dir: PathBuf,
    pub class: &'static str,
}

/// Where the worker extension lives, as an absolute directory, and which class it is.
///
/// `serve.sleep.extension_dir` if set (`~/` expanded, relative to `root`);
/// otherwise the first of `<root>/assets/vllm`, `<binary dir>/assets/vllm` and
/// `<binary dir>/../share/lumbridge-computer/assets/vllm` that holds
/// `lumbridge_computer_sleep_ext.py`. Missing is an error: without the
/// extension a level-2 wake can't restore the weights.
pub fn sleep_extension(root: &Path, sleep: &crate::config::SleepConfig) -> Result<SleepExtension> {
    use crate::config::{SLEEP_EXTENSION_CLS, SLEEP_EXTENSION_FILE};
    let bin = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let current: Vec<PathBuf> = match &sleep.extension_dir {
        Some(dir) => vec![root.join(expand(dir))],
        None => {
            let mut list = vec![root.join("assets/vllm")];
            if let Some(bin) = &bin {
                list.push(bin.join("assets/vllm"));
                list.push(bin.join("../share/lumbridge-computer/assets/vllm"));
            }
            // Unit tests run from target/debug/deps: use the checkout's copy.
            if cfg!(test) {
                list.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/vllm"));
            }
            list
        }
    };
    let found = |dirs: &[PathBuf], file: &str| -> Result<Option<PathBuf>> {
        for dir in dirs {
            if dir.join(file).is_file() {
                return dir
                    .canonicalize()
                    .map(Some)
                    .with_context(|| format!("resolving {}", dir.display()));
            }
        }
        Ok(None)
    };
    if let Some(dir) = found(&current, SLEEP_EXTENSION_FILE)? {
        return Ok(SleepExtension {
            dir,
            class: SLEEP_EXTENSION_CLS,
        });
    }
    bail!(
        "the vLLM worker extension {SLEEP_EXTENSION_FILE} is not in {}; install assets/vllm \
         or set serve.sleep.extension_dir",
        current
            .iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// The directory half of [`sleep_extension`].
pub fn sleep_extension_dir(root: &Path, sleep: &crate::config::SleepConfig) -> Result<PathBuf> {
    sleep_extension(root, sleep).map(|ext| ext.dir)
}

/// Point `--worker-extension-cls` in a built argv at `class`.
fn use_extension_class(argv: &mut [String], class: &str) {
    if let Some(i) = argv.iter().position(|a| a == "--worker-extension-cls") {
        if let Some(value) = argv.get_mut(i + 1) {
            *value = class.to_string();
        }
    }
}

/// Where `spawn` sends a model's stdout and stderr.
pub fn log_path(root: &Path, id: &str) -> PathBuf {
    state_dir(root).join("logs").join(format!("{id}.log"))
}

/// Spawn `id`'s model detached in its own process group, logs → `.compute/logs/<id>.log`.
/// Returns the child pid (also its process-group id).
///
/// With `scopes: on` the argv is wrapped in `systemd-run --user --scope`, which
/// execs in place: the returned pid is still the model process and its group
/// leader, so ownership, identity and `stop_owned` are unchanged (see `scope`).
pub fn spawn(root: &Path, id: &str, m: &Model, scopes: ScopesMode) -> Result<i32> {
    spawn_with(root, id, m, scopes, default_launcher())
}

#[cfg(not(test))]
fn default_launcher() -> &'static dyn Launcher {
    &crate::scope::SystemdRun
}

/// Unit tests never touch systemd; the scope path is covered by stubs in
/// `scope` and by the `#[ignore]` integration test below.
#[cfg(test)]
fn default_launcher() -> &'static dyn Launcher {
    struct NoSystemd;
    impl Launcher for NoSystemd {
        fn available(&self) -> bool {
            false
        }
        fn command(&self, _: &str, _: &crate::scope::Resolved, argv: &[String]) -> Vec<String> {
            argv.to_vec()
        }
    }
    &NoSystemd
}

pub fn spawn_with(
    root: &Path,
    id: &str,
    m: &Model,
    scopes: ScopesMode,
    launcher: &dyn Launcher,
) -> Result<i32> {
    let mut argv = build_argv(m, crate::mem::read().ok().map(|info| info.total_gb))?;
    if let Some(sleep) = &m.serve.sleep {
        let extension = sleep_extension(root, sleep)
            .with_context(|| format!("model '{}': serve.sleep", m.name))?;
        use_extension_class(&mut argv, extension.class);
    }
    // A `docker run` model writes its container id where the agent can read
    // it (see `docker`). Injected before any scope wrapping.
    let container = crate::docker::launches_container(&argv);
    let argv = if container {
        crate::docker::inject_cidfile(&argv, &crate::docker::cidfile_path(root, id))
    } else {
        argv
    };
    // Kept unwrapped: `clear_for_start` reads `--name` from it.
    let docker_argv = argv.clone();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64 & 0xff_ffff_ffff)
        .unwrap_or_default();
    let (argv, unit) = crate::scope::plan(launcher, scopes, id, m, argv, nonce);
    let path = log_path(root, id);
    if let Some(logdir) = path.parent() {
        fs::create_dir_all(logdir)?;
    }
    // Keep the previous run's log: after a supervised restart it is the only
    // record of why the old engine died (lost for every 2026-11-07 restart).
    if path.exists() {
        let _ = fs::rename(&path, path.with_extension("log.1"));
    }
    let mut log = File::create(&path)?;
    // One line so an operator reading the log knows which cgroup to inspect.
    writeln!(
        log,
        "lumbridge-computer: {}",
        unit.as_deref()
            .map_or("unscoped".to_string(), |u| format!("scope {u}"))
    )?;
    if container {
        crate::docker::clear_for_start(crate::docker::system(), root, id, &docker_argv, &mut log)?;
    } else if m.serve.kind == "docker" {
        writeln!(
            log,
            "lumbridge-computer: kind docker, but the command is not `docker run`; tracking the process only"
        )?;
    }
    let errlog = log.try_clone()?;

    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(errlog))
        .process_group(0); // own group → clean group-kill, immune to CLI's signals
    for (k, v) in launch_env(m, root)? {
        cmd.env(k, expand(&v));
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning '{id}': {}", argv.join(" ")))?;
    let pid = child.id() as i32;
    // A resident agent may outlive many model processes. Reap each child when
    // it exits so failed runtimes cannot accumulate as zombies under the agent.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(pid)
}

/// After `spawn` of a `docker run` model, the container id its CLI wrote to
/// the cidfile: polled for up to `docker::CIDFILE_WAIT` while the CLI lives.
/// `None` for every other model, and when the file never appears (a long image
/// pull): the record then falls back to the `--name`, like a legacy one.
pub fn await_container(root: &Path, id: &str, m: &Model, process: &Proc) -> Option<String> {
    let argv = build_argv(m, None).ok()?;
    if !crate::docker::launches_container(&argv) {
        return None;
    }
    let found = crate::docker::wait_cidfile(
        &crate::docker::cidfile_path(root, id),
        &|| process.owned_alive(),
        crate::docker::CIDFILE_WAIT,
        Duration::from_millis(100),
    );
    if found.is_none() {
        eprintln!(
            "docker: '{id}' wrote no cidfile within {}s; tracking its container by --name",
            crate::docker::CIDFILE_WAIT.as_secs()
        );
    }
    found
}

/// How long the memory watchdog lets a container drain: the same 6 s a
/// process group gets between SIGTERM and SIGKILL, because the floor is
/// breached *now*.
pub const SHED_GRACE: Duration = Duration::from_secs(6);

/// Stop only the exact process group captured in this ownership record, and,
/// for a `docker run` model, its container first (`docker stop`, 30 s grace).
pub fn stop_owned(proc: &Proc) -> Result<()> {
    stop_owned_with(
        proc,
        crate::docker::system(),
        Path::new("/proc"),
        crate::docker::DEFAULT_STOP_GRACE,
    )
}

/// `stop_owned` for the memory watchdog: a container gets `SHED_GRACE`.
pub fn shed_owned(proc: &Proc) -> Result<()> {
    stop_owned_with(
        proc,
        crate::docker::system(),
        Path::new("/proc"),
        SHED_GRACE,
    )
}

/// The container goes first: SIGTERM to the CLI is only proxied, and the
/// SIGKILL after 6 s would hit the CLI alone and leave the container (port,
/// memory, `--name`) behind. Once docker has stopped it the CLI exits on its
/// own, and the group stop below finds nothing or reaps a straggler. A
/// container still present afterwards (docker error, or `--rm` not yet done)
/// is removed.
fn stop_owned_with(
    proc: &Proc,
    docker: &dyn crate::docker::Docker,
    proc_root: &Path,
    grace: Duration,
) -> Result<()> {
    let container = proc.container(docker, proc_root);
    if let Some(id) = &container {
        if let Err(error) = crate::docker::stop(docker, id, grace) {
            eprintln!("docker: stopping container {id}: {error:#}; removing it after the CLI");
        }
    }
    stop_group(proc)?;
    if let Some(id) = &container {
        match crate::docker::status(docker, id) {
            Ok(None) => {}
            Ok(Some(_)) => crate::docker::remove(docker, id)?,
            Err(error) => {
                crate::docker::remove(docker, id).with_context(|| {
                    format!("container {id} may still be running (inspect failed: {error:#})")
                })?;
            }
        }
    }
    Ok(())
}

/// SIGTERM, then SIGKILL, to the recorded process group.
fn stop_group(proc: &Proc) -> Result<()> {
    if !proc.owned_alive() {
        return Ok(());
    }
    proc.validate_owned()?;
    let term = unsafe { libc::kill(-proc.pid, libc::SIGTERM) };
    if term != 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("sending SIGTERM to owned process group {}", proc.pid));
    }
    for _ in 0..60 {
        if !proc.owned_alive() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    proc.validate_owned()?;
    let kill = unsafe { libc::kill(-proc.pid, libc::SIGKILL) };
    if kill != 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("sending SIGKILL to owned process group {}", proc.pid));
    }
    for _ in 0..20 {
        if !proc.owned_alive() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    bail!("owned process group {} survived SIGKILL", proc.pid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exited_needs_positive_evidence_that_the_process_is_gone() {
        let root = std::env::temp_dir().join(format!(
            "lumbridge-computer-exited-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let model: Model = serde_yaml::from_str(
            "name: Sleeper\nfootprint_gb: 1\nserve:\n  kind: exec\n  command: [/usr/bin/sleep, '30']\n",
        )
        .unwrap();
        let pid = spawn(&root, "sleeper", &model, ScopesMode::Off).unwrap();
        let process = Proc::capture(pid, 1, None).unwrap();
        assert!(!process.exited());

        // A legacy record without identity is never reported as exited.
        let legacy = Proc {
            boot_id: None,
            start_time_ticks: None,
            ..process.clone()
        };
        assert!(!legacy.exited());

        stop_owned(&process).unwrap();
        let gone = (0..40).any(|_| {
            std::thread::sleep(Duration::from_millis(50));
            process.exited()
        });
        assert!(gone, "a stopped and reaped process reads as exited");
        assert!(!legacy.exited());
        fs::remove_dir_all(root).unwrap();
    }

    /// Real `systemd-run --user --scope` on a dev box with a user manager.
    /// Run with `cargo test -- --ignored scoped_spawn`.
    #[test]
    #[ignore = "needs a systemd user session"]
    fn scoped_spawn_keeps_the_model_pid_group_and_identity_and_stops_cleanly() {
        let root = std::env::temp_dir().join(format!("lumbridge-computer-scope-{}", process::id()));
        let model: Model = serde_yaml::from_str(
            "name: Sleeper\nfootprint_gb: 1\nserve:\n  kind: exec\n  command: [/bin/sh, -c, 'sleep 60 & exec sleep 61']\n",
        )
        .unwrap();
        assert!(crate::scope::Launcher::available(&crate::scope::SystemdRun));
        let pid = spawn_with(
            &root,
            "it-sleeper",
            &model,
            ScopesMode::On,
            &crate::scope::SystemdRun,
        )
        .unwrap();
        // Never leak the test's sleeps into the user manager on a failed assert.
        struct Reap(i32);
        impl Drop for Reap {
            fn drop(&mut self) {
                unsafe { libc::kill(-self.0, libc::SIGKILL) };
            }
        }
        let _reap = Reap(pid);
        // Captured immediately, while the pid is still `systemd-run`: exec
        // keeps pid, process group and start time, so the identity holds.
        let owned = Proc::capture(pid, 1, None).unwrap();
        let comm = || fs::read_to_string(format!("/proc/{pid}/comm")).unwrap();
        assert!(
            (0..250).any(|_| {
                std::thread::sleep(Duration::from_millis(20));
                comm().trim() == "sleep"
            }),
            "systemd-run exec'd into the model"
        );
        // systemd moves the pid into the scope before systemd-run execs.
        let cgroup = fs::read_to_string(format!("/proc/{pid}/cgroup")).unwrap();
        let cgroup = cgroup.trim().trim_start_matches("0::").to_string();
        assert!(cgroup.ends_with(".scope"), "{cgroup}");
        assert!(cgroup.contains("/lumbridge.slice/lumbridge-computer.slice/lumbridge-computer-models.slice/lumbridge-computer-model-it-sleeper-"));
        assert!(owned.owned_alive(), "same incarnation after exec");
        let dir = Path::new("/sys/fs/cgroup").join(cgroup.trim_start_matches('/'));
        let read = |f: &str| fs::read_to_string(dir.join(f)).unwrap().trim().to_string();
        assert_eq!(read("memory.swap.max"), "0");
        assert_eq!(read("memory.max"), "max");
        // The kernel rounds limits down to a page.
        let high: u64 = read("memory.high").parse().unwrap();
        let asked = (1.15f64 * 1073741824.0).round() as u64;
        assert!(asked - high < 65536, "memory.high {high} for {asked}");
        // Both sleeps (leader and backgrounded child) share the group and scope.
        assert_eq!(read("cgroup.procs").lines().count(), 2);

        stop_owned(&owned).unwrap();
        assert!(
            (0..100).any(|_| {
                std::thread::sleep(Duration::from_millis(20));
                !dir.exists()
            }),
            "the empty scope is collected"
        );
        assert!(owned.exited());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn among_equal_priority_the_model_furthest_over_its_footprint_is_shed_first() {
        let root = std::env::temp_dir().join(format!("lumbridge-computer-shed-{}", process::id()));
        let registry = crate::config::Registry::parse(
            "apiVersion: lumbridge/v1\nmodels:\n  embed:\n    name: e\n    footprint_gb: 3\n    serve: {kind: exec, command: [/usr/bin/sleep, '30']}\n  rerank:\n    name: r\n    footprint_gb: 5\n    serve: {kind: exec, command: [/usr/bin/sleep, '30']}\n  toy:\n    name: t\n    footprint_gb: 1\n    shed_priority: 50\n    serve: {kind: exec, command: [/usr/bin/sleep, '30']}\n",
        )
        .unwrap();
        let mut state = State::default();
        for (seq, id) in ["embed", "rerank", "toy"].into_iter().enumerate() {
            let pid = spawn(&root, id, &registry.models[id], ScopesMode::Off).unwrap();
            state
                .procs
                .insert(id.into(), Proc::capture(pid, seq as u64, None).unwrap());
        }
        let pick = |state: &State, over: &[(&str, f64)]| {
            let over = over.iter().map(|(k, v)| (k.to_string(), *v)).collect();
            state.shed_candidate_by(Some(&registry), &over).unwrap().0
        };
        // Priority still rules: the toy goes first however over the others are.
        assert_eq!(pick(&state, &[("embed", 7.5)]), "toy");
        let toy = state.procs.remove("toy").unwrap();
        stop_owned(&toy).unwrap();
        // Nothing measured: newest (M1 order). Measured: the over-budget one.
        assert_eq!(pick(&state, &[]), "rerank");
        assert_eq!(state.shed_candidate(Some(&registry)).unwrap().0, "rerank");
        assert_eq!(pick(&state, &[("embed", 7.5), ("rerank", 1.9)]), "embed");
        // Noise below 0.1 GB does not override recency.
        assert_eq!(pick(&state, &[("embed", 0.05)]), "rerank");
        for process in state.procs.values() {
            stop_owned(process).unwrap();
        }
        let _ = fs::remove_dir_all(root);
    }

    /// Build a Model the way the registry does — through serde — so these tests exercise
    /// the real deserialization path rather than a hand-assembled struct.
    fn vllm_model(footprint_gb: f64, extra_args: &str) -> Model {
        let yaml = format!(
            "name: brain\nfootprint_gb: {footprint_gb}\nserve:\n  kind: vllm\n  port: 8001\n  args:\n    max-model-len: 8192\n{extra_args}"
        );
        serde_yaml::from_str(&yaml).expect("test model yaml")
    }

    #[test]
    fn gpu_memory_utilization_is_derived_from_the_declared_footprint() {
        // 66 GB of a 121.6 GB pool is 0.543 — not the 0.55 that used to be typed in
        // separately, which is the whole point: one number, not two that can drift.
        let m = vllm_model(66.0, "");
        assert_eq!(derived_gpu_memory_utilization(&m, Some(121.6)), Some(0.543));
        let argv = build_argv(&m, Some(121.6)).unwrap();
        let i = argv
            .iter()
            .position(|a| a == "--gpu-memory-utilization")
            .unwrap();
        assert_eq!(argv[i + 1], "0.543");
    }

    #[test]
    fn an_explicit_flag_in_the_registry_always_wins() {
        let m = vllm_model(66.0, "    gpu-memory-utilization: 0.8\n");
        assert_eq!(derived_gpu_memory_utilization(&m, Some(121.6)), None);
        // ...and it is rendered exactly once, from serve.args.
        let argv = build_argv(&m, Some(121.6)).unwrap();
        assert_eq!(
            argv.iter()
                .filter(|a| *a == "--gpu-memory-utilization")
                .count(),
            1
        );
        let i = argv
            .iter()
            .position(|a| a == "--gpu-memory-utilization")
            .unwrap();
        assert_eq!(argv[i + 1], "0.8");
    }

    #[test]
    fn unknown_pool_size_leaves_vllms_own_default_alone() {
        let m = vllm_model(66.0, "");
        assert_eq!(derived_gpu_memory_utilization(&m, None), None);
        assert!(!build_argv(&m, None)
            .unwrap()
            .iter()
            .any(|a| a == "--gpu-memory-utilization"));
    }

    #[test]
    fn a_footprint_that_would_claim_the_whole_box_is_not_derived() {
        // Better to let vLLM apply its own default than to hand it 0.99 and wedge the box.
        let m = vllm_model(120.0, "");
        assert_eq!(derived_gpu_memory_utilization(&m, Some(121.6)), None);
    }

    #[test]
    fn nonsense_inputs_do_not_produce_a_flag() {
        assert_eq!(
            derived_gpu_memory_utilization(&vllm_model(0.0, ""), Some(121.6)),
            None
        );
        assert_eq!(
            derived_gpu_memory_utilization(&vllm_model(66.0, ""), Some(0.0)),
            None
        );
    }

    #[test]
    fn an_explicit_command_bypasses_the_builder_entirely() {
        let mut m = vllm_model(66.0, "");
        m.serve.command = vec!["python".into(), "-m".into(), "server".into()];
        assert_eq!(build_argv(&m, Some(121.6)).unwrap(), m.serve.command);
    }

    #[test]
    fn proc_stat_parser_handles_spaces_in_process_name() {
        let stat = "123 (worker process) S 1 123 123 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 4567";
        let (state, pgrp, start) = parse_stat_identity(stat).unwrap();
        assert_eq!(state, 'S');
        assert_eq!(pgrp, 123);
        assert_eq!(start, 4567);
    }

    #[test]
    fn legacy_records_are_never_treated_as_owned() {
        let proc = Proc {
            pid: std::process::id() as i32,
            seq: 1,
            port: None,
            boot_id: None,
            start_time_ticks: None,
            state: None,
            state_since_ms: None,
            container_id: None,
        };
        assert!(!proc.owned_alive());
    }

    fn flag_value<'a>(argv: &'a [String], flag: &str) -> Vec<&'a str> {
        argv.windows(2)
            .filter(|w| w[0] == flag)
            .map(|w| w[1].as_str())
            .collect()
    }

    /// A Lumbridge Computer root holding a copy of the worker extension in `assets/vllm`.
    fn root_with_extension() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "lumbridge-computer-ext-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("assets/vllm")).unwrap();
        fs::write(
            root.join("assets/vllm")
                .join(crate::config::SLEEP_EXTENSION_FILE),
            "# test copy\n",
        )
        .unwrap();
        root
    }

    #[test]
    fn a_sleep_block_adds_sleep_mode_the_worker_extension_and_dev_mode_env() {
        let root = root_with_extension();
        let mut m = vllm_model(10.0, "");
        let argv = build_argv(&m, None).unwrap();
        assert!(!argv.contains(&"--enable-sleep-mode".to_string()));
        assert!(!argv.contains(&"--worker-extension-cls".to_string()));
        assert!(!launch_env(&m, &root)
            .unwrap()
            .contains_key("VLLM_SERVER_DEV_MODE"));
        m.serve.sleep = Some(serde_yaml::from_str("level: 2").unwrap());
        let argv = build_argv(&m, None).unwrap();
        assert_eq!(
            flag_value(&argv, "--worker-extension-cls"),
            [crate::config::SLEEP_EXTENSION_CLS]
        );
        assert_eq!(
            argv.iter().filter(|a| *a == "--enable-sleep-mode").count(),
            1
        );
        let env = launch_env(&m, &root).unwrap();
        assert_eq!(env["VLLM_SERVER_DEV_MODE"], "1");
        let ext = root.join("assets/vllm").canonicalize().unwrap();
        assert_eq!(
            env["PYTHONPATH"].split(':').next(),
            Some(ext.to_str().unwrap()),
            "the extension directory comes first and is absolute"
        );
        // Spelled out in args: still exactly once.
        m.serve
            .args
            .insert("enable-sleep-mode".into(), serde_yaml::Value::Bool(true));
        let argv = build_argv(&m, None).unwrap();
        assert_eq!(
            argv.iter().filter(|a| *a == "--enable-sleep-mode").count(),
            1
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_extension_dir_is_prepended_to_the_registry_pythonpath_and_can_be_overridden() {
        let root = root_with_extension();
        let elsewhere = root.join("box/vllm-ext");
        fs::create_dir_all(&elsewhere).unwrap();
        fs::write(elsewhere.join(crate::config::SLEEP_EXTENSION_FILE), "#\n").unwrap();
        let mut m = vllm_model(10.0, "");
        m.serve.env.insert("PYTHONPATH".into(), "/opt/site".into());
        m.serve.sleep =
            Some(serde_yaml::from_str("{level: 2, extension_dir: box/vllm-ext}").unwrap());
        let env = launch_env(&m, &root).unwrap();
        assert_eq!(
            env["PYTHONPATH"],
            format!("{}:/opt/site", elsewhere.canonicalize().unwrap().display())
        );

        m.serve.sleep =
            Some(serde_yaml::from_str("{level: 2, extension_dir: /nonexistent/ext}").unwrap());
        let error = format!("{:#}", launch_env(&m, &root).unwrap_err());
        assert!(error.contains("lumbridge_computer_sleep_ext.py"), "{error}");
        assert!(error.contains("/nonexistent/ext"), "{error}");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_shipped_extension_defines_the_class_the_launch_names() {
        let file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/vllm")
            .join(crate::config::SLEEP_EXTENSION_FILE);
        let text = fs::read_to_string(file).unwrap();
        let (module, class) = crate::config::SLEEP_EXTENSION_CLS.split_once('.').unwrap();
        assert_eq!(format!("{module}.py"), crate::config::SLEEP_EXTENSION_FILE);
        assert!(text.contains(&format!("class {class}:")));
        // The agent calls it by name on models launched with either class.
        assert!(text.contains("def reload_all_fast("));
    }

    #[test]
    fn a_sleepable_engine_is_bound_to_loopback_whatever_the_recipe_says() {
        // Brain recipes often bind 0.0.0.0; dev mode would expose
        // /collective_rpc to the network.
        let mut m = vllm_model(10.0, "    host: 0.0.0.0\n");
        assert_eq!(
            flag_value(&build_argv(&m, None).unwrap(), "--host"),
            ["0.0.0.0", "0.0.0.0"]
        );
        m.serve.sleep = Some(serde_yaml::from_str("level: 2").unwrap());
        assert_eq!(
            flag_value(&build_argv(&m, None).unwrap(), "--host"),
            ["127.0.0.1"]
        );

        // The explicit escape hatch keeps the recipe's bind (and doctor warns).
        m.serve.sleep =
            Some(serde_yaml::from_str("{level: 2, allow_public_dev_mode: true}").unwrap());
        let argv = build_argv(&m, None).unwrap();
        assert_eq!(flag_value(&argv, "--host").last(), Some(&"0.0.0.0"));
    }

    const CID: &str = "31c245bd349d50351b56efd802d73d2eba1f05263f0b36a6a6c055af8d1da986";

    fn temp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lumbridge-computer-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A real `sleep` standing in for the docker CLI.
    fn fake_cli(root: &Path) -> Proc {
        let model: Model = serde_yaml::from_str(
            "name: cli\nfootprint_gb: 1\nserve:\n  kind: exec\n  command: [/usr/bin/sleep, '30']\n",
        )
        .unwrap();
        let pid = spawn(root, "cli", &model, ScopesMode::Off).unwrap();
        Proc::capture(pid, 1, None).unwrap()
    }

    #[test]
    fn a_container_is_stopped_with_docker_before_its_cli_and_removed_if_it_lingers() {
        use crate::docker::fake::FakeDocker;
        let root = temp("dstop");
        let mut cli = fake_cli(&root);
        cli.container_id = Some(CID.into());
        let docker = FakeDocker::default();
        docker
            .on("stop", Ok(""))
            .on("inspect -f {{.State.Status}}", Ok("removing\n"))
            .on("rm -f", Ok(""));
        stop_owned_with(&cli, &docker, Path::new("/proc"), Duration::from_secs(30)).unwrap();
        assert!(!cli.owned_alive(), "the CLI's group was stopped too");
        assert_eq!(
            docker.calls(),
            [
                format!("stop -t 30 {CID}"),
                format!("inspect -f {{{{.State.Status}}}} {CID}"),
                format!("rm -f {CID}"),
            ]
        );

        // Gone after the stop (`--rm` did its job): no rm. The CLI is already
        // dead here, and the container is still stopped by its stored id.
        let docker = FakeDocker::default();
        docker.on("stop", Ok(""));
        stop_owned_with(&cli, &docker, Path::new("/proc"), SHED_GRACE).unwrap();
        assert_eq!(
            docker.calls(),
            [
                format!("stop -t 6 {CID}"),
                format!("inspect -f {{{{.State.Status}}}} {CID}"),
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_legacy_docker_record_is_stopped_by_the_name_in_its_cli_argv() {
        use crate::docker::fake::FakeDocker;
        let root = temp("dlegacy");
        let cli = fake_cli(&root);
        // The CLI's argv, as /proc would show it for a containerised embed.
        let fake_proc = root.join("proc");
        fs::create_dir_all(fake_proc.join(cli.pid.to_string())).unwrap();
        fs::write(
            fake_proc.join(cli.pid.to_string()).join("cmdline"),
            "docker\0run\0--rm\0--init\0--name=embed\0vllm/vllm-openai:v0.33.0\0",
        )
        .unwrap();
        let docker = FakeDocker::default();
        docker
            .on("ps -aq --no-trunc -f name=^embed$", Ok(&format!("{CID}\n")))
            .on("stop", Ok(""));
        stop_owned_with(&cli, &docker, &fake_proc, Duration::from_secs(30)).unwrap();
        assert_eq!(
            docker.calls(),
            [
                "ps -aq --no-trunc -f name=^embed$".to_string(),
                format!("stop -t 30 {CID}"),
                format!("inspect -f {{{{.State.Status}}}} {CID}"),
            ]
        );
        // Once the CLI is gone a legacy record names nothing: a reused pid
        // must never lead to someone else's container.
        let docker = FakeDocker::default();
        assert_eq!(cli.container(&docker, &fake_proc), None);
        assert!(docker.calls().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_non_docker_stop_never_calls_docker() {
        use crate::docker::fake::FakeDocker;
        let root = temp("dnone");
        let cli = fake_cli(&root);
        let docker = FakeDocker::default();
        stop_owned_with(&cli, &docker, Path::new("/proc"), Duration::from_secs(30)).unwrap();
        assert!(docker.calls().is_empty());
        assert!(!cli.owned_alive());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_container_reported_finished_is_an_exit_but_a_docker_error_is_not() {
        use crate::docker::fake::FakeDocker;
        let root = temp("dexit");
        let mut cli = fake_cli(&root);
        cli.container_id = Some(CID.into());
        let proc_root = Path::new("/proc");
        let docker = FakeDocker::default();
        docker.on("inspect", Ok("running\n"));
        assert!(!cli.container_finished(&docker, proc_root));
        let docker = FakeDocker::default();
        docker.on("inspect", Ok("created\n"));
        assert!(
            !cli.container_finished(&docker, proc_root),
            "cidfile precedes start"
        );
        let docker = FakeDocker::default();
        docker.on("inspect", Ok("exited\n"));
        assert!(cli.container_finished(&docker, proc_root));
        let docker = FakeDocker::default();
        assert!(
            cli.container_finished(&docker, proc_root),
            "no such container"
        );
        let docker = FakeDocker::default();
        docker.on("inspect", Err("Cannot connect to the Docker daemon"));
        assert!(!cli.container_finished(&docker, proc_root));
        // The unit-test docker always errors, so `exited` follows the CLI.
        assert!(!cli.exited());
        stop_group(&cli).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_docker_model_gets_a_cidfile_and_a_container_id_from_it() {
        let root = temp("dspawn");
        // `docker` here is a stub script; the real CLI writes the cidfile.
        let bin = root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        let stub = bin.join("docker");
        fs::write(
            &stub,
            format!("#!/bin/sh\n[ \"$2\" = --cidfile ] && printf {CID} > \"$3\"\nexec sleep 30\n"),
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).unwrap();
        let model: Model = serde_yaml::from_str(&format!(
            "name: c\nfootprint_gb: 1\nserve:\n  kind: docker\n  command: [{}, run, --rm, img]\n",
            stub.display()
        ))
        .unwrap();
        let pid = spawn(&root, "c", &model, ScopesMode::Off).unwrap();
        let mut process = Proc::capture(pid, 1, None).unwrap();
        process.container_id = await_container(&root, "c", &model, &process);
        assert_eq!(process.container_id.as_deref(), Some(CID));
        let log = fs::read_to_string(log_path(&root, "c")).unwrap();
        assert!(log.starts_with("lumbridge-computer: unscoped\n"), "{log}");
        stop_group(&process).unwrap();
        // Non-docker: nothing to wait for.
        let sleeper = fake_cli(&root);
        let plain: Model = serde_yaml::from_str(
            "name: p\nfootprint_gb: 1\nserve:\n  kind: exec\n  command: [/usr/bin/sleep, '30']\n",
        )
        .unwrap();
        assert_eq!(await_container(&root, "p", &plain, &sleeper), None);
        stop_group(&sleeper).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_state_file_without_container_ids_reads_and_round_trips_unchanged() {
        let old = "seq: 2\nprocs:\n  embed:\n    pid: 1533243\n    seq: 2\n    port: 8012\n    boot_id: abc\n    start_time_ticks: 99\n";
        let state: State = serde_yaml::from_str(old).unwrap();
        assert_eq!(state.procs["embed"].container_id, None);
        assert!(!serde_yaml::to_string(&state)
            .unwrap()
            .contains("container_id"));
        let mut state = state;
        state.procs.get_mut("embed").unwrap().container_id = Some(CID.into());
        let back: State = serde_yaml::from_str(&serde_yaml::to_string(&state).unwrap()).unwrap();
        assert_eq!(back.procs["embed"].container_id.as_deref(), Some(CID));
    }

    #[test]
    fn a_state_file_from_before_m3_reads_every_process_as_ready() {
        let old = "seq: 7\nprocs:\n  brain:\n    pid: 4242\n    seq: 7\n    port: 8001\n    boot_id: abc\n    start_time_ticks: 99\ndesired_scene: everything\nactive_scene: everything\n";
        let state: State = serde_yaml::from_str(old).unwrap();
        let brain = &state.procs["brain"];
        assert_eq!(brain.state, None);
        assert_eq!(brain.model_state(), ModelState::Ready);
        // Re-saved without a state change, the file has no new keys, so a
        // pre-M3 binary reading it back sees exactly what it wrote.
        let again = serde_yaml::to_string(&state).unwrap();
        assert!(!again.contains("state:"), "{again}");
        assert!(!again.contains("state_since_ms"), "{again}");

        let mut state = state;
        state
            .procs
            .get_mut("brain")
            .unwrap()
            .set_state(ModelState::Sleeping { level: 2 }, 1_000);
        let text = serde_yaml::to_string(&state).unwrap();
        let back: State = serde_yaml::from_str(&text).unwrap();
        assert_eq!(
            back.procs["brain"].model_state(),
            ModelState::Sleeping { level: 2 }
        );
        assert_eq!(back.procs["brain"].state_since_ms, Some(1_000));
    }
}
