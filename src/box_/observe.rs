//! Reading reality: one pass over the host that doctor and the dry-run planner share.
//!
//! Every probe stores `Err(reason)` instead of aborting, so one broken subsystem
//! (Tailscale logged out, Docker down) still leaves a full report for the rest.

use super::cmd::Cmd;
use super::config::BoxConfig;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub type Probe<T> = Result<T, String>;

/// Lumbridge Computer's "GB" is GiB throughout (/proc/meminfo kB / 1024², cgroup bytes / 1024³),
/// so a reservation equal to the sum of `mem_limit`s compares exactly.
pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn probe<T>(r: anyhow::Result<T>) -> Probe<T> {
    r.map_err(|e| format!("{e:#}"))
}

#[derive(Debug, Clone)]
pub struct Observed {
    pub hardware: Hardware,
    pub disk: Probe<Disk>,
    pub mem: Probe<Mem>,
    /// Every process with pages in swap (`VmSwap` > 0), largest first.
    pub swap: Probe<Vec<SwapHolder>>,
    pub docker: Probe<()>,
    pub supabase: Supabase,
    pub tailscale: Tailscale,
    pub models: Models,
}

#[derive(Debug, Clone)]
pub struct Hardware {
    pub arch: Probe<String>,
    pub os: Probe<String>,
    pub gpu: Probe<Gpu>,
    pub cuda: Probe<String>,
    pub cgroup_v2: bool,
    pub user: Option<String>,
    pub linger: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gpu {
    pub name: String,
    pub driver: String,
    /// Raw `memory.total` as nvidia-smi prints it; `[N/A]` on GB10.
    pub memory_total: String,
}

#[derive(Debug, Clone)]
pub struct Disk {
    pub path: PathBuf,
    pub mount: String,
    pub size_bytes: u64,
    pub avail_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct Mem {
    pub total_gb: f64,
    pub available_gb: f64,
    pub swap_total_gb: f64,
    pub swap_free_gb: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwapHolder {
    pub pid: u32,
    pub name: String,
    /// cgroup v2 path from /proc/<pid>/cgroup, e.g. `/system.slice/docker.service`.
    pub cgroup: String,
    pub swap_kb: u64,
}

#[derive(Debug, Clone)]
pub struct Supabase {
    /// `Ok(None)`: Docker answered and the project does not exist.
    pub project: Probe<Option<ComposeProject>>,
    pub containers: Probe<Vec<Container>>,
    pub pg_isready: Probe<String>,
    /// What the deployment says it was built from (`.supabase-version`, then
    /// then `.lumbridge-computer-upstream`).
    pub stamp: Option<Stamp>,
    /// Every stamp file found, so doctor can flag stale ones.
    pub all_stamps: Vec<Stamp>,
    pub compose_services: Probe<BTreeMap<String, ComposeService>>,
    pub overlay_present: bool,
    pub env: Probe<EnvFacts>,
    pub slice_memory_gb: Option<f64>,
    /// `memory.min`/`memory.low` on the slice and its ancestors, and Lumbridge Computer's drop-ins.
    pub floor: super::floor::FloorFacts,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComposeProject {
    pub name: String,
    pub status: String,
    pub config_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Container {
    pub id: String,
    pub name: String,
    pub service: String,
    pub image: String,
    pub state: String,
    pub health: Option<String>,
    pub cgroup_parent: String,
    pub oom_score_adj: i64,
    pub memory_bytes: i64,
    pub bindings: Vec<Binding>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub container_port: String,
    pub host_ip: String,
    pub host_port: String,
}

impl Binding {
    /// Docker writes `""`, `0.0.0.0` or `::` for "every interface".
    pub fn is_loopback(&self) -> bool {
        matches!(self.host_ip.as_str(), "127.0.0.1" | "::1" | "[::1]")
            || self.host_ip.starts_with("127.")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stamp {
    pub file: String,
    pub ref_: String,
    pub commit: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComposeService {
    pub image: Option<String>,
    /// Services behind a compose profile are not expected to run by default.
    pub has_profile: bool,
}

/// Key names only. Values are dropped while parsing and never stored.
#[derive(Debug, Clone)]
pub struct EnvFacts {
    pub keys: BTreeSet<String>,
    pub mode: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Tailscale {
    pub status: Probe<TsStatus>,
    pub serve: Probe<Serve>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TsStatus {
    pub backend_state: String,
    pub hostname: String,
    pub dns_name: String,
    pub magic_dns_suffix: String,
    pub health: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Serve {
    pub routes: Vec<ServedRoute>,
    /// `host:port` entries with Funnel enabled.
    pub funnel: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServedRoute {
    pub host: String,
    pub port: u16,
    pub path: String,
    pub https: bool,
    /// `Some(url)` for a reverse proxy; `None` for file/text/TCP handlers.
    pub proxy: Option<String>,
    pub describe: String,
}

#[derive(Debug, Clone)]
pub struct Models {
    pub registry: Probe<RegistryFacts>,
    pub scenes: Probe<Vec<SceneFacts>>,
    pub state: Probe<SceneState>,
    /// The planner's verdict for the desired Scene: validation, preflight and the diff.
    pub plan: Probe<PlanFacts>,
    /// Engines that run (or may run) in vLLM dev mode, and where they listen.
    pub dev_mode: Probe<Vec<DevModeFacts>>,
}

/// One model that is, or by its registry entry would be, a vLLM in dev mode
/// (`VLLM_SERVER_DEV_MODE=1`), which serves `/collective_rpc`, `/sleep` and
/// `/wake_up` to anyone who can reach its port.
#[derive(Debug, Clone, PartialEq)]
pub struct DevModeFacts {
    pub id: String,
    /// Lumbridge Computer owns a process for it (a record in `.compute/state.yaml`).
    pub running: bool,
    /// Dev mode is on: from the running process's environment when readable,
    /// else from the registry (`serve.sleep`, or `serve.env`).
    pub dev_mode: bool,
    /// `serve.sleep.allow_public_dev_mode: true` in the registry.
    pub allow_public: bool,
    pub port: Option<u16>,
    /// Local addresses listening on `port` (`/proc/net/tcp`, `/proc/net/tcp6`).
    pub listening: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegistryFacts {
    pub models: usize,
    pub reservations: Vec<ReservationFacts>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReservationFacts {
    pub id: String,
    pub gb: f64,
    pub cgroup: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneFacts {
    pub name: String,
    pub budget_gb: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SceneState {
    pub active: Option<String>,
    pub desired: Option<String>,
    pub transition: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanFacts {
    pub budget_gb: f64,
    pub stop: Vec<String>,
    pub start: Vec<String>,
}

/// The `docker inspect` template. It names exactly the fields the checks use, so
/// container environments (which hold Supabase secrets) are never read at all.
pub const INSPECT_FORMAT: &str = concat!(
    r#"{"id":{{json .Id}},"name":{{json .Name}},"image":{{json .Config.Image}},"#,
    r#""service":{{json (index .Config.Labels "com.docker.compose.service")}},"#,
    r#""cgroup_parent":{{json .HostConfig.CgroupParent}},"oom_score_adj":{{json .HostConfig.OomScoreAdj}},"#,
    r#""memory":{{json .HostConfig.Memory}},"port_bindings":{{json .HostConfig.PortBindings}},"#,
    r#""ports":{{json .NetworkSettings.Ports}},"state":{{json .State.Status}},"#,
    r#""health":{{if .State.Health}}{{json .State.Health.Status}}{{else}}null{{end}}}"#
);

/// Supabase `.env` keys a working stack needs. Checked by name only.
pub const REQUIRED_ENV_KEYS: &[&str] = &[
    "POSTGRES_PASSWORD",
    "JWT_SECRET",
    "ANON_KEY",
    "SERVICE_ROLE_KEY",
    "DASHBOARD_PASSWORD",
    "SECRET_KEY_BASE",
    "VAULT_ENC_KEY",
    "PG_META_CRYPTO_KEY",
    "SUPABASE_PUBLIC_URL",
    "API_EXTERNAL_URL",
    "COMPOSE_FILE",
];

pub fn gather(cmd: &dyn Cmd, config: &BoxConfig) -> Observed {
    let docker = probe(cmd.run("docker", &["ps", "-q"]).map(|_| ()));
    Observed {
        hardware: gather_hardware(cmd),
        disk: probe(read_disk(cmd, &config.models.root)),
        mem: probe(
            cmd.read(Path::new("/proc/meminfo"))
                .map(|s| parse_meminfo(&s)),
        ),
        swap: probe(read_swap_holders(cmd)),
        supabase: gather_supabase(cmd, config, docker.is_ok()),
        docker,
        tailscale: Tailscale {
            status: probe(
                cmd.run("tailscale", &["status", "--json"])
                    .and_then(|s| parse_ts_status(&s)),
            ),
            serve: probe(
                cmd.run("tailscale", &["serve", "status", "--json"])
                    .and_then(|s| parse_serve(&s)),
            ),
        },
        models: gather_models(cmd, &config.models.root, &config.models.desired_scene),
    }
}

fn gather_hardware(cmd: &dyn Cmd) -> Hardware {
    let user = std::env::var("USER").ok().filter(|u| !u.is_empty());
    Hardware {
        arch: probe(cmd.run("uname", &["-m"]).map(|s| s.trim().to_string())),
        os: probe(
            cmd.read(Path::new("/etc/os-release"))
                .map(|s| parse_os_release(&s)),
        ),
        gpu: probe(
            cmd.run(
                "nvidia-smi",
                &[
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
            )
            .and_then(|s| parse_gpu(&s)),
        ),
        cuda: probe(cmd.run("nvidia-smi", &[]).and_then(|s| parse_cuda(&s))),
        cgroup_v2: cmd.exists(Path::new("/sys/fs/cgroup/cgroup.controllers")),
        linger: user
            .as_deref()
            .is_some_and(|u| cmd.exists(&Path::new("/var/lib/systemd/linger").join(u))),
        user,
    }
}

fn read_disk(cmd: &dyn Cmd, path: &Path) -> anyhow::Result<Disk> {
    let p = path.to_string_lossy();
    let out = cmd.run("df", &["-P", "-B1", &p])?;
    parse_df(&out, path)
}

fn gather_supabase(cmd: &dyn Cmd, config: &BoxConfig, docker_up: bool) -> Supabase {
    let s = &config.supabase;
    let project = if docker_up {
        probe(
            cmd.run("docker", &["compose", "ls", "--format", "json"])
                .and_then(|out| parse_compose_ls(&out, &s.project)),
        )
    } else {
        Err("docker is not answering".to_string())
    };
    let containers = match &project {
        Ok(Some(_)) => probe(read_containers(cmd, &s.project)),
        Ok(None) => Ok(Vec::new()),
        Err(e) => Err(e.clone()),
    };
    let pg_isready = match &containers {
        Ok(list) => match list.iter().find(|c| c.service == "db") {
            Some(db) => probe(cmd.run("docker", &["exec", &db.name, "pg_isready"])),
            None => Err("no db container".into()),
        },
        Err(e) => Err(e.clone()),
    };
    let mut all_stamps = Vec::new();
    let stamps = [".supabase-version", ".lumbridge-computer-upstream"];
    for file in stamps {
        if let Ok(text) = cmd.read(&s.dir.join(file)) {
            if let Some(stamp) = parse_stamp(file, &text) {
                all_stamps.push(stamp);
            }
        }
    }
    let env_path = s.dir.join(".env");
    Supabase {
        project,
        containers,
        pg_isready,
        stamp: all_stamps.first().cloned(),
        all_stamps,
        compose_services: probe(
            cmd.read(&s.dir.join("docker-compose.yml"))
                .and_then(|t| parse_compose_services(&t)),
        ),
        overlay_present: cmd.exists(&s.dir.join(&s.overlay)),
        env: probe(cmd.read(&env_path).map(|text| EnvFacts {
            keys: env_keys(&text),
            mode: cmd.mode(&env_path),
        })),
        slice_memory_gb: cmd
            .read(
                &Path::new("/sys/fs/cgroup")
                    .join(config.slice_cgroup())
                    .join("memory.current"),
            )
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .map(|b| b / GIB),
        floor: super::floor::observe(cmd, Path::new("/"), config, effective_uid()),
    }
}

/// Sum `VmSwap` per process from /proc/<pid>/status. Processes that exit between the
/// listing and the read are skipped; kernel threads have no VmSwap line.
fn read_swap_holders(cmd: &dyn Cmd) -> anyhow::Result<Vec<SwapHolder>> {
    let proc = Path::new("/proc");
    let mut out = Vec::new();
    for entry in cmd.list_dir(proc)? {
        let Ok(pid) = entry.parse::<u32>() else {
            continue;
        };
        let Ok(status) = cmd.read(&proc.join(&entry).join("status")) else {
            continue;
        };
        let Some((name, swap_kb)) = parse_status_swap(&status) else {
            continue;
        };
        if swap_kb == 0 {
            continue;
        }
        let cgroup = cmd
            .read(&proc.join(&entry).join("cgroup"))
            .map(|t| parse_proc_cgroup(&t))
            .unwrap_or_default();
        out.push(SwapHolder {
            pid,
            name,
            cgroup,
            swap_kb,
        });
    }
    out.sort_by(|a, b| b.swap_kb.cmp(&a.swap_kb).then(a.pid.cmp(&b.pid)));
    Ok(out)
}

/// `Name:` and `VmSwap:` (kB) from /proc/<pid>/status.
pub fn parse_status_swap(text: &str) -> Option<(String, u64)> {
    let mut name = None;
    let mut swap = None;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("Name:") {
            name = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("VmSwap:") {
            swap = v.split_whitespace().next().and_then(|n| n.parse().ok());
        }
    }
    Some((name?, swap?))
}

/// The unified-hierarchy line (`0::/path`) of /proc/<pid>/cgroup.
pub fn parse_proc_cgroup(text: &str) -> String {
    text.lines()
        .find_map(|l| l.strip_prefix("0::"))
        .unwrap_or("")
        .trim()
        .to_string()
}

pub fn effective_uid() -> u32 {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() }
}

fn read_containers(cmd: &dyn Cmd, project: &str) -> anyhow::Result<Vec<Container>> {
    let ps = cmd.run(
        "docker",
        &["compose", "-p", project, "ps", "--format", "json"],
    )?;
    let ids: Vec<String> = json_records(&ps)?
        .iter()
        .filter_map(|v| v.get("ID").and_then(Value::as_str).map(str::to_string))
        .collect();
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut args = vec!["inspect", "--format", INSPECT_FORMAT];
    args.extend(ids.iter().map(String::as_str));
    parse_inspect(&cmd.run("docker", &args)?)
}

/// Lifecycle owns the Scene logic; these are its read-only entry points. Registry and
/// scenes are files under the configured root, so they're read directly, not via `Cmd`.
pub fn gather_models(cmd: &dyn Cmd, root: &Path, desired: &str) -> Models {
    use crate::config::{load_scenes, Registry};
    Models {
        dev_mode: probe(gather_dev_mode(cmd, root)),
        registry: probe(Registry::load(root).map(|r| {
            RegistryFacts {
                models: r.models.len(),
                reservations: r
                    .reservations
                    .iter()
                    .map(|(id, res)| ReservationFacts {
                        id: id.clone(),
                        gb: res.gb,
                        cgroup: res.cgroup.clone(),
                    })
                    .collect(),
            }
        })),
        scenes: probe(load_scenes(root).map(|list| {
            list.iter()
                .map(|s| SceneFacts {
                    name: s.metadata.name.clone(),
                    budget_gb: s.budget_gb,
                })
                .collect()
        })),
        state: probe(crate::proc::State::load_checked(root).map(|s| SceneState {
            active: s.active_scene,
            desired: s.desired_scene,
            transition: s.transition_scene,
        })),
        plan: probe(
            crate::lifecycle::plan_read_only(root, desired).map(|p| PlanFacts {
                budget_gb: p.budget_gb,
                stop: p.stop,
                start: p.start,
            }),
        ),
    }
}

/// Every owned process that runs vLLM dev mode (or whose registry entry asks
/// for it), and every registry entry that allows a public dev-mode bind.
fn gather_dev_mode(cmd: &dyn Cmd, root: &Path) -> anyhow::Result<Vec<DevModeFacts>> {
    let registry = crate::config::Registry::load(root)?;
    let state = crate::proc::State::load_checked(root)?;
    let mut listeners = Vec::new();
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        if let Ok(text) = cmd.read(Path::new(table)) {
            listeners.extend(parse_listeners(&text));
        }
    }
    let mut facts = Vec::new();
    for (id, model) in &registry.models {
        let allow_public = model
            .serve
            .sleep
            .as_ref()
            .is_some_and(|sleep| sleep.allow_public_dev_mode);
        let declared = model.serve.sleep.is_some()
            || model
                .serve
                .env
                .get("VLLM_SERVER_DEV_MODE")
                .map(String::as_str)
                == Some("1");
        let process = state.procs.get(id);
        let environ = process.and_then(|p| {
            cmd.read(&Path::new("/proc").join(p.pid.to_string()).join("environ"))
                .ok()
        });
        let dev_mode = match &environ {
            Some(environ) => environ.split('\0').any(|kv| kv == "VLLM_SERVER_DEV_MODE=1"),
            None => declared,
        };
        if !(dev_mode || declared || allow_public) {
            continue;
        }
        let port = process.and_then(|p| p.port).or(model.serve.port);
        facts.push(DevModeFacts {
            id: id.clone(),
            running: process.is_some(),
            dev_mode,
            allow_public,
            port,
            listening: listeners
                .iter()
                .filter(|(_, p)| Some(*p) == port)
                .map(|(addr, _)| addr.clone())
                .collect(),
        });
    }
    Ok(facts)
}

/// `(address, port)` of every LISTEN socket in a `/proc/net/tcp{,6}` table.
pub fn parse_listeners(text: &str) -> Vec<(String, u16)> {
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let local = fields.nth(1)?;
            let state = fields.nth(1)?;
            if state != "0A" {
                return None;
            }
            let (addr, port) = local.split_once(':')?;
            let port = u16::from_str_radix(port, 16).ok()?;
            Some((proc_net_addr(addr)?, port))
        })
        .collect()
}

/// `/proc/net` writes addresses as host-order 32-bit words in hex.
fn proc_net_addr(hex: &str) -> Option<String> {
    let words: Vec<u32> = (0..hex.len() / 8)
        .map(|i| u32::from_str_radix(&hex[i * 8..i * 8 + 8], 16).ok())
        .collect::<Option<_>>()?;
    match words.as_slice() {
        [v4] => Some(std::net::Ipv4Addr::from(v4.to_le_bytes()).to_string()),
        [a, b, c, d] => {
            let mut bytes = [0u8; 16];
            for (i, word) in [a, b, c, d].iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            Some(std::net::Ipv6Addr::from(bytes).to_string())
        }
        _ => None,
    }
}

/// True for 127.0.0.0/8, ::1 and IPv4-mapped loopback.
pub fn is_loopback_addr(addr: &str) -> bool {
    match addr.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v4)) => v4.is_loopback(),
        Ok(std::net::IpAddr::V6(v6)) => {
            v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
        Err(_) => false,
    }
}

// ---- parsers -----------------------------------------------------------------

pub fn parse_meminfo(text: &str) -> Mem {
    let kb = |key: &str| -> f64 {
        text.lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|v| v.split_whitespace().next())
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            / 1024.0
            / 1024.0
    };
    Mem {
        total_gb: kb("MemTotal:"),
        available_gb: kb("MemAvailable:"),
        swap_total_gb: kb("SwapTotal:"),
        swap_free_gb: kb("SwapFree:"),
    }
}

pub fn parse_os_release(text: &str) -> String {
    let get = |key: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(key))
            .map(|v| v.trim_matches('"').to_string())
            .unwrap_or_default()
    };
    format!("{}-{}", get("ID="), get("VERSION_ID="))
}

pub fn parse_gpu(text: &str) -> anyhow::Result<Gpu> {
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("nvidia-smi listed no GPU"))?;
    let f: Vec<&str> = line.split(',').map(str::trim).collect();
    if f.len() < 3 {
        anyhow::bail!("unexpected nvidia-smi output: {line}");
    }
    Ok(Gpu {
        name: f[0].to_string(),
        driver: f[1].to_string(),
        memory_total: f[2].to_string(),
    })
}

pub fn parse_cuda(text: &str) -> anyhow::Result<String> {
    text.split("CUDA Version:")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("nvidia-smi did not report a CUDA version"))
}

pub fn parse_df(text: &str, path: &Path) -> anyhow::Result<Disk> {
    let line = text
        .lines()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("df printed no filesystem"))?;
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.len() < 6 {
        anyhow::bail!("unexpected df output: {line}");
    }
    Ok(Disk {
        path: path.to_path_buf(),
        size_bytes: f[1].parse()?,
        avail_bytes: f[3].parse()?,
        mount: f[5].to_string(),
    })
}

/// compose prints a JSON array (older releases) or one object per line (v2.21+).
fn json_records(text: &str) -> anyhow::Result<Vec<Value>> {
    let t = text.trim();
    if t.is_empty() {
        return Ok(Vec::new());
    }
    if t.starts_with('[') {
        return Ok(serde_json::from_str(t)?);
    }
    t.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}

pub fn parse_compose_ls(text: &str, project: &str) -> anyhow::Result<Option<ComposeProject>> {
    Ok(json_records(text)?.into_iter().find_map(|v| {
        let name = v.get("Name")?.as_str()?;
        (name == project).then(|| ComposeProject {
            name: name.to_string(),
            status: v.get("Status").and_then(Value::as_str).unwrap_or("").into(),
            config_files: v
                .get("ConfigFiles")
                .and_then(Value::as_str)
                .unwrap_or("")
                .split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect(),
        })
    }))
}

#[derive(Deserialize)]
struct InspectRow {
    id: String,
    name: String,
    image: String,
    #[serde(default)]
    service: Option<String>,
    #[serde(default)]
    cgroup_parent: String,
    #[serde(default)]
    oom_score_adj: i64,
    #[serde(default)]
    memory: i64,
    #[serde(default)]
    port_bindings: Option<BTreeMap<String, Option<Vec<RawBinding>>>>,
    #[serde(default)]
    ports: Option<BTreeMap<String, Option<Vec<RawBinding>>>>,
    state: String,
    #[serde(default)]
    health: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct RawBinding {
    #[serde(default)]
    host_ip: String,
    #[serde(default)]
    host_port: String,
}

pub fn parse_inspect(text: &str) -> anyhow::Result<Vec<Container>> {
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: InspectRow = serde_json::from_str(line)?;
        // Live bindings (`ports`) and requested ones (`port_bindings`) together: a
        // container that asked for 0.0.0.0 is a finding even if it isn't running.
        let mut bindings = Vec::new();
        for map in [row.ports.as_ref(), row.port_bindings.as_ref()]
            .into_iter()
            .flatten()
        {
            for (cport, list) in map {
                for b in list.iter().flatten() {
                    let binding = Binding {
                        container_port: cport.clone(),
                        host_ip: b.host_ip.clone(),
                        host_port: b.host_port.clone(),
                    };
                    if !bindings.contains(&binding) {
                        bindings.push(binding);
                    }
                }
            }
        }
        out.push(Container {
            id: row.id,
            name: row.name.trim_start_matches('/').to_string(),
            service: row.service.unwrap_or_default(),
            image: row.image,
            state: row.state,
            health: row.health.filter(|h| !h.is_empty()),
            cgroup_parent: row.cgroup_parent,
            oom_score_adj: row.oom_score_adj,
            memory_bytes: row.memory,
            bindings,
        });
    }
    out.sort_by(|a, b| a.service.cmp(&b.service));
    Ok(out)
}

/// `ref=self-hosted/vX` (update.sh) or `self-hosted/vX <commit>` (install.sh).
pub fn parse_stamp(file: &str, text: &str) -> Option<Stamp> {
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(r) = line.strip_prefix("ref=") {
            return Some(Stamp {
                file: file.into(),
                ref_: r.trim().into(),
                commit: None,
            });
        }
        let mut parts = line.split_whitespace();
        let ref_ = parts.next()?.to_string();
        return Some(Stamp {
            file: file.into(),
            ref_,
            commit: parts.next().map(str::to_string),
        });
    }
    None
}

pub fn parse_compose_services(text: &str) -> anyhow::Result<BTreeMap<String, ComposeService>> {
    let doc: serde_yaml::Value = serde_yaml::from_str(text)?;
    let services = doc
        .get("services")
        .and_then(|s| s.as_mapping())
        .ok_or_else(|| anyhow::anyhow!("docker-compose.yml has no services"))?;
    Ok(services
        .iter()
        .filter_map(|(k, v)| {
            Some((
                k.as_str()?.to_string(),
                ComposeService {
                    image: v.get("image").and_then(|i| i.as_str()).map(str::to_string),
                    has_profile: v.get("profiles").is_some(),
                },
            ))
        })
        .collect())
}

pub fn env_keys(text: &str) -> BTreeSet<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once('=').map(|(k, _)| k.trim()))
        .filter(|k| !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
        .map(str::to_string)
        .collect()
}

pub fn parse_ts_status(text: &str) -> anyhow::Result<TsStatus> {
    let v: Value = serde_json::from_str(text)?;
    let s = |v: Option<&Value>| v.and_then(Value::as_str).unwrap_or("").to_string();
    Ok(TsStatus {
        backend_state: s(v.get("BackendState")),
        hostname: s(v.pointer("/Self/HostName")),
        dns_name: s(v.pointer("/Self/DNSName")).trim_end_matches('.').into(),
        magic_dns_suffix: s(v.get("MagicDNSSuffix")),
        health: v
            .get("Health")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
    })
}

pub fn parse_serve(text: &str) -> anyhow::Result<Serve> {
    let v: Value = serde_json::from_str(text)?;
    let mut serve = Serve::default();
    let tcp = v.get("TCP").and_then(Value::as_object);
    if let Some(web) = v.get("Web").and_then(Value::as_object) {
        for (hostport, conf) in web {
            let (host, port) = hostport
                .rsplit_once(':')
                .and_then(|(h, p)| Some((h.to_string(), p.parse::<u16>().ok()?)))
                .ok_or_else(|| anyhow::anyhow!("unexpected serve key {hostport}"))?;
            let https = tcp
                .and_then(|t| t.get(&port.to_string()))
                .and_then(|t| t.get("HTTPS"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let handlers = conf.get("Handlers").and_then(Value::as_object);
            for (path, h) in handlers.into_iter().flatten() {
                let proxy = h.get("Proxy").and_then(Value::as_str).map(str::to_string);
                let describe = match &proxy {
                    Some(p) => p.clone(),
                    None if h.get("Path").is_some() => {
                        format!("files {}", h["Path"].as_str().unwrap_or("?"))
                    }
                    None if h.get("Text").is_some() => "static text".into(),
                    None => "unknown handler".into(),
                };
                serve.routes.push(ServedRoute {
                    host: host.clone(),
                    port,
                    path: path.clone(),
                    https,
                    proxy,
                    describe,
                });
            }
        }
    }
    // Raw TCP forwards have no Web entry; they're served too and must be accounted for.
    for (port, t) in tcp.into_iter().flatten() {
        if let Some(fwd) = t.get("TCPForward").and_then(Value::as_str) {
            serve.routes.push(ServedRoute {
                host: String::new(),
                port: port.parse().unwrap_or(0),
                path: String::new(),
                https: false,
                proxy: None,
                describe: format!("tcp forward to {fwd}"),
            });
        }
    }
    if let Some(funnel) = v.get("AllowFunnel").and_then(Value::as_object) {
        serve.funnel = funnel
            .iter()
            .filter(|(_, on)| on.as_bool().unwrap_or(false))
            .map(|(k, _)| k.clone())
            .collect();
    }
    serve
        .routes
        .sort_by(|a, b| (a.port, &a.path).cmp(&(b.port, &b.path)));
    Ok(serve)
}

#[cfg(test)]
pub(crate) mod fixtures {
    //! A made-up box, `box1`, in the exact shape real tools print (the inspect
    //! fixture uses `INSPECT_FORMAT`, the env fixture has keys only).
    use super::super::cmd::Fake;
    use super::INSPECT_FORMAT;

    pub const TS_STATUS: &str = include_str!("fixtures/box1/tailscale-status.json");
    pub const TS_SERVE: &str = include_str!("fixtures/box1/tailscale-serve.json");
    pub const COMPOSE_LS: &str = include_str!("fixtures/box1/compose-ls.json");
    pub const COMPOSE_PS: &str = include_str!("fixtures/box1/compose-ps.ndjson");
    pub const INSPECT: &str = include_str!("fixtures/box1/docker-inspect.ndjson");
    pub const MEMINFO: &str = include_str!("fixtures/box1/meminfo");
    pub const GPU: &str = include_str!("fixtures/box1/nvidia-smi-query.csv");
    pub const NVSMI: &str = include_str!("fixtures/box1/nvidia-smi.txt");
    pub const DF: &str = include_str!("fixtures/box1/df.txt");
    pub const OS_RELEASE: &str = include_str!("fixtures/box1/os-release");
    pub const COMPOSE_YML: &str = include_str!("fixtures/box1/docker-compose.yml");
    pub const PG_ISREADY: &str = include_str!("fixtures/box1/pg_isready.txt");
    pub const SUPABASE_VERSION: &str = include_str!("fixtures/box1/supabase-version");
    pub const LUMBRIDGE_UPSTREAM: &str = include_str!("fixtures/box1/lumbridge-computer-upstream");
    pub const ENV: &str = include_str!("fixtures/box1/env");

    pub const SB_DIR: &str = "/home/sam/supabase";

    const AGENT: &str =
        "/user.slice/user-1000.slice/user@1000.service/app.slice/lumbridge-computer-agent.service";
    /// The largest VmSwap holders on the fixture box (pid, name, kB, cgroup).
    pub const SWAP_PROCS: &[(u32, &str, u64, &str)] = &[
        (19499, "nemo-speech", 4076272, AGENT),
        (32896, "VLLM::EngineCor", 1641696, AGENT),
        (18466, "python", 1436400, AGENT),
        (31529, "VLLM::EngineCor", 1417768, AGENT),
        (32513, "vllm", 805024, AGENT),
        (30618, "vllm", 731948, AGENT),
        (
            3083,
            "gnome-shell",
            26236,
            "/user.slice/user-126.slice/session-c1.scope",
        ),
        (
            728325,
            ".postgres-wrapp",
            41920,
            "/lumbridge.slice/lumbridge-server.slice/docker-b34850dc80fd.scope",
        ),
        (
            2533,
            "systemd",
            0,
            "/user.slice/user-1000.slice/user@1000.service/init.scope",
        ),
    ];

    /// A fake host that answers as the fixture box does.
    pub fn box1() -> Fake {
        let mut f = Fake::default();
        f.on("uname -m", "aarch64\n")
            .on("tailscale status --json", TS_STATUS)
            .on("tailscale serve status --json", TS_SERVE)
            .on("docker ps -q", "f8c59ad6b759\n")
            .on("docker compose ls --format json", COMPOSE_LS)
            .on(
                "docker compose -p lumbridge-supabase ps --format json",
                COMPOSE_PS,
            )
            .on(
                &format!("docker inspect --format {INSPECT_FORMAT}"),
                INSPECT,
            )
            .on("docker exec supabase-db pg_isready", PG_ISREADY)
            .on("nvidia-smi --query-gpu", GPU)
            .on("nvidia-smi", NVSMI)
            .on("df -P -B1", DF)
            .file("/proc/meminfo", MEMINFO)
            .file("/etc/os-release", OS_RELEASE)
            .file(
                "/sys/fs/cgroup/cgroup.controllers",
                "cpuset cpu io memory pids\n",
            )
            .file(
                "/sys/fs/cgroup/lumbridge.slice/lumbridge-server.slice/memory.current",
                "4245000192\n",
            )
            // No floor anywhere on 2026-11-08 (systemctl show: MemoryMin=0, MemoryLow=0).
            .file("/sys/fs/cgroup/lumbridge.slice/memory.min", "0\n")
            .file("/sys/fs/cgroup/lumbridge.slice/memory.low", "0\n")
            .file(
                "/sys/fs/cgroup/lumbridge.slice/lumbridge-server.slice/memory.min",
                "0\n",
            )
            .file(
                "/sys/fs/cgroup/lumbridge.slice/lumbridge-server.slice/memory.low",
                "0\n",
            )
            .file(&format!("{SB_DIR}/docker-compose.yml"), COMPOSE_YML)
            .file(
                &format!("{SB_DIR}/lumbridge.override.yml"),
                "services: {}\n",
            )
            .file(&format!("{SB_DIR}/.supabase-version"), SUPABASE_VERSION)
            .file(
                &format!("{SB_DIR}/.lumbridge-computer-upstream"),
                LUMBRIDGE_UPSTREAM,
            )
            .file(&format!("{SB_DIR}/.env"), ENV);
        if let Ok(user) = std::env::var("USER") {
            f.file(&format!("/var/lib/systemd/linger/{user}"), "");
        }
        for (pid, name, kb, cgroup) in SWAP_PROCS {
            f.file(
                &format!("/proc/{pid}/status"),
                &format!("Name:\t{name}\nState:\tS (sleeping)\nVmSwap:\t{kb:>8} kB\n"),
            )
            .file(&format!("/proc/{pid}/cgroup"), &format!("0::{cgroup}\n"));
        }
        // A kernel thread: no VmSwap line at all.
        f.file("/proc/2/status", "Name:\tkthreadd\nState:\tS (sleeping)\n");
        f
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    #[test]
    fn box1_serve_status_parses_into_four_https_proxy_routes() {
        let s = parse_serve(TS_SERVE).unwrap();
        assert_eq!(s.routes.len(), 4);
        assert!(s.funnel.is_empty());
        let r = s.routes.iter().find(|r| r.port == 8443).unwrap();
        assert_eq!(r.proxy.as_deref(), Some("http://127.0.0.1:8000"));
        assert!(r.https);
        assert_eq!(r.host, "box1.tail0000.ts.net");
    }

    #[test]
    fn funnel_and_tcp_forwards_are_detected() {
        let text = r#"{"TCP":{"443":{"HTTPS":true},"2222":{"TCPForward":"127.0.0.1:22"}},
            "Web":{"h.ts.net:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:1"}}}},
            "AllowFunnel":{"h.ts.net:443":true}}"#;
        let s = parse_serve(text).unwrap();
        assert_eq!(s.funnel, vec!["h.ts.net:443"]);
        assert!(s.routes.iter().any(|r| r.port == 2222 && r.proxy.is_none()));
    }

    #[test]
    fn box1_tailscale_status_reports_identity() {
        let s = parse_ts_status(TS_STATUS).unwrap();
        assert_eq!(s.backend_state, "Running");
        assert_eq!(s.hostname, "box1");
        assert_eq!(s.dns_name, "box1.tail0000.ts.net");
        assert_eq!(s.magic_dns_suffix, "tail0000.ts.net");
    }

    #[test]
    fn box1_inspect_rows_carry_slice_oom_and_loopback_ports() {
        let c = parse_inspect(INSPECT).unwrap();
        assert_eq!(c.len(), 11);
        let db = c.iter().find(|c| c.service == "db").unwrap();
        assert_eq!(db.name, "supabase-db");
        assert_eq!(db.oom_score_adj, -900);
        assert_eq!(db.cgroup_parent, "lumbridge-server.slice");
        let gw = c.iter().find(|c| c.service == "api-gw").unwrap();
        assert_eq!(gw.bindings.len(), 1);
        assert!(gw.bindings[0].is_loopback());
    }

    #[test]
    fn an_empty_host_ip_means_every_interface() {
        let b = Binding {
            container_port: "8000/tcp".into(),
            host_ip: String::new(),
            host_port: "8000".into(),
        };
        assert!(!b.is_loopback());
    }

    #[test]
    fn compose_ls_finds_only_the_named_project() {
        let p = parse_compose_ls(COMPOSE_LS, "lumbridge-supabase")
            .unwrap()
            .unwrap();
        assert_eq!(p.config_files.len(), 2);
        assert!(parse_compose_ls(COMPOSE_LS, "other").unwrap().is_none());
        assert!(parse_compose_ls("[]", "x").unwrap().is_none());
    }

    #[test]
    fn stamps_parse_both_update_and_install_formats() {
        let s = parse_stamp(".supabase-version", SUPABASE_VERSION).unwrap();
        assert_eq!(s.ref_, "self-hosted/v0.8.4");
        assert_eq!(s.commit, None);
        let l = parse_stamp(".lumbridge-computer-upstream", LUMBRIDGE_UPSTREAM).unwrap();
        assert_eq!(l.ref_, "self-hosted/v0.8.2");
        assert_eq!(
            l.commit.as_deref(),
            Some("564eab8ad7840b13324f68b1bfac074ef8d51c21")
        );
    }

    #[test]
    fn compose_file_images_are_read_per_service() {
        let s = parse_compose_services(COMPOSE_YML).unwrap();
        assert_eq!(
            s["db"].image.as_deref(),
            Some("supabase/postgres:17.6.1.136")
        );
        assert_eq!(s.values().filter(|s| !s.has_profile).count(), 11);
    }

    #[test]
    fn env_parsing_keeps_names_and_drops_values() {
        let keys = env_keys("# c\nJWT_SECRET=hunter2\nPOSTGRES_PASSWORD = x=y\n\nbad line\n");
        assert_eq!(
            keys.into_iter().collect::<Vec<_>>(),
            vec!["JWT_SECRET", "POSTGRES_PASSWORD"]
        );
    }

    #[test]
    fn host_probes_parse_box1_outputs() {
        let m = parse_meminfo(MEMINFO);
        assert!(m.total_gb > 120.0 && m.swap_total_gb > 15.0);
        assert_eq!(parse_os_release(OS_RELEASE), "ubuntu-24.04");
        let g = parse_gpu(GPU).unwrap();
        assert_eq!(g.name, "NVIDIA GB10");
        assert_eq!(g.memory_total, "[N/A]");
        assert_eq!(parse_cuda(NVSMI).unwrap(), "13.0");
        let d = parse_df(DF, Path::new("/x")).unwrap();
        assert!(d.avail_bytes > 1_000_000_000_000);
    }

    #[test]
    fn models_are_read_through_the_registry_and_planner_without_writing() {
        let root = std::env::temp_dir().join(format!(
            "lumbridge-computer-box-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(root.join("registry")).unwrap();
        std::fs::create_dir_all(root.join("scenes")).unwrap();
        std::fs::write(
            root.join("registry/models.yaml"),
            "apiVersion: lumbridge/v1\nreservations:\n  server:\n    gb: 8.5\n    cgroup: lumbridge-computer.slice\nmodels:\n  idle:\n    name: Idle\n    footprint_gb: 1\n    serve:\n      kind: exec\n      command: [/bin/true]\n",
        )
        .unwrap();
        std::fs::write(
            root.join("scenes/empty.scene.yaml"),
            "apiVersion: lumbridge/v1\nmetadata:\n  name: empty\nmodels: []\nbudget_gb: 50\n",
        )
        .unwrap();
        let m = gather_models(&crate::box_::cmd::Fake::default(), &root, "empty");
        assert_eq!(m.registry.as_ref().unwrap().reservations[0].gb, 8.5);
        assert_eq!(m.scenes.as_ref().unwrap()[0].name, "empty");
        let plan = m.plan.unwrap();
        assert!(plan.stop.is_empty() && plan.start.is_empty());
        assert_eq!(m.state.unwrap(), SceneState::default());
        assert!(m.dev_mode.unwrap().is_empty());
        assert!(
            gather_models(&crate::box_::cmd::Fake::default(), &root, "missing")
                .plan
                .is_err()
        );
        // The read-only planner must not create the transition lock or state dir.
        assert!(!root.join(".compute").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn swap_holders_are_read_per_process_largest_first_skipping_kernel_threads() {
        let config =
            BoxConfig::parse(super::super::config::EXAMPLE, Some(Path::new("/home/sam"))).unwrap();
        let swap = gather(&box1(), &config).swap.unwrap();
        assert_eq!(swap.len(), 8, "{swap:#?}");
        assert_eq!(swap[0].name, "nemo-speech");
        assert_eq!(swap[0].pid, 19499);
        assert!(swap[0].cgroup.ends_with("lumbridge-computer-agent.service"));
        assert!(swap.windows(2).all(|w| w[0].swap_kb >= w[1].swap_kb));
        assert_eq!(parse_status_swap("Name:\tx\n"), None);
        assert_eq!(parse_proc_cgroup("0::/a/b\n"), "/a/b");
    }

    #[test]
    fn a_failing_docker_daemon_degrades_every_supabase_probe_without_panicking() {
        let config =
            BoxConfig::parse(super::super::config::EXAMPLE, Some(Path::new("/home/sam"))).unwrap();
        let mut host = box1();
        host.fail("docker", "Cannot connect to the Docker daemon");
        let o = gather(&host, &config);
        assert!(o.docker.is_err());
        assert!(o.supabase.project.is_err());
        assert!(o.supabase.containers.is_err());
        assert!(o.tailscale.status.is_ok());
    }

    #[test]
    fn listeners_are_read_from_proc_net_tcp_in_host_byte_order() {
        let tcp = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n\
   0: 0100007F:1F41 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 1 1 0\n\
   1: 00000000:1F41 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 2 1 0\n\
   2: 070200C0:1F69 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 3 1 0\n\
   3: 0100007F:1F41 0100007F:D2F0 01 00000000:00000000 00:00000000 00000000  1000        0 4 1 0\n";
        assert_eq!(
            parse_listeners(tcp),
            [
                ("127.0.0.1".to_string(), 8001),
                ("0.0.0.0".to_string(), 8001),
                ("192.0.2.7".to_string(), 8041),
            ]
        );
        let tcp6 = "  sl  local_address                         remote_address                        st\n\
   0: 00000000000000000000000001000000:1F41 00000000000000000000000000000000:0000 0A 0\n\
   1: 00000000000000000000000000000000:1F4B 00000000000000000000000000000000:0000 0A 0\n";
        assert_eq!(
            parse_listeners(tcp6),
            [("::1".to_string(), 8001), ("::".to_string(), 8011)]
        );
        assert!(is_loopback_addr("127.0.0.1") && is_loopback_addr("::1"));
        assert!(is_loopback_addr("::ffff:127.0.0.1"));
        assert!(!is_loopback_addr("0.0.0.0") && !is_loopback_addr("::"));
        assert!(!is_loopback_addr("192.0.2.7"));
    }

    #[test]
    fn dev_mode_facts_come_from_the_process_environment_and_its_port() {
        let root = std::env::temp_dir().join(format!(
            "lumbridge-computer-devmode-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(root.join("registry")).unwrap();
        std::fs::create_dir_all(root.join(".compute")).unwrap();
        std::fs::write(
            root.join("registry/models.yaml"),
            "apiVersion: lumbridge/v1\nmodels:\n  brain:\n    name: b\n    footprint_gb: 40\n    serve: {kind: vllm, port: 8001}\n  sleepy:\n    name: s\n    footprint_gb: 4\n    serve: {kind: vllm, port: 8002, sleep: {allow_public_dev_mode: true}}\n  plain:\n    name: p\n    footprint_gb: 1\n    serve: {kind: exec, port: 8003, command: [/bin/true]}\n",
        )
        .unwrap();
        std::fs::write(
            root.join(".compute/state.yaml"),
            "procs:\n  brain: {pid: 4242, seq: 1, port: 8001}\n  plain: {pid: 4343, seq: 2, port: 8003}\n",
        )
        .unwrap();
        let mut host = crate::box_::cmd::Fake::default();
        host.file("/proc/4242/environ", "HOME=/home/sam\0VLLM_SERVER_DEV_MODE=1\0")
            .file("/proc/4343/environ", "HOME=/home/sam\0")
            .file(
                "/proc/net/tcp",
                "  sl  local_address rem_address   st\n   0: 00000000:1F41 00000000:0000 0A 0\n   1: 00000000:1F43 00000000:0000 0A 0\n",
            );
        let facts = gather_dev_mode(&host, &root).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
        assert_eq!(
            facts,
            [
                // Dev mode set by hand in a running engine: seen in its environment.
                DevModeFacts {
                    id: "brain".into(),
                    running: true,
                    dev_mode: true,
                    allow_public: false,
                    port: Some(8001),
                    listening: vec!["0.0.0.0".into()],
                },
                // Not running: known from the registry only.
                DevModeFacts {
                    id: "sleepy".into(),
                    running: false,
                    dev_mode: true,
                    allow_public: true,
                    port: Some(8002),
                    listening: vec![],
                },
            ]
        );
    }
}
