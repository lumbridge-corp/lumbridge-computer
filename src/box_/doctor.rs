//! `lumbridge-computer doctor`: read-only verdicts on the box, one line each.
//!
//! Pure functions of (`lumbridge-computer.toml`, [`Observed`]); nothing here touches the host.

use super::config::{normalize_proxy, BoxConfig};
use super::floor::{self, FloorFacts, Level};
use super::observe::{
    is_loopback_addr, Container, DevModeFacts, Observed, Probe, SwapHolder, GIB, REQUIRED_ENV_KEYS,
};
use serde::Serialize;

/// Free space under which the models filesystem is a warning / a failure.
pub const DISK_WARN_GB: f64 = 100.0;
pub const DISK_FAIL_GB: f64 = 20.0;
/// Share of swap in use that is worth a warning on a unified-memory box.
pub const SWAP_WARN_SHARE: f64 = 0.5;
/// Slice use above this share of its reservation is a warning.
pub const SLICE_WARN_SHARE: f64 = 0.9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub id: String,
    pub status: Status,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

struct Report(Vec<Check>);

impl Report {
    fn push(&mut self, id: &str, status: Status, reason: impl Into<String>, hint: Option<&str>) {
        self.0.push(Check {
            id: id.into(),
            status,
            reason: reason.into(),
            hint: hint.map(str::to_string),
        });
    }
    fn ok(&mut self, id: &str, reason: impl Into<String>) {
        self.push(id, Status::Ok, reason, None);
    }
    fn warn(&mut self, id: &str, reason: impl Into<String>, hint: &str) {
        self.push(id, Status::Warn, reason, Some(hint));
    }
    fn fail(&mut self, id: &str, reason: impl Into<String>, hint: &str) {
        self.push(id, Status::Fail, reason, Some(hint));
    }
    /// A probe that could not run is a failure of the check that needed it.
    fn probe<'a, T>(&mut self, id: &str, p: &'a Probe<T>, hint: &str) -> Option<&'a T> {
        match p {
            Ok(v) => Some(v),
            Err(e) => {
                self.fail(id, e.clone(), hint);
                None
            }
        }
    }
}

pub fn run(c: &BoxConfig, o: &Observed) -> Vec<Check> {
    let mut r = Report(Vec::new());
    hardware(&mut r, c, o);
    disk_and_memory(&mut r, c, o);
    supabase(&mut r, c, o);
    tailscale(&mut r, c, o);
    models(&mut r, c, o);
    if let Some(facts) = r.probe(
        "models.dev_mode",
        &o.models.dev_mode,
        "fix the registry and .compute/state.yaml",
    ) {
        dev_mode(&mut r, facts);
    }
    r.0
}

/// vLLM's dev mode (`VLLM_SERVER_DEV_MODE=1`, needed for sleep) serves
/// `/collective_rpc` (any worker method), `/sleep` and `/wake_up` to whoever
/// reaches the port. A running dev-mode engine listening anywhere but loopback
/// fails; the registry's `allow_public_dev_mode` escape hatch warns.
fn dev_mode(r: &mut Report, facts: &[DevModeFacts]) {
    let exposed: Vec<String> = facts
        .iter()
        .filter(|f| f.running && f.dev_mode)
        .filter_map(|f| {
            let public: Vec<&str> = f
                .listening
                .iter()
                .map(String::as_str)
                .filter(|addr| !is_loopback_addr(addr))
                .collect();
            (!public.is_empty()).then(|| {
                format!(
                    "{} listens on {}:{}",
                    f.id,
                    public.join(","),
                    f.port.unwrap_or(0)
                )
            })
        })
        .collect();
    let allowed: Vec<&str> = facts
        .iter()
        .filter(|f| f.allow_public)
        .map(|f| f.id.as_str())
        .collect();
    if !exposed.is_empty() {
        r.fail(
            "models.dev_mode",
            format!(
                "vLLM dev mode reachable off-box ({}): /collective_rpc runs any worker method",
                exposed.join("; ")
            ),
            "bind it to 127.0.0.1 (serve.sleep does, unless allow_public_dev_mode) and restart it; \
             clients go through the gateway",
        );
    } else if !allowed.is_empty() {
        r.warn(
            "models.dev_mode",
            format!(
                "serve.sleep.allow_public_dev_mode is set for {}: a dev-mode engine would keep a \
                 public bind",
                allowed.join(", ")
            ),
            "remove allow_public_dev_mode; clients go through the gateway on :8011",
        );
    } else {
        let running = facts.iter().filter(|f| f.running && f.dev_mode).count();
        r.ok(
            "models.dev_mode",
            format!("{running} dev-mode engine(s) running, all on loopback"),
        );
    }
}

fn hardware(r: &mut Report, c: &BoxConfig, o: &Observed) {
    let h = &o.hardware;
    if let Some(arch) = r.probe("hardware.arch", &h.arch, "uname -m must work") {
        if *arch == c.box_.arch {
            r.ok("hardware.arch", arch.clone());
        } else {
            r.fail(
                "hardware.arch",
                format!("{arch}, expected {}", c.box_.arch),
                "Lumbridge Computer v1 runs on the DGX Spark (aarch64) only",
            );
        }
    }
    if let Some(want) = &c.box_.os {
        if let Some(os) = r.probe("hardware.os", &h.os, "/etc/os-release must be readable") {
            if os == want {
                r.ok("hardware.os", os.clone());
            } else {
                r.warn(
                    "hardware.os",
                    format!("{os}, expected {want}"),
                    "update [box] os after a DGX OS upgrade, or check the box",
                );
            }
        }
    }
    if let Some(gpu) = r.probe(
        "hardware.gpu",
        &h.gpu,
        "install the NVIDIA driver; nvidia-smi must list the GB10",
    ) {
        if gpu.name.contains("GB10") {
            r.ok("hardware.gpu", gpu.name.clone());
        } else {
            r.fail(
                "hardware.gpu",
                format!("{} is not a GB10", gpu.name),
                "Lumbridge Computer v1 targets the DGX Spark (GB10) only",
            );
        }
        match &h.cuda {
            Ok(cuda) => r.ok(
                "hardware.driver",
                format!("driver {} · CUDA {cuda}", gpu.driver),
            ),
            Err(e) => r.warn(
                "hardware.driver",
                format!("driver {} · CUDA unknown ({e})", gpu.driver),
                "nvidia-smi should print a CUDA Version header",
            ),
        }
        // GB10 shares one pool with the CPU and NVML can't report it; that is the
        // expected state, and memory truth comes from /proc/meminfo and cgroups.
        let m = gpu.memory_total.to_ascii_lowercase();
        if m.contains("n/a") || m.contains("not supported") {
            r.ok(
                "hardware.nvml_memory",
                "NVML memory Not Supported (expected on GB10; Lumbridge Computer reads /proc/meminfo)",
            );
        } else {
            r.warn(
                "hardware.nvml_memory",
                format!("NVML reports memory.total {}", gpu.memory_total),
                "unexpected on GB10; Lumbridge Computer still ignores NVML memory",
            );
        }
    }
    if h.cgroup_v2 {
        r.ok("hardware.cgroup_v2", "cgroup v2 mounted at /sys/fs/cgroup");
    } else {
        r.fail(
            "hardware.cgroup_v2",
            "cgroup v2 not mounted",
            "boot with systemd.unified_cgroup_hierarchy=1",
        );
    }
    match (&h.user, h.linger) {
        (Some(u), true) => r.ok("hardware.linger", format!("lingering on for {u}")),
        (Some(u), false) => r.fail(
            "hardware.linger",
            format!("lingering off for {u}: user services stop at logout"),
            &format!("sudo loginctl enable-linger {u}"),
        ),
        (None, _) => r.warn(
            "hardware.linger",
            "USER is not set",
            "run doctor as the box user",
        ),
    }
}

fn disk_and_memory(r: &mut Report, c: &BoxConfig, o: &Observed) {
    if let Some(d) = r.probe("disk.models", &o.disk, "the models root must exist") {
        let free = d.avail_bytes as f64 / GIB;
        let line = format!(
            "{free:.0} of {:.0} GB free on {} ({})",
            d.size_bytes as f64 / GIB,
            d.mount,
            d.path.display()
        );
        if free < DISK_FAIL_GB {
            r.fail(
                "disk.models",
                line,
                "free space before pulling or converting weights",
            );
        } else if free < DISK_WARN_GB {
            r.warn(
                "disk.models",
                line,
                "a single model can need 60+ GB; prune old weights",
            );
        } else {
            r.ok("disk.models", line);
        }
    }
    let Some(m) = r.probe("memory.available", &o.mem, "/proc/meminfo must be readable") else {
        return;
    };
    let mc = &c.memory;
    let pool = crate::pool::Pool::from_config(mc);
    let line = format!(
        "{:.1} GB available of {:.1} GB in the {} pool (floor {:.0})",
        m.available_gb, m.total_gb, pool.id, pool.floor_gb
    );
    if pool.below_floor(m.available_gb) {
        r.fail(
            "memory.available",
            line,
            "the watchdog is about to shed a model; stop one now",
        );
    } else if m.available_gb < pool.floor_gb + pool.safety_margin_gb {
        r.warn("memory.available", line, "no room to admit another model");
    } else {
        r.ok("memory.available", line);
    }
    let used = m.swap_total_gb - m.swap_free_gb;
    if m.swap_total_gb <= 0.0 {
        r.ok("memory.swap", "no swap configured");
    } else if used / m.swap_total_gb > SWAP_WARN_SHARE {
        r.warn(
            "memory.swap",
            format!("{used:.1} of {:.1} GB swap in use", m.swap_total_gb),
            "pages were pushed out under pressure; find what grew past its footprint",
        );
    } else {
        r.ok(
            "memory.swap",
            format!("{used:.1} of {:.1} GB swap in use", m.swap_total_gb),
        );
    }
    swap_top(r, c, o, used / m.swap_total_gb.max(f64::MIN_POSITIVE));
    let cap = m.total_gb - c.supabase.reservation_gb;
    if mc.budget_gb > cap + 0.5 {
        r.warn(
            "memory.budget",
            format!(
                "budget {:.0} GB exceeds {cap:.1} GB (MemTotal minus the Supabase reservation)",
                mc.budget_gb
            ),
            "the governor caps it anyway; lower [memory] budget_gb to match",
        );
    } else {
        r.ok(
            "memory.budget",
            format!("budget {:.0} GB within {cap:.1} GB", mc.budget_gb),
        );
    }
    // This build's governor still has these compiled in; say so rather than imply the
    // file controls them.
    let built = pool.id.pool();
    let (margin, floor) = (built.safety_margin_gb, built.floor_gb);
    if (pool.safety_margin_gb - margin).abs() > f64::EPSILON
        || (pool.floor_gb - floor).abs() > f64::EPSILON
    {
        r.warn(
            "memory.governor",
            format!(
                "lumbridge-computer.toml says margin {}/floor {}, the governor uses {margin}/{floor}",
                mc.safety_margin_gb, mc.watchdog_floor_gb
            ),
            "these are not configurable from lumbridge-computer.toml yet; match the build",
        );
    } else {
        r.ok(
            "memory.governor",
            format!("margin {margin} GB · floor {floor} GB"),
        );
    }
}

/// Whose pages are in swap, grouped by cgroup. Warns alongside `memory.swap` so the
/// "how much" always comes with a "who".
fn swap_top(r: &mut Report, c: &BoxConfig, o: &Observed, used_share: f64) {
    const ID: &str = "memory.swap.top";
    let holders = match &o.swap {
        Ok(h) => h,
        Err(e) => {
            r.warn(ID, e.clone(), "/proc must be readable");
            return;
        }
    };
    if holders.is_empty() {
        r.ok(ID, "no process has pages in swap");
        return;
    }
    let groups = swap_by_cgroup(holders);
    let line = groups
        .iter()
        .take(3)
        .map(|g| {
            let names = g
                .names
                .iter()
                .take(3)
                .map(|(n, kb)| format!("{n} {}", swap_size(*kb)))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{} {} [{}] ({names})",
                swap_size(g.kb),
                g.cgroup.rsplit('/').next().unwrap_or(&g.cgroup),
                swap_class(c, &g.cgroup)
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    if used_share <= SWAP_WARN_SHARE {
        r.ok(ID, line);
        return;
    }
    let hint = match swap_class(c, &groups[0].cgroup) {
        "supabase" => "Postgres pages are in swap: lumbridge-computer apply --only supabase-floor",
        "models" => {
            "cold pages of long-idle models; harmless without swap-in traffic (vmstat pswpin). For latency-critical models see docs/memory-pools.md (MemorySwapMax=0 per model scope)"
        }
        _ => "not model memory; see docs/memory-pools.md before touching vm.swappiness",
    };
    r.warn(ID, line, hint);
}

/// GB with one decimal, or MB below 0.1 GB so small holders don't all read "0.0".
fn swap_size(kb: u64) -> String {
    if kb >= 104_858 {
        format!("{:.1} GB", kb as f64 / 1_048_576.0)
    } else {
        format!("{} MB", kb / 1024)
    }
}

struct SwapGroup {
    cgroup: String,
    kb: u64,
    /// Process names in this cgroup with their summed swap, largest first.
    names: Vec<(String, u64)>,
}

fn swap_by_cgroup(holders: &[SwapHolder]) -> Vec<SwapGroup> {
    let mut by: std::collections::BTreeMap<&str, std::collections::BTreeMap<&str, u64>> =
        Default::default();
    for h in holders {
        *by.entry(&h.cgroup).or_default().entry(&h.name).or_default() += h.swap_kb;
    }
    let mut groups: Vec<SwapGroup> = by
        .into_iter()
        .map(|(cgroup, names)| {
            let mut names: Vec<(String, u64)> = names
                .into_iter()
                .map(|(n, kb)| (n.to_string(), kb))
                .collect();
            names.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            SwapGroup {
                cgroup: cgroup.to_string(),
                kb: names.iter().map(|(_, kb)| kb).sum(),
                names,
            }
        })
        .collect();
    groups.sort_by(|a, b| b.kb.cmp(&a.kb).then_with(|| a.cgroup.cmp(&b.cgroup)));
    groups
}

/// Who a cgroup belongs to, for the swap report.
fn swap_class(c: &BoxConfig, cgroup: &str) -> &'static str {
    let path = cgroup.trim_start_matches('/');
    if path.starts_with(&c.slice_cgroup()) {
        "supabase"
    } else if path.contains("lumbridge-computer") {
        // The resident agent's cgroup holds every model it starts without a
        // scope; the scopes and their slice carry the same name.
        "models"
    } else if path.contains("/session-")
        || path.contains("gnome")
        || path.contains("user-126.slice")
    {
        "desktop"
    } else {
        "other"
    }
}

fn supabase(r: &mut Report, c: &BoxConfig, o: &Observed) {
    let s = &c.supabase;
    match &o.docker {
        Ok(()) => r.ok("docker.daemon", "docker answers"),
        Err(e) => {
            r.fail("docker.daemon", e.clone(), "sudo systemctl start docker");
            return;
        }
    }
    let sb = &o.supabase;
    let Some(project) = r.probe(
        "supabase.project",
        &sb.project,
        "docker compose ls must work",
    ) else {
        return;
    };
    let Some(project) = project else {
        r.fail(
            "supabase.project",
            format!("compose project {} does not exist", s.project),
            "install it with assets/supabase/install.sh",
        );
        return;
    };
    r.ok(
        "supabase.project",
        format!("{} {}", project.name, project.status),
    );

    if let Some(containers) = r.probe(
        "supabase.health",
        &sb.containers,
        "docker compose ps must work",
    ) {
        health(r, containers);
        overlay(
            r,
            c,
            project.config_files.as_slice(),
            sb.overlay_present,
            containers,
        );
        images(r, c, o, containers);
        limits(r, c, sb.slice_memory_gb, containers);
    }
    memory_floor(r, c, &sb.floor);
    match &sb.pg_isready {
        Ok(out) => r.ok("supabase.pg_isready", out.trim().to_string()),
        Err(e) => r.fail("supabase.pg_isready", e.clone(), "docker logs supabase-db"),
    }
    match &sb.env {
        Ok(env) => {
            let missing: Vec<&str> = REQUIRED_ENV_KEYS
                .iter()
                .copied()
                .filter(|k| !env.keys.contains(*k))
                .collect();
            if !missing.is_empty() {
                r.fail(
                    "supabase.env",
                    format!(".env lacks {}", missing.join(", ")),
                    "compare with .env.example; never regenerate secrets on a live box",
                );
            } else if env.mode.is_some_and(|m| m & 0o077 != 0) {
                r.fail(
                    "supabase.env",
                    format!(
                        ".env is mode {:o}; it holds the database secrets",
                        env.mode.unwrap_or(0)
                    ),
                    &format!("chmod 600 {}", s.dir.join(".env").display()),
                );
            } else {
                r.ok(
                    "supabase.env",
                    format!(
                        "{} required keys present, mode 600",
                        REQUIRED_ENV_KEYS.len()
                    ),
                );
            }
        }
        Err(e) => r.fail(
            "supabase.env",
            e.clone(),
            "the stack cannot start without .env",
        ),
    }
}

fn health(r: &mut Report, containers: &[Container]) {
    let bad: Vec<String> = containers
        .iter()
        .filter(|c| c.state != "running" || c.health.as_deref().is_some_and(|h| h != "healthy"))
        .map(|c| {
            format!(
                "{} ({}{})",
                c.service,
                c.state,
                c.health
                    .as_deref()
                    .map(|h| format!("/{h}"))
                    .unwrap_or_default()
            )
        })
        .collect();
    if containers.is_empty() {
        r.fail(
            "supabase.health",
            "no containers running",
            "docker compose up -d in the Supabase dir",
        );
    } else if bad.is_empty() {
        r.ok(
            "supabase.health",
            format!("{} containers running and healthy", containers.len()),
        );
    } else {
        r.fail(
            "supabase.health",
            format!("unhealthy: {}", bad.join(", ")),
            "docker compose logs <service> in the Supabase dir",
        );
    }
}

fn overlay(
    r: &mut Report,
    c: &BoxConfig,
    config_files: &[String],
    overlay_present: bool,
    containers: &[Container],
) {
    let s = &c.supabase;
    let overlay_path = s.dir.join(&s.overlay);
    let loaded = config_files
        .iter()
        .any(|f| std::path::Path::new(f) == overlay_path);
    if !overlay_present {
        r.fail(
            "supabase.overlay",
            format!("{} is missing", overlay_path.display()),
            "copy the overlay from assets/supabase",
        );
    } else if !loaded {
        r.fail(
            "supabase.overlay",
            format!("project was started without {}", s.overlay),
            "set COMPOSE_FILE=docker-compose.yml:<overlay> in .env and recreate",
        );
    } else {
        r.ok("supabase.overlay", format!("{} loaded", s.overlay));
    }

    let outside: Vec<&str> = containers
        .iter()
        .filter(|ct| ct.cgroup_parent != s.slice)
        .map(|ct| ct.service.as_str())
        .collect();
    if outside.is_empty() {
        r.ok("supabase.slice", format!("all containers in {}", s.slice));
    } else {
        r.fail(
            "supabase.slice",
            format!("outside {}: {}", s.slice, outside.join(", ")),
            "the overlay's cgroup_parent was not applied; recreate with the overlay",
        );
    }

    match containers.iter().find(|ct| ct.service == "db") {
        Some(db) if db.oom_score_adj < 0 => r.ok(
            "supabase.db_oom",
            format!("db oom_score_adj {}", db.oom_score_adj),
        ),
        Some(db) => r.fail(
            "supabase.db_oom",
            format!(
                "db oom_score_adj {}: the kernel may kill Postgres before a model",
                db.oom_score_adj
            ),
            "recreate db with the overlay (oom_score_adj: -900)",
        ),
        None => r.fail(
            "supabase.db_oom",
            "no db container",
            "docker compose up -d db",
        ),
    }

    let exposed: Vec<String> = containers
        .iter()
        .flat_map(|ct| {
            ct.bindings
                .iter()
                .filter(|b| !b.is_loopback())
                .map(move |b| {
                    let ip = if b.host_ip.is_empty() {
                        "0.0.0.0"
                    } else {
                        &b.host_ip
                    };
                    format!("{} {ip}:{}", ct.service, b.host_port)
                })
        })
        .collect();
    if exposed.is_empty() {
        r.ok(
            "supabase.loopback",
            "every published port is bound to 127.0.0.1",
        );
    } else {
        r.fail(
            "supabase.loopback",
            format!("published beyond loopback: {}", exposed.join(", ")),
            "bind ports to 127.0.0.1 via the overlay (`ports: !override`) and recreate",
        );
    }
}

fn images(r: &mut Report, c: &BoxConfig, o: &Observed, containers: &[Container]) {
    let s = &c.supabase;
    let sb = &o.supabase;
    match &sb.stamp {
        Some(st) if st.ref_ == s.upstream_ref => match &st.commit {
            Some(commit) if *commit != s.upstream_commit => r.fail(
                "supabase.pin",
                format!(
                    "{} {} records commit {commit}, pin says {}",
                    st.file, st.ref_, s.upstream_commit
                ),
                "the tag moved or the pin is wrong; verify with git ls-remote",
            ),
            _ => r.ok("supabase.pin", format!("{} ({})", st.ref_, st.file)),
        },
        Some(st) => r.fail(
            "supabase.pin",
            format!(
                "deployed {} ({}), pin says {}",
                st.ref_, st.file, s.upstream_ref
            ),
            "upgrade with update.sh after a backup, or fix upstream_ref",
        ),
        None => r.warn(
            "supabase.pin",
            "no version stamp in the Supabase dir",
            "install.sh writes .lumbridge-computer-upstream; update.sh writes .supabase-version",
        ),
    }
    if let Some(current) = &sb.stamp {
        let stale: Vec<String> = sb
            .all_stamps
            .iter()
            .filter(|st| st.ref_ != current.ref_)
            .map(|st| format!("{} says {}", st.file, st.ref_))
            .collect();
        if !stale.is_empty() {
            r.warn(
                "supabase.stamps",
                format!(
                    "stale stamp: {} (current {})",
                    stale.join(", "),
                    current.ref_
                ),
                "update or remove the stale stamp so tools agree on the version",
            );
        }
    }
    match &sb.compose_services {
        Ok(services) => {
            let mut drift = Vec::new();
            for ct in containers {
                if let Some(want) = services.get(&ct.service).and_then(|s| s.image.as_deref()) {
                    if !want.contains('$') && want != ct.image {
                        drift.push(format!(
                            "{} runs {} (compose: {want})",
                            ct.service, ct.image
                        ));
                    }
                }
            }
            let missing: Vec<&str> = services
                .iter()
                .filter(|(name, svc)| {
                    !svc.has_profile && !containers.iter().any(|ct| &ct.service == *name)
                })
                .map(|(name, _)| name.as_str())
                .collect();
            if !drift.is_empty() {
                r.fail(
                    "supabase.images",
                    drift.join("; "),
                    "docker compose up -d recreates services on the pinned images",
                );
            } else if !missing.is_empty() {
                r.fail(
                    "supabase.images",
                    format!("not running: {}", missing.join(", ")),
                    "docker compose up -d in the Supabase dir",
                );
            } else {
                r.ok(
                    "supabase.images",
                    format!(
                        "{} services run the images in docker-compose.yml",
                        containers.len()
                    ),
                );
            }
        }
        Err(e) => r.fail(
            "supabase.images",
            e.clone(),
            "docker-compose.yml must be readable",
        ),
    }
}

fn limits(r: &mut Report, c: &BoxConfig, slice_gb: Option<f64>, containers: &[Container]) {
    let reserved = c.supabase.reservation_gb;
    let limits_gb: f64 = containers
        .iter()
        .map(|ct| ct.memory_bytes as f64 / GIB)
        .sum();
    let unlimited: Vec<&str> = containers
        .iter()
        .filter(|ct| ct.memory_bytes <= 0)
        .map(|ct| ct.service.as_str())
        .collect();
    let used = slice_gb
        .map(|u| format!("{u:.1} GB in use"))
        .unwrap_or_else(|| "slice usage unreadable".into());
    if !unlimited.is_empty() {
        r.warn(
            "supabase.limits",
            format!("no memory limit: {}", unlimited.join(", ")),
            "every container needs mem_limit for the reservation to mean anything",
        );
    } else if limits_gb > reserved + 0.05 {
        r.warn(
            "supabase.limits",
            format!(
                "container limits sum to {limits_gb:.1} GB, reservation is {reserved} GB ({used})"
            ),
            "raise reservation_gb (and the registry reservation) or lower mem_limit values",
        );
    } else {
        r.ok(
            "supabase.limits",
            format!("limits {limits_gb:.1} GB within reservation {reserved} GB ({used})"),
        );
    }
    if let Some(u) = slice_gb {
        if u > reserved * SLICE_WARN_SHARE {
            r.warn(
                "supabase.memory",
                format!("{u:.1} GB used of the {reserved} GB reservation"),
                "Supabase is about to outgrow its reservation",
            );
        }
    }
}

/// Is the kernel guaranteeing Supabase its floor? Read from the slice's own
/// `memory.min`/`memory.low` (and its ancestors', which cap it), never from config.
fn memory_floor(r: &mut Report, c: &BoxConfig, facts: &FloorFacts) {
    const ID: &str = "supabase.memory_floor";
    let target = floor::target_bytes(c);
    let want = c.supabase.memory_min_gb;
    if target == 0 {
        r.ok(ID, "no floor configured ([supabase] memory_min_gb = 0)");
        return;
    }
    match facts.level {
        Level::Missing => {
            r.warn(
                ID,
                format!(
                    "{} has no cgroup: the stack is not running",
                    c.supabase.slice
                ),
                "start Supabase, then re-run doctor",
            );
            return;
        }
        Level::User => {
            r.warn(
                ID,
                format!(
                    "{} is a user slice; user.slice caps its memory.min, so nothing protects Postgres",
                    c.supabase.slice
                ),
                "put the containers in a system slice (the overlay's cgroup_parent)",
            );
            return;
        }
        Level::System => {}
    }
    let now = floor::describe(facts);
    if facts.effective_min_bytes() >= target {
        r.ok(ID, format!("memory.min guarantees {want} GiB: {now}"));
    } else if facts.effective_low_bytes() >= target {
        r.ok(
            ID,
            format!("memory.low protects {want} GiB (best effort, not memory.min): {now}"),
        );
    } else {
        r.warn(
            ID,
            format!(
                "unprotected ({now}): under pressure the kernel may reclaim Postgres before models"
            ),
            "lumbridge-computer apply --only supabase-floor (system slice: prints the sudo commands)",
        );
    }
}

fn tailscale(r: &mut Report, c: &BoxConfig, o: &Observed) {
    let t = &c.tailscale;
    let Some(st) = r.probe(
        "tailscale.login",
        &o.tailscale.status,
        "is tailscaled running?",
    ) else {
        return;
    };
    if st.backend_state != "Running" {
        r.fail(
            "tailscale.login",
            format!("backend state {}", st.backend_state),
            "sudo tailscale up (interactive; Lumbridge Computer will not do this for you)",
        );
        return;
    }
    r.ok("tailscale.login", format!("logged in as {}", st.dns_name));
    if st.hostname == t.hostname && st.magic_dns_suffix == t.tailnet {
        r.ok(
            "tailscale.identity",
            format!("{}.{}", st.hostname, st.magic_dns_suffix),
        );
    } else {
        r.fail(
            "tailscale.identity",
            format!(
                "{} on {}, expected {} on {}",
                st.hostname, st.magic_dns_suffix, t.hostname, t.tailnet
            ),
            "wrong tailnet or hostname: fix the box or lumbridge-computer.toml",
        );
    }
    if !st.health.is_empty() {
        r.warn(
            "tailscale.health",
            st.health.join("; "),
            "see tailscale status",
        );
    }
    let Some(serve) = r.probe(
        "tailscale.routes",
        &o.tailscale.serve,
        "tailscale serve status must work",
    ) else {
        return;
    };
    let changes = super::plan::tailscale_changes(c, o);
    if changes.is_empty() {
        r.ok(
            "tailscale.routes",
            format!(
                "{} serve routes match lumbridge-computer.toml",
                serve.routes.len()
            ),
        );
    } else {
        r.fail(
            "tailscale.routes",
            changes
                .iter()
                .map(|ch| ch.summary())
                .collect::<Vec<_>>()
                .join("; "),
            "update [[tailscale.routes]] or the serve config; see apply --dry-run",
        );
    }
    if serve.funnel.is_empty() {
        r.ok(
            "tailscale.funnel",
            "Funnel off: nothing is on the public internet",
        );
    } else {
        r.fail(
            "tailscale.funnel",
            format!("Funnel on for {}", serve.funnel.join(", ")),
            "tailscale funnel <port> off — Lumbridge Computer serves the tailnet only",
        );
    }
    // The URL devices are given must actually reach the API gateway.
    let gw_port = match &o.supabase.containers {
        Ok(cs) => cs
            .iter()
            .filter(|ct| ct.service == "api-gw" || ct.service == "kong")
            .flat_map(|ct| ct.bindings.iter())
            .find(|b| b.container_port.starts_with("8000"))
            .map(|b| b.host_port.clone()),
        Err(_) => None,
    };
    let served = serve
        .routes
        .iter()
        .find(|rt| rt.port == c.supabase.tailnet_port && rt.path == "/");
    match (served, gw_port) {
        (Some(rt), Some(port)) => {
            let want = normalize_proxy(&port);
            if rt.proxy.as_deref().map(normalize_proxy) == Some(want) {
                r.ok(
                    "supabase.public_url",
                    format!("{} -> api gateway :{port}", c.supabase.public_url),
                );
            } else {
                r.fail(
                    "supabase.public_url",
                    format!(
                        ":{} serves {}, the API gateway is on :{port}",
                        rt.port, rt.describe
                    ),
                    "point the serve route at the gateway port",
                );
            }
        }
        (None, _) => r.fail(
            "supabase.public_url",
            format!("nothing served on :{}", c.supabase.tailnet_port),
            "add a [[tailscale.routes]] entry for the Supabase gateway",
        ),
        (Some(_), None) => r.warn(
            "supabase.public_url",
            "API gateway port unknown",
            "is the api-gw container running?",
        ),
    }
}

fn models(r: &mut Report, c: &BoxConfig, o: &Observed) {
    let m = &o.models;
    let root = c.models.root.display().to_string();
    if let Some(reg) = r.probe(
        "models.registry",
        &m.registry,
        &format!("fix {root}/registry/models.yaml"),
    ) {
        r.ok("models.registry", format!("{} models", reg.models));
        let want = c.slice_cgroup();
        match reg
            .reservations
            .iter()
            .find(|res| res.cgroup.as_deref() == Some(want.as_str()))
        {
            Some(res) if (res.gb - c.supabase.reservation_gb).abs() < 0.05 => r.ok(
                "supabase.reservation",
                format!("registry reserves {} GB for {want}", res.gb),
            ),
            Some(res) => r.fail(
                "supabase.reservation",
                format!(
                    "registry reserves {} GB for {want}, lumbridge-computer.toml says {}",
                    res.gb, c.supabase.reservation_gb
                ),
                "make registry reservations and [supabase] reservation_gb agree",
            ),
            None => r.fail(
                "supabase.reservation",
                format!("no registry reservation with cgroup {want}"),
                "add one under reservations: so admission holds Supabase's memory back",
            ),
        }
    }
    if let Some(scenes) = r.probe("models.scenes", &m.scenes, &format!("fix {root}/scenes")) {
        match scenes.iter().find(|s| s.name == c.models.desired_scene) {
            Some(s) => {
                r.ok("models.scenes", format!("{} scenes", scenes.len()));
                if let Some(b) = s.budget_gb.filter(|b| *b > c.memory.budget_gb) {
                    r.warn(
                        "memory.scene_budget",
                        format!(
                            "scene {} asks for {b} GB, [memory] budget is {}",
                            s.name, c.memory.budget_gb
                        ),
                        "lower the scene budget_gb or raise [memory] budget_gb",
                    );
                }
            }
            None => r.fail(
                "models.scenes",
                format!("no scene named {}", c.models.desired_scene),
                "fix [models] desired_scene",
            ),
        }
    }
    match &m.plan {
        Ok(p) => r.ok(
            "models.desired_scene",
            format!(
                "{} validates (budget {:.0} GB)",
                c.models.desired_scene, p.budget_gb
            ),
        ),
        Err(e) => r.fail(
            "models.desired_scene",
            e.clone(),
            "lumbridge-computer scene show <name>",
        ),
    }
    match &m.state {
        Ok(st) => {
            let active = st.active.as_deref().unwrap_or("-");
            if st.transition.is_some() {
                r.warn(
                    "models.active",
                    format!(
                        "a transition to {} is in progress",
                        st.transition.as_deref().unwrap_or("?")
                    ),
                    "wait for it to finish",
                );
            } else if active == c.models.desired_scene {
                r.ok("models.active", format!("active scene {active}"));
            } else {
                r.warn(
                    "models.active",
                    format!("active scene {active}, desired {}", c.models.desired_scene),
                    "see apply --dry-run",
                );
            }
        }
        Err(e) => r.fail(
            "models.active",
            e.clone(),
            "the .compute/state.yaml is unreadable",
        ),
    }
}

pub fn render_text(checks: &[Check]) -> String {
    let mut out = String::new();
    for ch in checks {
        let tag = match ch.status {
            Status::Ok => "ok  ",
            Status::Warn => "WARN",
            Status::Fail => "FAIL",
        };
        out.push_str(&format!("{tag}  {:<22} {}\n", ch.id, ch.reason));
        if ch.status != Status::Ok {
            if let Some(h) = &ch.hint {
                out.push_str(&format!("      {:<22} fix: {h}\n", ""));
            }
        }
    }
    let count = |s| checks.iter().filter(|c| c.status == s).count();
    out.push_str(&format!(
        "\n{} ok · {} warn · {} fail\n",
        count(Status::Ok),
        count(Status::Warn),
        count(Status::Fail)
    ));
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::cmd::Fake;
    use super::super::config::{BoxConfig, EXAMPLE};
    use super::super::observe::{
        self, fixtures::*, Models, PlanFacts, RegistryFacts, ReservationFacts, SceneFacts,
        SceneState,
    };
    use super::*;
    use std::path::Path;

    pub fn config() -> BoxConfig {
        BoxConfig::parse(EXAMPLE, Some(Path::new("/home/sam"))).unwrap()
    }

    /// The models half of the fixture box: registry, scenes and a clean plan.
    pub fn box1_models() -> Models {
        Models {
            registry: Ok(RegistryFacts {
                models: 60,
                reservations: vec![ReservationFacts {
                    id: "server".into(),
                    gb: 8.5,
                    cgroup: Some("lumbridge.slice/lumbridge-server.slice".into()),
                }],
            }),
            scenes: Ok(vec![SceneFacts {
                name: "everything".into(),
                budget_gb: Some(111.0),
            }]),
            state: Ok(SceneState {
                active: Some("everything".into()),
                desired: Some("everything".into()),
                transition: None,
            }),
            plan: Ok(PlanFacts {
                budget_gb: 111.0,
                stop: vec![],
                start: vec![],
            }),
            dev_mode: Ok(vec![]),
        }
    }

    pub fn observe(host: &Fake) -> Observed {
        let mut o = observe::gather(host, &config());
        o.models = box1_models();
        o
    }

    fn status_of(checks: &[Check], id: &str) -> Status {
        checks
            .iter()
            .find(|c| c.id == id)
            .unwrap_or_else(|| panic!("no check {id}: {checks:#?}"))
            .status
    }

    fn doctor(host: &Fake) -> Vec<Check> {
        run(&config(), &observe(host))
    }

    #[test]
    fn box1_fixtures_pass_every_check_except_the_known_findings() {
        let checks = doctor(&box1());
        let not_ok: Vec<(&str, Status)> = checks
            .iter()
            .filter(|c| c.status != Status::Ok)
            .map(|c| (c.id.as_str(), c.status))
            .collect();
        // The fixture box's findings: 62% swap in use, install.sh's stamp still
        // says v0.8.2 after update.sh moved the stack to v0.8.4, and no
        // memory.min on the Supabase slice or its parent.
        assert_eq!(
            not_ok,
            vec![
                ("memory.swap", Status::Warn),
                ("memory.swap.top", Status::Warn),
                ("supabase.stamps", Status::Warn),
                ("supabase.memory_floor", Status::Warn),
            ],
            "{}",
            render_text(&checks)
        );
    }

    fn dev(
        id: &str,
        running: bool,
        dev_mode: bool,
        allow_public: bool,
        listening: &[&str],
    ) -> DevModeFacts {
        DevModeFacts {
            id: id.into(),
            running,
            dev_mode,
            allow_public,
            port: Some(8001),
            listening: listening.iter().map(|a| a.to_string()).collect(),
        }
    }

    fn dev_mode_verdict(facts: Vec<DevModeFacts>) -> Check {
        let mut o = observe(&box1());
        o.models.dev_mode = Ok(facts);
        run(&config(), &o)
            .into_iter()
            .find(|c| c.id == "models.dev_mode")
            .unwrap()
    }

    #[test]
    fn a_running_dev_mode_engine_bound_off_loopback_fails() {
        // A brain recipe that binds 0.0.0.0:8001; in dev mode that is
        // /collective_rpc for the whole network.
        for addr in ["0.0.0.0", "192.0.2.7", "::"] {
            let check =
                dev_mode_verdict(vec![dev("brain-osprey-stock", true, true, false, &[addr])]);
            assert_eq!(check.status, Status::Fail, "{addr}");
            assert!(
                check.reason.contains("brain-osprey-stock"),
                "{}",
                check.reason
            );
            assert!(check.reason.contains(addr), "{}", check.reason);
        }
        // Even with the escape hatch: the hatch warns, an open port fails.
        let check = dev_mode_verdict(vec![dev("b", true, true, true, &["127.0.0.1", "0.0.0.0"])]);
        assert_eq!(check.status, Status::Fail);
    }

    #[test]
    fn dev_mode_on_loopback_is_ok_and_the_public_escape_hatch_warns() {
        let ok = dev_mode_verdict(vec![dev("b", true, true, false, &["127.0.0.1"])]);
        assert_eq!(ok.status, Status::Ok, "{}", ok.reason);
        // Not running, or running without dev mode: a public bind is fine.
        let ok = dev_mode_verdict(vec![
            dev("asleep-later", false, true, false, &["0.0.0.0"]),
            dev("plain", true, false, false, &["0.0.0.0"]),
        ]);
        assert_eq!(ok.status, Status::Ok, "{}", ok.reason);
        let warn = dev_mode_verdict(vec![dev("b", true, true, true, &["::1"])]);
        assert_eq!(warn.status, Status::Warn);
        assert!(warn.reason.contains("allow_public_dev_mode"));
    }

    #[test]
    fn wrong_architecture_fails() {
        let mut h = box1();
        h.on("uname -m", "x86_64\n");
        assert_eq!(status_of(&doctor(&h), "hardware.arch"), Status::Fail);
    }

    #[test]
    fn os_drift_warns() {
        let mut h = box1();
        h.file("/etc/os-release", "ID=ubuntu\nVERSION_ID=\"26.04\"\n");
        assert_eq!(status_of(&doctor(&h), "hardware.os"), Status::Warn);
    }

    #[test]
    fn a_gpu_other_than_gb10_fails() {
        let mut h = box1();
        h.on("nvidia-smi --query-gpu", "NVIDIA H100, 580.1, 81559 MiB\n");
        let c = doctor(&h);
        assert_eq!(status_of(&c, "hardware.gpu"), Status::Fail);
        assert_eq!(status_of(&c, "hardware.nvml_memory"), Status::Warn);
    }

    #[test]
    fn missing_nvidia_smi_fails_the_gpu_check() {
        let mut h = box1();
        h.fail("nvidia-smi", "nvidia-smi: not found");
        assert_eq!(status_of(&doctor(&h), "hardware.gpu"), Status::Fail);
    }

    #[test]
    fn nvml_memory_not_supported_is_ok_on_gb10() {
        let c = doctor(&box1());
        assert_eq!(status_of(&c, "hardware.nvml_memory"), Status::Ok);
        let driver = c.iter().find(|c| c.id == "hardware.driver").unwrap();
        assert!(driver.reason.contains("580.173.02") && driver.reason.contains("13.0"));
    }

    #[test]
    fn cgroup_v1_fails() {
        let mut h = box1();
        h.remove_file("/sys/fs/cgroup/cgroup.controllers");
        assert_eq!(status_of(&doctor(&h), "hardware.cgroup_v2"), Status::Fail);
    }

    #[test]
    fn lingering_off_fails_with_the_enable_command() {
        let mut h = box1();
        h.files
            .retain(|p, _| !p.starts_with("/var/lib/systemd/linger"));
        let c = doctor(&h);
        let l = c.iter().find(|c| c.id == "hardware.linger").unwrap();
        if std::env::var("USER").is_ok() {
            assert_eq!(l.status, Status::Fail);
            assert!(l.hint.as_deref().unwrap().contains("enable-linger"));
        }
    }

    #[test]
    fn low_disk_warns_then_fails() {
        let mut h = box1();
        h.on("df -P -B1", "Filesystem 1-blocks Used Available Capacity Mounted on\n/dev/x 4000000000000 3950000000000 50000000000 99% /\n");
        assert_eq!(status_of(&doctor(&h), "disk.models"), Status::Warn);
        h.on("df -P -B1", "Filesystem 1-blocks Used Available Capacity Mounted on\n/dev/x 4000000000000 3990000000000 10000000000 99% /\n");
        assert_eq!(status_of(&doctor(&h), "disk.models"), Status::Fail);
    }

    #[test]
    fn mem_available_below_the_floor_fails() {
        let mut h = box1();
        h.file(
            "/proc/meminfo",
            "MemTotal: 127600816 kB\nMemAvailable: 2000000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
        );
        let c = doctor(&h);
        assert_eq!(status_of(&c, "memory.available"), Status::Fail);
        assert_eq!(status_of(&c, "memory.swap"), Status::Ok);
    }

    #[test]
    fn swap_top_names_the_model_cgroup_and_its_processes_on_box1() {
        let checks = doctor(&box1());
        let top = checks.iter().find(|c| c.id == "memory.swap.top").unwrap();
        assert_eq!(top.status, Status::Warn);
        assert!(
            top.reason.starts_with(
                "9.6 GB lumbridge-computer-agent.service [models] (nemo-speech 3.9 GB, VLLM::EngineCor 2.9 GB, vllm 1.5 GB); 40 MB docker-b34850dc80fd.scope [supabase] (.postgres-wrapp 40 MB); 25 MB session-c1.scope [desktop]"
            ),
            "{}",
            top.reason
        );
        assert!(top.reason.contains("[supabase]") && top.reason.contains("[desktop]"));
        assert!(top.hint.as_deref().unwrap().contains("MemorySwapMax=0"));
    }

    #[test]
    fn swap_top_is_ok_when_little_swap_is_used_and_points_at_the_floor_for_postgres() {
        let mut h = box1();
        h.file(
            "/proc/meminfo",
            "MemTotal: 127600816 kB\nMemAvailable: 60000000 kB\nSwapTotal: 16777212 kB\nSwapFree: 16000000 kB\n",
        );
        let checks = doctor(&h);
        assert_eq!(status_of(&checks, "memory.swap.top"), Status::Ok);
        let mut h = box1();
        h.file(
            "/proc/728325/status",
            "Name:\tpostgres\nVmSwap:\t 20000000 kB\n",
        );
        let checks = doctor(&h);
        let top = checks.iter().find(|c| c.id == "memory.swap.top").unwrap();
        assert!(
            top.reason
                .starts_with("19.1 GB docker-b34850dc80fd.scope [supabase]"),
            "{}",
            top.reason
        );
        assert!(top.hint.as_deref().unwrap().contains("supabase-floor"));
    }

    #[test]
    fn budget_above_capacity_warns() {
        let mut c = config();
        c.memory.budget_gb = 125.0;
        let checks = run(&c, &observe(&box1()));
        assert_eq!(status_of(&checks, "memory.budget"), Status::Warn);
    }

    #[test]
    fn governor_constants_that_differ_from_the_file_warn() {
        let mut c = config();
        c.memory.watchdog_floor_gb = 5.0;
        assert_eq!(
            status_of(&run(&c, &observe(&box1())), "memory.governor"),
            Status::Warn
        );
    }

    #[test]
    fn docker_down_fails_and_skips_supabase_details() {
        let mut h = box1();
        h.fail("docker", "Cannot connect to the Docker daemon");
        let c = doctor(&h);
        assert_eq!(status_of(&c, "docker.daemon"), Status::Fail);
        assert!(!c.iter().any(|c| c.id == "supabase.health"));
    }

    #[test]
    fn missing_compose_project_fails() {
        let mut h = box1();
        h.on("docker compose ls --format json", "[]");
        assert_eq!(status_of(&doctor(&h), "supabase.project"), Status::Fail);
    }

    #[test]
    fn an_unhealthy_container_fails_health() {
        let mut h = box1();
        h.on(
            &format!("docker inspect --format {}", observe::INSPECT_FORMAT),
            &INSPECT.replacen("\"health\":\"healthy\"", "\"health\":\"unhealthy\"", 1),
        );
        assert_eq!(status_of(&doctor(&h), "supabase.health"), Status::Fail);
    }

    #[test]
    fn pg_isready_failure_fails() {
        let mut h = box1();
        h.fail("docker exec supabase-db pg_isready", "no response");
        assert_eq!(status_of(&doctor(&h), "supabase.pg_isready"), Status::Fail);
    }

    #[test]
    fn a_container_outside_the_slice_fails() {
        let mut h = box1();
        h.on(
            &format!("docker inspect --format {}", observe::INSPECT_FORMAT),
            &INSPECT.replacen(
                "\"cgroup_parent\":\"lumbridge-server.slice\"",
                "\"cgroup_parent\":\"\"",
                1,
            ),
        );
        assert_eq!(status_of(&doctor(&h), "supabase.slice"), Status::Fail);
    }

    #[test]
    fn db_without_negative_oom_score_fails() {
        let mut h = box1();
        h.on(
            &format!("docker inspect --format {}", observe::INSPECT_FORMAT),
            &INSPECT.replace("\"oom_score_adj\":-900", "\"oom_score_adj\":0"),
        );
        assert_eq!(status_of(&doctor(&h), "supabase.db_oom"), Status::Fail);
    }

    #[test]
    fn any_port_on_every_interface_fails() {
        for ip in ["0.0.0.0", "", "::"] {
            let mut h = box1();
            h.on(
                &format!("docker inspect --format {}", observe::INSPECT_FORMAT),
                &INSPECT.replace(
                    "\"HostIp\":\"127.0.0.1\",\"HostPort\":\"5432\"",
                    &format!("\"HostIp\":\"{ip}\",\"HostPort\":\"5432\""),
                ),
            );
            assert_eq!(
                status_of(&doctor(&h), "supabase.loopback"),
                Status::Fail,
                "{ip:?}"
            );
        }
    }

    #[test]
    fn overlay_not_loaded_or_missing_fails() {
        let mut h = box1();
        h.on(
            "docker compose ls --format json",
            &COMPOSE_LS.replace(",/home/sam/supabase/lumbridge.override.yml", ""),
        );
        assert_eq!(status_of(&doctor(&h), "supabase.overlay"), Status::Fail);
        let mut h = box1();
        h.remove_file(&format!("{SB_DIR}/lumbridge.override.yml"));
        assert_eq!(status_of(&doctor(&h), "supabase.overlay"), Status::Fail);
    }

    #[test]
    fn deployed_version_other_than_the_pin_fails() {
        let mut c = config();
        c.supabase.upstream_ref = "self-hosted/v0.8.2".into();
        assert_eq!(
            status_of(&run(&c, &observe(&box1())), "supabase.pin"),
            Status::Fail
        );
    }

    #[test]
    fn running_image_that_differs_from_compose_fails() {
        let mut h = box1();
        h.on(
            &format!("docker inspect --format {}", observe::INSPECT_FORMAT),
            &INSPECT.replace("supabase/gotrue:v2.197.0", "supabase/gotrue:v2.196.0"),
        );
        let c = doctor(&h);
        let ch = c.iter().find(|c| c.id == "supabase.images").unwrap();
        assert_eq!(ch.status, Status::Fail);
        assert!(ch.reason.contains("auth"));
    }

    #[test]
    fn missing_env_keys_or_loose_mode_fail() {
        let mut h = box1();
        h.file(
            &format!("{SB_DIR}/.env"),
            &ENV.replace("JWT_SECRET=redacted\n", ""),
        );
        assert_eq!(status_of(&doctor(&h), "supabase.env"), Status::Fail);
        let mut h = box1();
        h.modes.insert(format!("{SB_DIR}/.env").into(), 0o644);
        assert_eq!(status_of(&doctor(&h), "supabase.env"), Status::Fail);
    }

    #[test]
    fn container_limits_above_the_reservation_warn() {
        let mut c = config();
        c.supabase.reservation_gb = 8.0;
        let checks = run(&c, &observe(&box1()));
        assert_eq!(status_of(&checks, "supabase.limits"), Status::Warn);
    }

    #[test]
    fn a_reservation_mismatch_fails() {
        let mut c = config();
        c.supabase.reservation_gb = 12.0;
        let checks = run(&c, &observe(&box1()));
        assert_eq!(status_of(&checks, "supabase.reservation"), Status::Fail);
        assert_eq!(status_of(&checks, "supabase.limits"), Status::Ok);
    }

    #[test]
    fn the_memory_floor_is_ok_only_when_every_slice_on_the_path_carries_it() {
        let server = "/sys/fs/cgroup/lumbridge.slice/lumbridge-server.slice";
        let mut h = box1();
        h.file(&format!("{server}/memory.min"), "4294967296\n");
        let floor = |h: &Fake| {
            doctor(h)
                .into_iter()
                .find(|c| c.id == "supabase.memory_floor")
                .unwrap()
        };
        assert_eq!(floor(&h).status, Status::Warn, "parent caps it at 0");
        h.file("/sys/fs/cgroup/lumbridge.slice/memory.min", "4294967296\n");
        let ok = floor(&h);
        assert_eq!(ok.status, Status::Ok);
        assert!(
            ok.reason.contains("memory.min guarantees 4 GiB"),
            "{}",
            ok.reason
        );
        // memory.low alone is accepted as best-effort protection.
        let mut h = box1();
        for dir in ["/sys/fs/cgroup/lumbridge.slice", server] {
            h.file(&format!("{dir}/memory.low"), "8589934592\n");
        }
        assert!(floor(&h).reason.contains("best effort"));
    }

    #[test]
    fn a_missing_slice_cgroup_warns_that_the_stack_is_not_running() {
        let mut h = box1();
        h.files
            .retain(|p, _| !p.starts_with("/sys/fs/cgroup/lumbridge.slice"));
        let c = doctor(&h);
        let ch = c.iter().find(|c| c.id == "supabase.memory_floor").unwrap();
        assert_eq!(ch.status, Status::Warn);
        assert!(ch.reason.contains("not running"));
    }

    #[test]
    fn tailscale_logged_out_fails() {
        let mut h = box1();
        h.on(
            "tailscale status --json",
            &TS_STATUS.replace("\"Running\"", "\"NeedsLogin\""),
        );
        assert_eq!(status_of(&doctor(&h), "tailscale.login"), Status::Fail);
    }

    #[test]
    fn wrong_hostname_or_tailnet_fails_identity() {
        let mut c = config();
        c.tailscale.tailnet = "tail9999.ts.net".into();
        assert_eq!(
            status_of(&run(&c, &observe(&box1())), "tailscale.identity"),
            Status::Fail
        );
    }

    #[test]
    fn serve_route_drift_fails() {
        let mut c = config();
        c.tailscale.routes.pop();
        assert_eq!(
            status_of(&run(&c, &observe(&box1())), "tailscale.routes"),
            Status::Fail
        );
    }

    #[test]
    fn any_funnel_route_fails() {
        let mut h = box1();
        let serve = TS_SERVE.trim_end().trim_end_matches('}').to_string()
            + ",\"AllowFunnel\":{\"box1.tail0000.ts.net:443\":true}}";
        h.on("tailscale serve status --json", &serve);
        assert_eq!(status_of(&doctor(&h), "tailscale.funnel"), Status::Fail);
    }

    #[test]
    fn public_url_that_does_not_reach_the_gateway_fails() {
        let mut h = box1();
        h.on(
            "tailscale serve status --json",
            &TS_SERVE.replace("http://127.0.0.1:8000", "http://127.0.0.1:9000"),
        );
        assert_eq!(status_of(&doctor(&h), "supabase.public_url"), Status::Fail);
    }

    #[test]
    fn a_scene_that_fails_planning_fails_models() {
        let mut o = observe(&box1());
        o.models.plan = Err("Scene 'everything' references unknown model id(s): x".into());
        o.models.state = Ok(SceneState {
            active: Some("music".into()),
            ..Default::default()
        });
        let c = run(&config(), &o);
        assert_eq!(status_of(&c, "models.desired_scene"), Status::Fail);
        assert_eq!(status_of(&c, "models.active"), Status::Warn);
    }

    #[test]
    fn json_output_uses_lowercase_statuses() {
        let c = doctor(&box1());
        let j = serde_json::to_value(&c).unwrap();
        assert!(j.as_array().unwrap().iter().any(|v| v["status"] == "warn"));
        assert!(j[0]["id"].is_string());
    }
}
