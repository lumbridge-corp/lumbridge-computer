//! `lumbridge-computer.toml`: the whole box's desired state, in one small file.
//!
//! TOML rather than YAML because this file is typed by hand on the box and diffed in
//! a box repo: TOML has no implicit typing (`no` stays a string), comments survive,
//! and it is what Rust tooling (Cargo) already uses. Every table denies unknown
//! keys, so a typo is a load error instead of a silently ignored setting.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The slice the shipped Supabase overlay puts containers in when nothing else is said.
pub const DEFAULT_SLICE: &str = "lumbridge-computer.slice";

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoxConfig {
    #[serde(rename = "box")]
    pub box_: BoxSection,
    pub memory: MemorySection,
    pub models: ModelsSection,
    pub supabase: SupabaseSection,
    pub tailscale: TailscaleSection,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoxSection {
    pub name: String,
    pub hardware: Hardware,
    /// `uname -m` the box must report.
    #[serde(default = "default_arch")]
    pub arch: String,
    /// `<ID>-<VERSION_ID>` from /etc/os-release, e.g. `ubuntu-24.04`. Unchecked when absent.
    #[serde(default)]
    pub os: Option<String>,
}

/// Lumbridge Computer v1 targets one machine; the enum makes any other value a load error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Hardware {
    DgxSpark,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemorySection {
    pub pool: PoolId,
    /// Ceiling for model memory on this box; no Scene's `budget_gb` should exceed it.
    pub budget_gb: f64,
    pub safety_margin_gb: f64,
    pub watchdog_floor_gb: f64,
}

/// GB10 has one memory pool shared by CPU and GPU; the registry uses the same names.
pub use crate::pool::PoolId;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsSection {
    /// Directory holding `registry/`, `scenes/` and the `.compute/` state.
    pub root: PathBuf,
    /// Relative to `root`. M1 reads it from the standard place only (see `validate`).
    #[serde(default = "default_registry")]
    pub registry: PathBuf,
    #[serde(default = "default_scenes")]
    pub scenes: PathBuf,
    pub desired_scene: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SupabaseSection {
    /// docker compose project name.
    pub project: String,
    /// Directory holding upstream `docker-compose.yml`, the overlay and `.env`.
    pub dir: PathBuf,
    /// Upstream tag, e.g. `self-hosted/v0.8.4`.
    pub upstream_ref: String,
    /// The commit the tag peels to; install refuses a tag that moved.
    pub upstream_commit: String,
    /// Overlay file name inside `dir`.
    pub overlay: String,
    /// systemd slice every container's cgroup lives under.
    #[serde(default = "default_slice")]
    pub slice: String,
    /// Unified memory held back from models for this stack.
    pub reservation_gb: f64,
    /// cgroup v2 `memory.min` guaranteed to the slice, so under pressure the kernel
    /// reclaims from models first. Default: the Postgres container's 4 GiB limit.
    #[serde(default = "default_memory_min_gb")]
    pub memory_min_gb: f64,
    /// The URL devices use (tailnet only).
    pub public_url: String,
    /// The tailnet HTTPS port `public_url` is served on.
    pub tailnet_port: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TailscaleSection {
    /// MagicDNS suffix of the tailnet, e.g. `tail0000.ts.net`.
    pub tailnet: String,
    pub hostname: String,
    /// Every `tailscale serve` route on the box. Anything served that is not listed is drift.
    #[serde(default)]
    pub routes: Vec<Route>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Route {
    pub https_port: u16,
    #[serde(default = "default_path")]
    pub path: String,
    /// Backend: `http://127.0.0.1:8000`, `127.0.0.1:8000` or a bare port.
    pub proxy: String,
    /// Free-form label for humans: which service owns this route.
    #[serde(default)]
    pub owner: Option<String>,
}

fn default_arch() -> String {
    "aarch64".into()
}
fn default_registry() -> PathBuf {
    PathBuf::from("registry/models.yaml")
}
fn default_scenes() -> PathBuf {
    PathBuf::from("scenes")
}
fn default_slice() -> String {
    DEFAULT_SLICE.into()
}
fn default_memory_min_gb() -> f64 {
    4.0
}
fn default_path() -> String {
    "/".into()
}

impl BoxConfig {
    pub fn load(path: &Path) -> Result<BoxConfig> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text, home_dir().as_deref())
            .with_context(|| format!("loading {}", path.display()))
    }

    /// Parse, expand `~` against `home`, then validate.
    pub fn parse(text: &str, home: Option<&Path>) -> Result<BoxConfig> {
        let mut config: BoxConfig = toml::from_str(text)?;
        config.models.root = expand_tilde(&config.models.root, home)?;
        config.supabase.dir = expand_tilde(&config.supabase.dir, home)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.box_.name.trim().is_empty() {
            bail!("[box] name must not be empty");
        }
        let m = &self.memory;
        if !(m.budget_gb > 0.0 && m.safety_margin_gb >= 0.0 && m.watchdog_floor_gb >= 0.0) {
            bail!("[memory] budget_gb must be > 0 and margins >= 0");
        }
        // The planner reads `<root>/registry/models.yaml` and `<root>/scenes`; accepting
        // another path here would make doctor check one registry and the plan use another.
        if self.models.registry != default_registry() || self.models.scenes != default_scenes() {
            bail!(
                "[models] registry and scenes must be registry/models.yaml and scenes \
                 (relative to root) in this release; move root instead"
            );
        }
        let s = &self.supabase;
        if s.project.is_empty()
            || !s
                .project
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
        {
            bail!("[supabase] project must be lowercase letters, digits, '-' or '_'");
        }
        if s.upstream_commit.len() != 40
            || !s.upstream_commit.chars().all(|c| c.is_ascii_hexdigit())
        {
            bail!("[supabase] upstream_commit must be a full 40-character commit hash");
        }
        if s.overlay.contains('/') || s.overlay.is_empty() {
            bail!("[supabase] overlay is a file name inside dir, not a path");
        }
        if !s.slice.ends_with(".slice") || s.slice.contains('/') {
            bail!("[supabase] slice must be a systemd slice name like lumbridge-computer.slice");
        }
        // A floor above the reservation would guarantee the stack memory admission
        // hands to models, and the two promises could not both be kept.
        if !(s.memory_min_gb >= 0.0 && s.memory_min_gb <= s.reservation_gb) {
            bail!(
                "[supabase] memory_min_gb must be between 0 and reservation_gb ({})",
                s.reservation_gb
            );
        }
        let (host, port) = url_host_port(&s.public_url)
            .context("[supabase] public_url must be https://<host>[:port]")?;
        if port != s.tailnet_port {
            bail!(
                "[supabase] public_url port {port} disagrees with tailnet_port {}",
                s.tailnet_port
            );
        }
        let fqdn = format!("{}.{}", self.tailscale.hostname, self.tailscale.tailnet);
        if host != fqdn {
            bail!("[supabase] public_url host {host} is not this box's tailnet name {fqdn}");
        }
        let mut seen = BTreeSet::new();
        for r in &self.tailscale.routes {
            if !r.path.starts_with('/') {
                bail!("[[tailscale.routes]] path must start with '/': {}", r.path);
            }
            if normalize_proxy(&r.proxy).is_none() {
                bail!(
                    "[[tailscale.routes]] proxy is not a loopback URL or port: {}",
                    r.proxy
                );
            }
            if !seen.insert((r.https_port, r.path.clone())) {
                bail!(
                    "[[tailscale.routes]] :{}{} is listed twice",
                    r.https_port,
                    r.path
                );
            }
        }
        Ok(())
    }

    /// Expected cgroup v2 path of the Supabase slice under /sys/fs/cgroup.
    pub fn slice_cgroup(&self) -> String {
        slice_cgroup_path(&self.supabase.slice)
    }
}

/// systemd nests `a-b.slice` inside `a.slice`, so `lumbridge-server.slice` lives at
/// `lumbridge.slice/lumbridge-server.slice`.
pub fn slice_cgroup_path(slice: &str) -> String {
    let stem = slice.trim_end_matches(".slice");
    let parts: Vec<&str> = stem.split('-').collect();
    (1..=parts.len())
        .map(|n| format!("{}.slice", parts[..n].join("-")))
        .collect::<Vec<_>>()
        .join("/")
}

/// `8000`, `127.0.0.1:8000` and `http://127.0.0.1:8000/` all mean the same backend;
/// tailscale prints `http://127.0.0.1:8000`. Non-loopback backends are refused: serving
/// another host's port from this box is not something this file should express.
pub fn normalize_proxy(proxy: &str) -> Option<String> {
    let p = proxy.trim().trim_end_matches('/');
    let with_scheme = if p.chars().all(|c| c.is_ascii_digit()) {
        format!("http://127.0.0.1:{p}")
    } else if p.contains("://") {
        p.to_string()
    } else {
        format!("http://{p}")
    };
    let rest = with_scheme
        .strip_prefix("http://")
        .or_else(|| with_scheme.strip_prefix("https+insecure://"))
        .or_else(|| with_scheme.strip_prefix("https://"))?;
    let host = rest.rsplit_once(':').map(|(h, _)| h).unwrap_or(rest);
    matches!(host, "127.0.0.1" | "localhost" | "[::1]").then_some(with_scheme)
}

fn url_host_port(url: &str) -> Option<(String, u16)> {
    let rest = url.strip_prefix("https://")?.trim_end_matches('/');
    if rest.contains('/') {
        return None;
    }
    match rest.rsplit_once(':') {
        Some((host, port)) => Some((host.to_string(), port.parse().ok()?)),
        None => Some((rest.to_string(), 443)),
    }
}

pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn expand_tilde(path: &Path, home: Option<&Path>) -> Result<PathBuf> {
    let Some(text) = path.to_str() else {
        return Ok(path.to_path_buf());
    };
    if text == "~" || text.starts_with("~/") {
        let home = home.context("path uses ~ but HOME is not set")?;
        return Ok(home.join(text.trim_start_matches('~').trim_start_matches('/')));
    }
    if text.starts_with('~') {
        bail!("~user paths are not supported: {text}");
    }
    Ok(path.to_path_buf())
}

#[cfg(test)]
pub(crate) const EXAMPLE: &str = include_str!("fixtures/lumbridge-computer.toml");

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<BoxConfig> {
        BoxConfig::parse(text, Some(Path::new("/home/sam")))
    }

    #[test]
    fn example_config_loads_and_expands_tilde_paths() {
        let c = parse(EXAMPLE).unwrap();
        assert_eq!(c.models.root, Path::new("/home/sam/lumbridge-computer"));
        assert_eq!(c.supabase.dir, Path::new("/home/sam/supabase"));
        assert_eq!(c.box_.hardware, Hardware::DgxSpark);
        assert_eq!(c.tailscale.routes.len(), 4);
        assert_eq!(c.slice_cgroup(), "lumbridge.slice/lumbridge-server.slice");
    }

    #[test]
    fn unknown_keys_are_rejected_in_every_table() {
        for (table, typo) in [
            ("[box]", "nmae = \"x\""),
            ("[memory]", "budget = 1"),
            ("[supabase]", "overlays = \"x\""),
            ("[tailscale]", "hostnmae = \"x\""),
        ] {
            let text = EXAMPLE.replacen(table, &format!("{table}\n{typo}"), 1);
            let err = format!("{:#}", parse(&text).unwrap_err());
            assert!(err.contains("unknown field"), "{table}: {err}");
        }
        let text = EXAMPLE.replacen("owner = \"router\"", "ownr = \"x\"", 1);
        assert!(parse(&text).is_err());
    }

    #[test]
    fn hardware_other_than_dgx_spark_is_rejected() {
        let text = EXAMPLE.replace("hardware = \"dgx-spark\"", "hardware = \"rtx-pro\"");
        assert!(parse(&text).is_err());
    }

    #[test]
    fn memory_min_defaults_to_4_gib_and_cannot_exceed_the_reservation() {
        assert_eq!(parse(EXAMPLE).unwrap().supabase.memory_min_gb, 4.0);
        let text = EXAMPLE.replace(
            "reservation_gb = 8.5",
            "reservation_gb = 8.5\nmemory_min_gb = 6",
        );
        assert_eq!(parse(&text).unwrap().supabase.memory_min_gb, 6.0);
        let text = EXAMPLE.replace(
            "reservation_gb = 8.5",
            "reservation_gb = 8.5\nmemory_min_gb = 9",
        );
        assert!(parse(&text).is_err());
        let text = EXAMPLE.replace(
            "reservation_gb = 8.5",
            "reservation_gb = 8.5\nmemory_min_gb = -1",
        );
        assert!(parse(&text).is_err());
    }

    #[test]
    fn slice_defaults_to_lumbridge_computer_slice() {
        let text: String = EXAMPLE
            .lines()
            .filter(|l| !l.starts_with("slice ="))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(parse(&text).unwrap().supabase.slice, DEFAULT_SLICE);
    }

    #[test]
    fn public_url_must_match_tailnet_port_and_hostname() {
        let text = EXAMPLE.replace("tailnet_port = 8443", "tailnet_port = 443");
        assert!(parse(&text).is_err());
        let text = EXAMPLE.replace("hostname = \"box1\"", "hostname = \"box2\"");
        assert!(parse(&text).is_err());
    }

    #[test]
    fn duplicate_routes_and_remote_proxies_are_rejected() {
        let dup = format!(
            "{EXAMPLE}\n[[tailscale.routes]]\nhttps_port = 443\npath = \"/\"\nproxy = \"9999\"\n"
        );
        assert!(parse(&dup).is_err());
        let remote = format!(
            "{EXAMPLE}\n[[tailscale.routes]]\nhttps_port = 4443\nproxy = \"http://10.0.0.5:80\"\n"
        );
        assert!(parse(&remote).is_err());
    }

    #[test]
    fn non_standard_registry_location_is_refused() {
        let text = EXAMPLE.replace(
            "desired_scene = \"everything\"",
            "desired_scene = \"everything\"\nregistry = \"elsewhere.yaml\"",
        );
        assert!(parse(&text).is_err());
    }

    #[test]
    fn proxies_normalize_to_tailscale_spelling() {
        for p in ["8000", "127.0.0.1:8000", "http://127.0.0.1:8000/"] {
            assert_eq!(normalize_proxy(p).as_deref(), Some("http://127.0.0.1:8000"));
        }
        assert_eq!(normalize_proxy("http://example.com:80"), None);
    }

    #[test]
    fn nested_slice_names_map_to_systemd_cgroup_paths() {
        assert_eq!(
            slice_cgroup_path("lumbridge-computer.slice"),
            "lumbridge.slice/lumbridge-computer.slice"
        );
        assert_eq!(
            slice_cgroup_path("a-b-c.slice"),
            "a.slice/a-b.slice/a-b-c.slice"
        );
    }
}
