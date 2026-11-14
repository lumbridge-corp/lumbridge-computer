//! `lumbridge-computer apply --dry-run`: what applying `lumbridge-computer.toml` would change.
//!
//! Pure functions of (`lumbridge-computer.toml`, [`Observed`]). An area whose reality can't be
//! read is a blocker, not "no changes": silence must mean "verified equal".

use super::config::{normalize_proxy, BoxConfig};
use super::floor::{self, PlannedFile};
use super::observe::Observed;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Change {
    pub area: &'static str,
    /// `add` | `remove` | `update` | `recreate` | `install` | `upgrade` | `start` | `stop` | `activate`
    pub action: &'static str,
    pub target: String,
    pub detail: String,
    /// Files applying this change writes, with their exact contents.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<PlannedFile>,
}

impl Change {
    fn new(
        area: &'static str,
        action: &'static str,
        target: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Change {
            area,
            action,
            target: target.into(),
            detail: detail.into(),
            files: Vec::new(),
        }
    }
    pub fn summary(&self) -> String {
        format!("{} {}: {}", self.action, self.target, self.detail)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DryRun {
    pub changes: Vec<Change>,
    /// Areas whose current state could not be read; the diff for them is unknown.
    pub blockers: Vec<String>,
}

impl DryRun {
    pub fn is_clean(&self) -> bool {
        self.changes.is_empty() && self.blockers.is_empty()
    }
}

pub fn dry_run(c: &BoxConfig, o: &Observed) -> DryRun {
    let mut blockers = Vec::new();
    let mut changes = Vec::new();
    match (&o.tailscale.status, &o.tailscale.serve) {
        (Ok(_), Ok(_)) => changes.extend(tailscale_changes(c, o)),
        (Err(e), _) | (_, Err(e)) => blockers.push(format!("tailscale: {e}")),
    }
    match supabase_changes(c, o) {
        Ok(list) => changes.extend(list),
        Err(e) => blockers.push(format!("supabase: {e}")),
    }
    match model_changes(c, o) {
        Ok(list) => changes.extend(list),
        Err(e) => blockers.push(format!("models: {e}")),
    }
    DryRun { changes, blockers }
}

pub fn tailscale_changes(c: &BoxConfig, o: &Observed) -> Vec<Change> {
    let t = &c.tailscale;
    let mut out = Vec::new();
    if let Ok(st) = &o.tailscale.status {
        if st.hostname != t.hostname {
            out.push(Change::new(
                "tailscale",
                "update",
                "hostname",
                format!("{} -> {}", st.hostname, t.hostname),
            ));
        }
        if st.magic_dns_suffix != t.tailnet {
            out.push(Change::new(
                "tailscale",
                "update",
                "tailnet",
                format!("{} -> {}", st.magic_dns_suffix, t.tailnet),
            ));
        }
    }
    let Ok(serve) = &o.tailscale.serve else {
        return out;
    };
    let fqdn = format!("{}.{}", t.hostname, t.tailnet);
    let mut actual: BTreeMap<(u16, String), &super::observe::ServedRoute> = BTreeMap::new();
    for r in &serve.routes {
        actual.insert((r.port, r.path.clone()), r);
    }
    for want in &t.routes {
        let target = format!(":{}{}", want.https_port, want.path);
        let owner = want
            .owner
            .as_deref()
            .map(|o| format!(" ({o})"))
            .unwrap_or_default();
        let proxy = normalize_proxy(&want.proxy).unwrap_or_else(|| want.proxy.clone());
        match actual.remove(&(want.https_port, want.path.clone())) {
            None => out.push(Change::new(
                "tailscale",
                "add",
                target,
                format!("serve -> {proxy}{owner}"),
            )),
            Some(got) => {
                let got_proxy = got.proxy.as_deref().and_then(normalize_proxy);
                if got_proxy.as_deref() != Some(proxy.as_str()) {
                    out.push(Change::new(
                        "tailscale",
                        "update",
                        target.clone(),
                        format!("{} -> {proxy}{owner}", got.describe),
                    ));
                }
                if !got.https {
                    out.push(Change::new(
                        "tailscale",
                        "update",
                        target.clone(),
                        "http -> https".to_string(),
                    ));
                }
                if got.host != fqdn {
                    out.push(Change::new(
                        "tailscale",
                        "update",
                        target,
                        format!("served as {} -> {fqdn}", got.host),
                    ));
                }
            }
        }
    }
    for ((port, path), got) in actual {
        out.push(Change::new(
            "tailscale",
            "remove",
            format!(":{port}{path}"),
            format!("unlisted route {}", got.describe),
        ));
    }
    for f in &serve.funnel {
        out.push(Change::new(
            "tailscale",
            "update",
            format!("funnel {f}"),
            "on -> off",
        ));
    }
    out
}

pub fn supabase_changes(c: &BoxConfig, o: &Observed) -> Result<Vec<Change>, String> {
    let s = &c.supabase;
    let sb = &o.supabase;
    let mut out = Vec::new();
    let project = sb.project.clone()?;
    let Some(project) = project else {
        out.push(Change::new(
            "supabase",
            "install",
            s.project.clone(),
            format!("{} into {}", s.upstream_ref, s.dir.display()),
        ));
        return Ok(out);
    };
    let compose = s.dir.join("docker-compose.yml");
    if !project
        .config_files
        .iter()
        .any(|f| std::path::Path::new(f) == compose)
    {
        out.push(Change::new(
            "supabase",
            "update",
            "project dir",
            format!("{} -> {}", project.config_files.join(","), s.dir.display()),
        ));
    }
    let overlay = s.dir.join(&s.overlay);
    if !sb.overlay_present
        || !project
            .config_files
            .iter()
            .any(|f| std::path::Path::new(f) == overlay)
    {
        out.push(Change::new(
            "supabase",
            "update",
            "overlay",
            format!("load {}", s.overlay),
        ));
    }
    match &sb.stamp {
        Some(st) if st.ref_ != s.upstream_ref => out.push(Change::new(
            "supabase",
            "upgrade",
            "version",
            format!("{} -> {}", st.ref_, s.upstream_ref),
        )),
        Some(st)
            if st
                .commit
                .as_deref()
                .is_some_and(|commit| commit != s.upstream_commit) =>
        {
            out.push(Change::new(
                "supabase",
                "upgrade",
                "version",
                format!(
                    "commit {} -> {}",
                    st.commit.as_deref().unwrap_or(""),
                    s.upstream_commit
                ),
            ))
        }
        Some(_) => {}
        None => out.push(Change::new(
            "supabase",
            "update",
            "version",
            format!("unknown -> {}", s.upstream_ref),
        )),
    }
    let containers = sb.containers.clone()?;
    let services = sb.compose_services.clone()?;
    for (name, svc) in &services {
        if svc.has_profile {
            continue;
        }
        let Some(ct) = containers.iter().find(|ct| &ct.service == name) else {
            out.push(Change::new(
                "supabase",
                "start",
                name.clone(),
                "service is not running",
            ));
            continue;
        };
        let mut why = Vec::new();
        if let Some(img) = svc.image.as_deref().filter(|i| !i.contains('$')) {
            if img != ct.image {
                why.push(format!("image {} -> {img}", ct.image));
            }
        }
        if ct.cgroup_parent != s.slice {
            why.push(format!("slice {:?} -> {}", ct.cgroup_parent, s.slice));
        }
        if name == "db" && ct.oom_score_adj >= 0 {
            why.push(format!("oom_score_adj {} -> -900", ct.oom_score_adj));
        }
        for b in ct.bindings.iter().filter(|b| !b.is_loopback()) {
            let ip = if b.host_ip.is_empty() {
                "0.0.0.0"
            } else {
                &b.host_ip
            };
            why.push(format!(
                "bind {ip}:{} -> 127.0.0.1:{}",
                b.host_port, b.host_port
            ));
        }
        if !why.is_empty() {
            out.push(Change::new(
                "supabase",
                "recreate",
                name.clone(),
                why.join(", "),
            ));
        }
    }
    out.extend(floor_change(c, o));
    if let Ok(reg) = &o.models.registry {
        let want = c.slice_cgroup();
        match reg
            .reservations
            .iter()
            .find(|r| r.cgroup.as_deref() == Some(want.as_str()))
        {
            Some(r) if (r.gb - s.reservation_gb).abs() < 0.05 => {}
            Some(r) => out.push(Change::new(
                "supabase",
                "update",
                format!("reservation {}", r.id),
                format!("{} GB -> {} GB", r.gb, s.reservation_gb),
            )),
            None => out.push(Change::new(
                "supabase",
                "add",
                "reservation",
                format!("{} GB for {want}", s.reservation_gb),
            )),
        }
    }
    Ok(out)
}

/// The Supabase memory floor, as drop-ins for every slice on the path. Applied
/// on its own by `lumbridge-computer apply --only supabase-floor`.
pub fn floor_change(c: &BoxConfig, o: &Observed) -> Option<Change> {
    let facts = &o.supabase.floor;
    let p = floor::plan(c, facts);
    if p.satisfied {
        return None;
    }
    let reload = if p.files.is_empty() {
        "drop-ins in place; systemctl daemon-reload"
    } else {
        "then systemctl daemon-reload (root)"
    };
    let mut change = Change::new(
        "supabase",
        "update",
        "memory floor",
        format!(
            "MemoryMin={} on {} (now {}); {reload}",
            floor::systemd_size(floor::target_bytes(c)),
            floor::slice_chain(&c.supabase.slice).join(", "),
            floor::describe(facts),
        ),
    );
    change.files = p.files;
    Some(change)
}

pub fn model_changes(c: &BoxConfig, o: &Observed) -> Result<Vec<Change>, String> {
    let m = &o.models;
    let desired = &c.models.desired_scene;
    let plan = m.plan.clone()?;
    let state = m.state.clone()?;
    let mut out = Vec::new();
    if let Some(t) = &state.transition {
        return Err(format!("a transition to {t} is in progress"));
    }
    if state.active.as_deref() != Some(desired.as_str()) {
        out.push(Change::new(
            "models",
            "activate",
            format!("scene {desired}"),
            format!(
                "active {} -> {desired}",
                state.active.as_deref().unwrap_or("none")
            ),
        ));
    }
    for id in &plan.stop {
        out.push(Change::new(
            "models",
            "stop",
            id.clone(),
            format!("not in scene {desired}"),
        ));
    }
    for id in &plan.start {
        out.push(Change::new(
            "models",
            "start",
            id.clone(),
            format!("in scene {desired}, not serving"),
        ));
    }
    Ok(out)
}

pub fn render_text(d: &DryRun) -> String {
    if d.is_clean() {
        return "no changes\n".into();
    }
    let mut out = String::new();
    for ch in &d.changes {
        let sigil = match ch.action {
            "add" | "install" | "start" => '+',
            "remove" | "stop" => '-',
            _ => '~',
        };
        out.push_str(&format!(
            "{sigil} {:<9} {:<8} {:<24} {}\n",
            ch.area, ch.action, ch.target, ch.detail
        ));
        for f in &ch.files {
            out.push_str(&format!("    write {}\n", f.path.display()));
            for line in f.contents.lines() {
                out.push_str(&format!("      | {line}\n"));
            }
        }
    }
    for b in &d.blockers {
        out.push_str(&format!("! cannot diff {b}\n"));
    }
    out.push_str(&format!(
        "\n{} change(s){}; nothing was changed (dry run)\n",
        d.changes.len(),
        if d.blockers.is_empty() {
            String::new()
        } else {
            format!(", {} area(s) unreadable", d.blockers.len())
        }
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::super::doctor::tests::{config, observe};
    use super::super::observe::{fixtures, fixtures::*, INSPECT_FORMAT};
    use super::*;

    /// The fixture box with the memory floor applied, so tests of other areas see only their
    /// own changes. The tests of the floor itself use the fixture as it is.
    fn box1() -> super::super::cmd::Fake {
        let mut h = fixtures::box1();
        for dir in ["lumbridge.slice", "lumbridge.slice/lumbridge-server.slice"] {
            h.file(&format!("/sys/fs/cgroup/{dir}/memory.min"), "4294967296\n");
        }
        h
    }

    #[test]
    fn box1_as_it_is_needs_only_the_memory_floor() {
        let d = dry_run(&config(), &observe(&fixtures::box1()));
        assert!(d.blockers.is_empty(), "{}", render_text(&d));
        let got: Vec<String> = d.changes.iter().map(Change::summary).collect();
        assert_eq!(
            got,
            vec![
                "update memory floor: MemoryMin=4G on lumbridge.slice, lumbridge-server.slice \
                 (now lumbridge.slice min 0.0 / low 0.0 GB, lumbridge-server.slice min 0.0 / low \
                 0.0 GB); then systemctl daemon-reload (root)"
            ]
        );
        let text = render_text(&d);
        assert!(text.contains(
            "    write /etc/systemd/system/lumbridge-server.slice.d/50-lumbridge-computer-memory-min.conf\n"
        ));
        assert!(
            text.contains("      | [Slice]\n      | MemoryMin=4G\n"),
            "{text}"
        );
    }

    #[test]
    fn box1_with_the_floor_in_place_needs_no_changes() {
        let d = dry_run(&config(), &observe(&box1()));
        assert!(d.is_clean(), "{}", render_text(&d));
        assert_eq!(render_text(&d), "no changes\n");
    }

    #[test]
    fn the_floor_change_serializes_its_files_for_json_output() {
        let d = dry_run(&config(), &observe(&fixtures::box1()));
        let j = serde_json::to_value(&d).unwrap();
        let files = &j["changes"][0]["files"];
        assert_eq!(files.as_array().unwrap().len(), 2);
        assert!(files[1]["contents"]
            .as_str()
            .unwrap()
            .contains("MemoryMin=4G"));
    }

    #[test]
    fn missing_extra_and_changed_routes_are_add_remove_update() {
        let mut c = config();
        c.tailscale.routes.retain(|r| r.https_port != 8484);
        c.tailscale.routes[0].proxy = "8021".into();
        c.tailscale.routes.push(super::super::config::Route {
            https_port: 9443,
            path: "/".into(),
            proxy: "9000".into(),
            owner: None,
        });
        let d = dry_run(&c, &observe(&box1()));
        let got: Vec<(&str, &str)> = d
            .changes
            .iter()
            .map(|c| (c.action, c.target.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("update", ":10000/"),
                ("add", ":9443/"),
                ("remove", ":8484/")
            ]
        );
    }

    #[test]
    fn funnel_on_is_a_change() {
        let mut h = box1();
        let serve = TS_SERVE.trim_end().trim_end_matches('}').to_string()
            + ",\"AllowFunnel\":{\"box1.tail0000.ts.net:443\":true}}";
        h.on("tailscale serve status --json", &serve);
        let d = dry_run(&config(), &observe(&h));
        assert!(d.changes.iter().any(|c| c.target.starts_with("funnel")));
    }

    #[test]
    fn a_missing_project_is_one_install() {
        let mut h = box1();
        h.on("docker compose ls --format json", "[]");
        let d = dry_run(&config(), &observe(&h));
        assert_eq!(d.changes.len(), 1);
        assert_eq!(d.changes[0].action, "install");
    }

    #[test]
    fn a_pin_ahead_of_the_deployment_is_an_upgrade() {
        let mut c = config();
        c.supabase.upstream_ref = "self-hosted/v0.8.5".into();
        let d = dry_run(&c, &observe(&box1()));
        assert_eq!(
            d.changes[0].summary(),
            "upgrade version: self-hosted/v0.8.4 -> self-hosted/v0.8.5"
        );
    }

    #[test]
    fn overlay_drift_recreates_the_affected_services() {
        let mut h = box1();
        let inspect = INSPECT
            .replace("\"oom_score_adj\":-900", "\"oom_score_adj\":0")
            .replace(
                "\"HostIp\":\"127.0.0.1\",\"HostPort\":\"8000\"",
                "\"HostIp\":\"0.0.0.0\",\"HostPort\":\"8000\"",
            )
            .replacen(
                "\"cgroup_parent\":\"lumbridge-server.slice\"",
                "\"cgroup_parent\":\"\"",
                1,
            );
        h.on(
            &format!("docker inspect --format {INSPECT_FORMAT}"),
            &inspect,
        );
        let d = dry_run(&config(), &observe(&h));
        let targets: Vec<&str> = d.changes.iter().map(|c| c.target.as_str()).collect();
        assert_eq!(
            targets,
            vec!["api-gw", "db", "realtime"],
            "{}",
            render_text(&d)
        );
        assert!(d.changes.iter().all(|c| c.action == "recreate"));
    }

    #[test]
    fn a_stopped_service_is_a_start() {
        let mut h = box1();
        let inspect: String = INSPECT
            .lines()
            .filter(|l| !l.contains("\"service\":\"studio\""))
            .map(|l| format!("{l}\n"))
            .collect();
        h.on(
            &format!("docker inspect --format {INSPECT_FORMAT}"),
            &inspect,
        );
        let d = dry_run(&config(), &observe(&h));
        assert_eq!(
            d.changes,
            vec![Change::new(
                "supabase",
                "start",
                "studio",
                "service is not running"
            )]
        );
    }

    #[test]
    fn scene_drift_lists_activation_stops_and_starts() {
        let mut o = observe(&box1());
        o.models.state.as_mut().unwrap().active = Some("music".into());
        let plan = o.models.plan.as_mut().unwrap();
        plan.stop = vec!["music-gen".into()];
        plan.start = vec!["embed".into()];
        let d = dry_run(&config(), &o);
        let got: Vec<&str> = d.changes.iter().map(|c| c.action).collect();
        assert_eq!(got, vec!["activate", "stop", "start"]);
    }

    #[test]
    fn unreadable_state_is_a_blocker_not_no_changes() {
        let mut h = box1();
        h.fail("tailscale", "tailscaled not running");
        let mut o = observe(&h);
        o.models.plan = Err("refused".into());
        let d = dry_run(&config(), &o);
        assert!(!d.is_clean());
        assert_eq!(d.blockers.len(), 2);
        assert!(render_text(&d).contains("cannot diff tailscale"));
    }
}
