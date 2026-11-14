//! The gateway's route table: which name goes to which upstream right now.
//!
//! Built from the registry plus the persisted desired Scene: every model in that
//! Scene whose serving port accepts connections is routable under its registry id
//! and each of its `served_name`s. A Scene model that isn't answering gets a 503
//! (it should be back); a registry model outside the Scene gets a 404 that says
//! so; anything else is a plain unknown model.
//!
//! With M3 on-demand enabled, a Scene model whose mode is `sleepable` or
//! `on_demand` is routable even while its port is closed: the gateway holds the
//! request and asks the lifecycle to wake or start it.

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::config::{find_scene, Model, Registry};
use crate::{governor, proc};

#[derive(Debug, PartialEq, Eq)]
pub struct Target {
    /// Registry id: the in-flight counter key and the `x-lumbridge-computer-model` value.
    pub id: String,
    /// `host:port` the request is sent to.
    pub addr: String,
    /// `Some(name)` when the model opted into `gateway.rewrite_model`.
    pub rewrite_to: Option<String>,
    /// The model's mode in the Scene (always `Resident` without on-demand).
    pub mode: SceneMode,
    /// `serve.sleep.wake_timeout_sec` for a model that sleeps: how long a
    /// request is held while it wakes from sleep. A cold start uses the
    /// gateway's own `--wake-timeout-sec`.
    pub sleep_wake_timeout: Option<Duration>,
}

/// A model's mode within a Scene (M3): the one Scene format, `config::Mode`.
pub use crate::config::Mode as SceneMode;

pub enum Lookup<'a> {
    Running(&'a Arc<Target>),
    /// A `sleepable`/`on_demand` Scene model whose port is closed: wake it.
    Dormant(&'a Arc<Target>),
    /// In the Scene but not answering: transient, worth retrying.
    NotRunning {
        id: &'a str,
    },
    /// In the registry but not in the Scene: it won't come back on its own.
    NotInScene {
        id: &'a str,
    },
    Unknown,
}

#[derive(Debug, Default)]
pub struct Routes {
    /// Every routable name: running models, plus dormant on-demand ones.
    running: HashMap<String, Arc<Target>>,
    /// Ids of routable models whose port is closed (on-demand only).
    dormant: HashSet<String>,
    /// Every registry name → (model id, model is in the Scene).
    known: HashMap<String, (String, bool)>,
    /// The Scene the table was built from, for error messages.
    pub scene: Option<String>,
    /// Routable names in a stable order, with their model id, for `/v1/models`.
    listed: Vec<(String, String)>,
    /// Config problems found while building. Reported through `summary()`, which
    /// is logged only when it changes, so a bad field isn't logged every refresh.
    warnings: Vec<String>,
    pub created: u64,
}

impl Routes {
    /// `scene_models` in Scene order; `is_running` decides which of them route.
    /// Every model is `resident` (the pre-M3 table).
    #[cfg(test)]
    pub fn build(
        registry: &Registry,
        scene_models: &[String],
        is_running: impl FnMut(&Model) -> bool,
    ) -> Routes {
        Routes::build_with_modes(registry, scene_models, &HashMap::new(), is_running)
    }

    /// Like `build`, but a model whose `modes` entry wakes on demand is routable
    /// even when `is_running` says no (it is looked up as `Dormant`).
    pub fn build_with_modes(
        registry: &Registry,
        scene_models: &[String],
        modes: &HashMap<String, SceneMode>,
        mut is_running: impl FnMut(&Model) -> bool,
    ) -> Routes {
        let mut routes = Routes {
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            ..Routes::default()
        };
        for (id, model) in &registry.models {
            for name in names(id, model) {
                let in_scene = scene_models.contains(id);
                routes
                    .known
                    .entry(name.to_string())
                    .or_insert((id.clone(), in_scene));
            }
        }

        let mut targets = Vec::new();
        for id in scene_models {
            let Some(model) = registry.models.get(id) else {
                continue;
            };
            let Some(addr) = upstream_addr(id, model, &mut routes.warnings) else {
                continue;
            };
            let mode = modes.get(id).copied().unwrap_or_default();
            if !is_running(model) {
                if !mode.wakes_on_demand() {
                    continue;
                }
                routes.dormant.insert(id.clone());
            }
            let rewrite_to = model
                .gateway
                .rewrite_model
                .then(|| model.serve.served_name.first().unwrap_or(id).to_string());
            targets.push((
                model,
                Arc::new(Target {
                    id: id.clone(),
                    addr,
                    rewrite_to,
                    mode,
                    sleep_wake_timeout: model
                        .serve
                        .sleep
                        .as_ref()
                        .map(|sleep| sleep.wake_timeout()),
                }),
            ));
        }
        // Registry ids win over served names, so `brain` always means the model
        // with id `brain` even if another model also advertises that name.
        for (_, target) in &targets {
            routes.insert(&target.id.clone(), target);
        }
        for (model, target) in &targets {
            for name in &model.serve.served_name {
                routes.insert(name, target);
            }
        }
        routes
    }

    fn insert(&mut self, name: &str, target: &Arc<Target>) {
        match self.running.get(name) {
            Some(existing) if existing.id != target.id => self.warnings.push(format!(
                "name '{name}' is served by both '{}' and '{}'; routing it to '{}'",
                existing.id, target.id, existing.id
            )),
            Some(_) => {}
            None => {
                self.running.insert(name.to_string(), target.clone());
                self.listed.push((name.to_string(), target.id.clone()));
            }
        }
    }

    pub fn lookup(&self, name: &str) -> Lookup<'_> {
        if let Some(target) = self.running.get(name) {
            if self.dormant.contains(&target.id) {
                return Lookup::Dormant(target);
            }
            return Lookup::Running(target);
        }
        match self.known.get(name) {
            Some((id, true)) => Lookup::NotRunning { id },
            Some((id, false)) => Lookup::NotInScene { id },
            None => Lookup::Unknown,
        }
    }

    /// The running model behind `addr`, so model-less requests sent to the
    /// default upstream still count against that model's in-flight total.
    pub fn by_addr(&self, addr: &str) -> Option<&Arc<Target>> {
        self.listed
            .iter()
            .filter_map(|(name, _)| self.running.get(name))
            .find(|target| target.addr == addr)
    }

    /// `(name, model id)` for every routable name.
    pub fn listed(&self) -> &[(String, String)] {
        &self.listed
    }

    /// One line per running model for the log when the table changes.
    pub fn summary(&self) -> String {
        let mut seen = HashSet::new();
        let parts: Vec<String> = self
            .listed
            .iter()
            .filter_map(|(name, _)| self.running.get(name))
            .filter(|target| seen.insert(target.id.clone()))
            .map(|target| match target.mode {
                SceneMode::Resident => format!("{}->{}", target.id, target.addr),
                mode => format!("{}->{} ({})", target.id, target.addr, mode.as_str()),
            })
            .collect();
        let mut summary = if parts.is_empty() {
            "no running models".to_string()
        } else {
            parts.join(" ")
        };
        for warning in &self.warnings {
            summary.push_str("; warning: ");
            summary.push_str(warning);
        }
        summary
    }
}

fn names<'a>(id: &'a str, model: &'a Model) -> impl Iterator<Item = &'a str> {
    std::iter::once(id).chain(model.serve.served_name.iter().map(String::as_str))
}

fn upstream_addr(id: &str, model: &Model, warnings: &mut Vec<String>) -> Option<String> {
    match &model.gateway.upstream {
        // Parsed as a socket address (no DNS) so a typo is caught here, once, in
        // the log, rather than as a 502 on every request.
        Some(upstream) => match upstream.parse::<SocketAddr>() {
            Ok(addr) => Some(addr.to_string()),
            Err(_) => {
                warnings.push(format!(
                    "model '{id}' has gateway.upstream '{upstream}', which is not ip:port; using its serve.port"
                ));
                model.serve.port.map(|port| format!("127.0.0.1:{port}"))
            }
        },
        None => model.serve.port.map(|port| format!("127.0.0.1:{port}")),
    }
}

/// Build the live table from disk. A model counts as running when its serving
/// port accepts a TCP connection: models in one Scene never share a port, so the
/// connect is enough, and unlike the `health` probe it puts no request in the
/// engines' access logs every refresh.
///
/// `on_demand`: read the Scene's per-model modes (only when a lifecycle can wake
/// models; otherwise every model is treated as `resident`, as before M3).
pub fn load(root: &Path, on_demand: bool) -> Result<Routes> {
    let registry = Registry::load(root)?;
    let state = proc::State::load_checked(root)?;
    let scene = state.desired_scene.or(state.active_scene);
    // Modes come from the Scene itself (`modes:`); without a lifecycle nothing
    // can wake a model, so every model is treated as resident, as before M3.
    let (scene_models, modes) = match &scene {
        Some(name) => {
            let scene = find_scene(root, name)?;
            let modes: HashMap<String, SceneMode> = if on_demand {
                scene
                    .modes
                    .iter()
                    .map(|(id, mode)| (id.clone(), mode.mode))
                    .collect()
            } else {
                HashMap::new()
            };
            (scene.models, modes)
        }
        None => (Vec::new(), HashMap::new()),
    };
    let mut routes = Routes::build_with_modes(&registry, &scene_models, &modes, |model| {
        model.serve.port.is_some_and(governor::port_open)
    });
    routes.scene = scene;
    Ok(routes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> Registry {
        serde_yaml::from_str(
            "apiVersion: lumbridge/v1
models:
  brain:
    name: Osprey stock
    footprint_gb: 40
    gateway:
      upstream: 127.0.0.1:8041
    serve: {kind: vllm, port: 8001, served_name: [brain, osprey]}
  brain-alt:
    name: Osprey fine-tune
    footprint_gb: 40
    gateway:
      rewrite_model: true
    serve: {kind: vllm, port: 8002, served_name: [osprey-alt, brain]}
  embed:
    name: Embed
    footprint_gb: 4
    serve: {kind: vllm, port: 8012}
",
        )
        .unwrap()
    }

    fn scene() -> Vec<String> {
        vec!["brain".into(), "brain-alt".into(), "embed".into()]
    }

    #[test]
    fn running_scene_models_route_by_id_and_served_name() {
        let routes = Routes::build(&registry(), &scene(), |m| m.serve.port != Some(8012));
        let Lookup::Running(brain) = routes.lookup("osprey") else {
            panic!("osprey should route");
        };
        assert_eq!(brain.id, "brain");
        assert_eq!(
            brain.addr, "127.0.0.1:8041",
            "gateway.upstream overrides the port"
        );
        let Lookup::Running(tuned) = routes.lookup("brain-alt") else {
            panic!("brain-alt should route");
        };
        assert_eq!(tuned.addr, "127.0.0.1:8002");
        assert_eq!(tuned.rewrite_to.as_deref(), Some("osprey-alt"));
        // `brain` is brain-alt's served name too, but the registry id wins.
        let Lookup::Running(by_id) = routes.lookup("brain") else {
            panic!("brain should route");
        };
        assert_eq!(by_id.id, "brain");
        assert!(matches!(
            routes.lookup("embed"),
            Lookup::NotRunning { id: "embed" }
        ));
        assert!(matches!(routes.lookup("gpt-4"), Lookup::Unknown));
        assert_eq!(routes.by_addr("127.0.0.1:8041").unwrap().id, "brain");
        assert_eq!(
            routes.summary(),
            "brain->127.0.0.1:8041 brain-alt->127.0.0.1:8002; warning: name 'brain' is served by both 'brain' and 'brain-alt'; routing it to 'brain'"
        );
    }

    #[test]
    fn on_demand_scene_models_route_while_down_but_resident_ones_do_not() {
        let modes = HashMap::from([
            ("brain-alt".to_string(), SceneMode::OnDemand),
            ("brain".to_string(), SceneMode::Sleepable),
        ]);
        let routes = Routes::build_with_modes(&registry(), &scene(), &modes, |m| {
            m.serve.port == Some(8001)
        });
        let Lookup::Running(brain) = routes.lookup("osprey") else {
            panic!("a sleepable model with an open port is running");
        };
        assert_eq!(brain.mode, SceneMode::Sleepable);
        let Lookup::Dormant(tuned) = routes.lookup("osprey-alt") else {
            panic!("an on_demand model with a closed port is dormant");
        };
        assert_eq!(tuned.id, "brain-alt");
        assert!(matches!(
            routes.lookup("embed"),
            Lookup::NotRunning { id: "embed" }
        ));
        assert_eq!(
            routes.listed().len(),
            4,
            "dormant models are listed in /v1/models"
        );
        assert!(routes.summary().starts_with(
            "brain->127.0.0.1:8041 (sleepable) brain-alt->127.0.0.1:8002 (on_demand)"
        ));
    }

    #[test]
    fn registry_models_outside_the_scene_are_known_but_not_routed() {
        let routes = Routes::build(&registry(), &["embed".to_string()], |_| true);
        assert!(matches!(
            routes.lookup("osprey"),
            Lookup::NotInScene { id: "brain" }
        ));
        assert_eq!(
            routes.listed(),
            &[("embed".to_string(), "embed".to_string())]
        );
    }
}
