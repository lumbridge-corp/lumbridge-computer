//! MCP server: the Governor's surface, exposed to agents as tools.
//!
//! Agents already drive Lumbridge Computer by shelling out to the CLI and scraping the
//! column-aligned output. That works until a column moves. Speaking MCP means
//! the agent gets the same numbers the Governor reasons about, as data, with a
//! schema attached — and it means the *shape* of what an agent may do becomes
//! something this file decides rather than something `bash` decides.
//!
//! Two deliberate constraints shape everything below.
//!
//! **Read-first.** Every tool here except `scene_activate` is a pure read.
//! Activating a Scene stops every registered model that is not in it, which on
//! this box means taking down whatever is currently serving — a live voice
//! pipeline included. There is no undo an agent can reach for: if the target
//! Scene then fails to come up, recovery depends on rollback that may itself
//! fail. That asymmetry is why `scene adopt`, `scene resume`, and
//! `scene deactivate` are absent entirely; they are operator verbs whose
//! correctness depends on knowing what the box was doing five minutes ago.
//! `scene_activate` is exposed because planning a switch is genuinely the
//! useful thing an agent wants, and it defaults to planning only.
//!
//! **The safety boundary is the operator's, not the agent's.** A `dry_run`
//! argument defaulting to `true` documents intent but guards nothing: the agent
//! writes the arguments, so it can write `false`. The only boundary an agent
//! cannot cross is one set before it connects, so a real transition also
//! requires `lumbridge-computer mcp --allow-activate`, chosen by the human who
//! launched the server. Without that flag `dry_run: false` is refused, and the
//! refusal says so rather than silently planning instead.

use anyhow::{Context, Result};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_handler, tool_router, ErrorData, Json, ServerHandler, ServiceExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::os::fd::FromRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::config::{find_scene, load_scenes, Registry};
use crate::proc::State;
use crate::{eval, gateway, governor, lifecycle};

pub fn run(root: &Path, allow_activate: bool) -> Result<()> {
    let transport_stdout = hand_over_stdout()?;
    // The banner has to go to stderr for the same reason everything else does;
    // it doubles as confirmation to the operator that the mutating tool is off.
    eprintln!(
        "Lumbridge Computer MCP server on stdio · root {} · scene activation {}",
        root.display(),
        if allow_activate {
            "ENABLED (--allow-activate)"
        } else {
            "disabled; plan only"
        }
    );

    let server = ComputeMcp::new(root.to_path_buf(), allow_activate);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building the MCP runtime")?;
    runtime.block_on(async move {
        let transport = (
            tokio::io::stdin(),
            tokio::fs::File::from_std(transport_stdout),
        );
        let service = server
            .serve(transport)
            .await
            .context("negotiating the MCP stdio session")?;
        service.waiting().await.context("serving MCP over stdio")?;
        Ok(())
    })
}

/// Hand the real stdout to the transport and point fd 1 at stderr.
///
/// On stdio transport fd 1 *is* the JSON-RPC framing, and this crate reports
/// progress with `println!` throughout — `lifecycle::activate` narrates every
/// stop and start. One such line interleaved into the framing desynchronises
/// the client mid-transition, which is the worst possible moment to lose it.
/// Rather than audit every print (and every future one), move the file
/// descriptor: library output lands on stderr, where operators already read
/// this server's logs, and the protocol gets a channel nothing else can write.
fn hand_over_stdout() -> Result<std::fs::File> {
    let saved = unsafe { libc::dup(libc::STDOUT_FILENO) };
    if saved < 0 {
        return Err(std::io::Error::last_os_error())
            .context("duplicating stdout for the MCP transport");
    }
    if unsafe { libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) } < 0 {
        let error = std::io::Error::last_os_error();
        unsafe { libc::close(saved) };
        return Err(error).context("redirecting stdout to stderr");
    }
    // SAFETY: `saved` is a fresh descriptor from `dup` that nothing else owns.
    Ok(unsafe { std::fs::File::from_raw_fd(saved) })
}

#[derive(Clone)]
struct ComputeMcp {
    root: Arc<PathBuf>,
    allow_activate: bool,
    tool_router: ToolRouter<ComputeMcp>,
}

#[tool_router]
impl ComputeMcp {
    fn new(root: PathBuf, allow_activate: bool) -> Self {
        Self {
            root: Arc::new(root),
            allow_activate,
            tool_router: Self::tool_router(),
        }
    }

    /// Governor status for this node: unified-memory totals, the hard budget
    /// and its margins, how much is committed by models that are actually
    /// serving, how much is still admittable, the persisted Scene state, and
    /// the last transition or watchdog error. Read this before reasoning about
    /// whether anything else will fit.
    #[tool(
        name = "governor_status",
        annotations(
            title = "Governor status",
            read_only_hint = true,
            open_world_hint = false
        )
    )]
    async fn governor_status(&self) -> Result<Json<StatusReport>, ErrorData> {
        let root = self.root.clone();
        offload(move || collect_status(&root)).await.map(Json)
    }

    /// List every model in the node's registry with its declared worst-case
    /// footprint, serving port, and whether it is currently serving. Footprints
    /// are what admission control is decided against, not observed usage.
    #[tool(
        name = "model_list",
        annotations(
            title = "List registered models",
            read_only_hint = true,
            open_world_hint = false
        )
    )]
    async fn model_list(&self) -> Result<Json<ModelList>, ErrorData> {
        let root = self.root.clone();
        offload(move || collect_models(&root)).await.map(Json)
    }

    /// List the Scenes this node can activate, with each one's total footprint
    /// and member models. A Scene is a named set of models brought up as a
    /// single unit.
    #[tool(
        name = "scene_list",
        annotations(title = "List scenes", read_only_hint = true, open_world_hint = false)
    )]
    async fn scene_list(&self) -> Result<Json<SceneList>, ErrorData> {
        let root = self.root.clone();
        offload(move || collect_scenes(&root)).await.map(Json)
    }

    /// Show one Scene: its models, their footprints and live state, and the
    /// Governor's admission verdict — whether the Scene fits its budget once
    /// the safety margin is added, and by how much it fits or misses.
    #[tool(
        name = "scene_show",
        annotations(
            title = "Show a scene and its admission verdict",
            read_only_hint = true,
            open_world_hint = false
        )
    )]
    async fn scene_show(
        &self,
        Parameters(params): Parameters<SceneNameParams>,
    ) -> Result<Json<SceneDetail>, ErrorData> {
        let root = self.root.clone();
        offload(move || collect_scene(&root, &params.name))
            .await
            .map(Json)
    }

    /// List the evaluation suites available on this node, with their versions
    /// and case counts. Running a suite is deliberately not exposed: it drives
    /// real load against a serving model for an unbounded time.
    #[tool(
        name = "eval_list",
        annotations(
            title = "List eval suites",
            read_only_hint = true,
            open_world_hint = false
        )
    )]
    async fn eval_list(&self) -> Result<Json<EvalList>, ErrorData> {
        let root = self.root.clone();
        offload(move || collect_evals(&root)).await.map(Json)
    }

    /// Plan — or, if the operator allowed it, perform — a Scene transition.
    ///
    /// With `dry_run` true (the default) this runs every check that guards a
    /// real transition and returns the models it would stop and start, changing
    /// nothing. With `dry_run` false it actually switches the node: models
    /// outside the Scene are stopped, which will interrupt anything they serve,
    /// and the call blocks until every Scene model reports healthy or the
    /// transition rolls back. A real transition additionally requires the
    /// server to have been started with `--allow-activate`; without it,
    /// `dry_run: false` is refused. Ask a human before setting it.
    #[tool(
        name = "scene_activate",
        annotations(
            title = "Activate a scene (plans by default)",
            read_only_hint = false,
            destructive_hint = true,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn scene_activate(
        &self,
        Parameters(params): Parameters<SceneActivateParams>,
    ) -> Result<Json<ActivationReport>, ErrorData> {
        if !params.dry_run && !self.allow_activate {
            return Err(ErrorData::invalid_params(
                "this MCP server is running plan-only: a real Scene transition needs an \
                 operator to restart it as `lumbridge-computer mcp --allow-activate`. \
                 Re-run with dry_run: true to see the plan.",
                None,
            ));
        }
        let root = self.root.clone();
        offload(move || activate_scene(&root, &params.name, params.dry_run))
            .await
            .map(Json)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ComputeMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                "lumbridge-computer",
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Lumbridge Computer runs many AI models on one unified-memory box. There is a \
                 single memory pool shared by CPU and GPU, so over-committing does not fail \
                 gracefully — the machine thrashes and wedges before the OOM killer acts. The \
                 Governor prevents that by admitting a model only if its declared footprint \
                 plus a safety margin still fits a hard budget.\n\n\
                 Start from `governor_status` for what is committed and what is admittable, \
                 and `scene_show` for whether a named Scene would be admitted. Everything is \
                 read-only except `scene_activate`, which plans by default. Activating a Scene \
                 stops every registered model outside it, so treat it as a production change \
                 and get a human's agreement before asking for a non-dry run.",
            )
    }
}

// ---- tool parameters -------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct SceneNameParams {
    /// Scene name, as reported by `scene_list`.
    name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SceneActivateParams {
    /// Scene name, as reported by `scene_list`.
    name: String,
    /// Validate and report the plan without changing anything. Defaults to
    /// true; set it to false only with a human's explicit agreement.
    #[serde(default = "yes")]
    dry_run: bool,
}

fn yes() -> bool {
    true
}

// ---- tool results ----------------------------------------------------------
//
// Lists are wrapped in objects rather than returned bare: MCP structured
// content must be a JSON object, and a named field leaves room to report
// alongside the list later without breaking a client's schema.

#[derive(Debug, Serialize, JsonSchema)]
pub(crate) struct StatusReport {
    memory_total_gb: f64,
    memory_available_gb: f64,
    budget_gb: f64,
    safety_margin_gb: f64,
    watchdog_floor_gb: f64,
    /// Sum of the footprints of every registered model currently serving.
    committed_gb: f64,
    /// What a new model could still claim without breaching the safety margin.
    headroom_gb: f64,
    /// Memory promised to non-model tenants (the Factory's Server database).
    reserved_gb: f64,
    /// The part of `reserved_gb` those tenants have not used yet; held back from models.
    reserved_unused_gb: f64,
    /// Scene whose exact model health was last verified.
    active_scene: Option<String>,
    /// Scene the node should return to after a restart.
    desired_scene: Option<String>,
    /// Previous proven Scene, used if the desired one cannot resume.
    last_known_good_scene: Option<String>,
    last_error: Option<String>,
    running: Vec<RunningModel>,
    /// The resident agent's gateway, or null when no agent answers on its address.
    gateway: Option<GatewayReport>,
    /// Lifecycle state of every model in the desired Scene, and of any other
    /// model Lumbridge Computer owns a process for (M3).
    models: Vec<ModelStateReport>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub(crate) struct ModelStateReport {
    pub(crate) id: String,
    /// `resident`, `sleepable` or `on_demand` in the desired Scene; null outside it.
    pub(crate) mode: Option<String>,
    /// `stopped`, `starting`, `ready`, `draining`, `sleeping`, `waking`.
    pub(crate) state: String,
    /// Sleep level while sleeping.
    pub(crate) sleep_level: Option<u8>,
    /// Seconds since the last finished request (or since it became ready, if the
    /// gateway saw none) while ready; null when unknown or not ready.
    pub(crate) idle_sec: Option<u64>,
    /// What the model holds (or would hold) asleep; null for engines that can't sleep.
    pub(crate) sleeping_footprint_gb: Option<f64>,
    /// What admission counts it for right now.
    pub(crate) counted_gb: f64,
}

#[derive(Debug, Serialize, JsonSchema)]
pub(crate) struct GatewayReport {
    addr: String,
    /// Requests the gateway holds open per registry model id, streams included.
    /// Every running model is listed, at 0 when idle.
    in_flight: BTreeMap<String, u32>,
    total_in_flight: u32,
}

#[derive(Debug, Serialize, JsonSchema)]
struct RunningModel {
    id: String,
    name: String,
    footprint_gb: f64,
    port: Option<u16>,
    /// True when Lumbridge Computer owns this process identity and may signal it. False
    /// means the model is serving but was started outside Lumbridge Computer, so a Scene
    /// transition will refuse to touch it until it is adopted.
    compute_owned: bool,
}

#[derive(Debug, Serialize, JsonSchema)]
pub(crate) struct ModelList {
    models: Vec<ModelEntry>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ModelEntry {
    id: String,
    name: String,
    /// Declared worst-case unified memory once serving, not observed usage.
    footprint_gb: f64,
    port: Option<u16>,
    running: bool,
}

#[derive(Debug, Serialize, JsonSchema)]
pub(crate) struct SceneList {
    scenes: Vec<SceneEntry>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct SceneEntry {
    name: String,
    version: u32,
    description: String,
    /// Total footprint of the Scene's models that exist in the registry.
    footprint_gb: f64,
    models: Vec<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub(crate) struct SceneDetail {
    name: String,
    version: u32,
    description: String,
    budget_gb: f64,
    models: Vec<SceneModel>,
    /// Model ids the Scene references that the registry does not define.
    missing_models: Vec<String>,
    footprint_gb: f64,
    /// Footprint plus the Governor's safety margin — the figure compared
    /// against the budget.
    required_gb: f64,
    /// The Governor's budget verdict: do the declared footprints fit?
    admits: bool,
    verdict: String,
    /// What `scene activate` would do right now, from the same lock-free planner
    /// `apply --dry-run` uses (process ownership, ports, live memory). Null when
    /// activation would refuse; `plan_error` says why.
    plan: Option<ScenePlan>,
    plan_error: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ScenePlan {
    /// Serving models outside the Scene, stopped first.
    stop: Vec<String>,
    /// Scene models not yet serving, in the order they would be admitted.
    start: Vec<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct SceneModel {
    id: String,
    /// `resident` (default), `sleepable` or `on_demand` (M3).
    mode: String,
    /// Absent when the id is missing from the registry.
    name: Option<String>,
    footprint_gb: Option<f64>,
    running: bool,
}

#[derive(Debug, Serialize, JsonSchema)]
pub(crate) struct EvalList {
    suites: Vec<EvalSuite>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct EvalSuite {
    name: String,
    version: u32,
    cases: usize,
    description: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ActivationReport {
    scene: String,
    budget_gb: f64,
    /// Serving models outside the Scene, which the transition stops first.
    stop: Vec<String>,
    /// Scene models not yet serving, in the order they would be admitted.
    start: Vec<String>,
    /// False when this was a plan and the node was left untouched.
    applied: bool,
}

// ---- collectors ------------------------------------------------------------
//
// These compose the same primitives the CLI prints from — `governor`, `mem`,
// `config`, `proc::State` — rather than parsing the CLI's output. The numbers
// an agent sees are therefore the numbers admission control uses, by
// construction.

pub(crate) fn collect_status(root: &Path) -> Result<StatusReport> {
    collect_status_at(root, gateway::DEFAULT_LISTEN)
}

/// Status with the gateway read from `gateway_addr`. A missing agent is a normal state
/// (a box mid-cutover, a laptop), so it reads as `gateway: null`, never as an error.
pub(crate) fn collect_status_at(root: &Path, gateway_addr: &str) -> Result<StatusReport> {
    let registry = Registry::load(root)?;
    let pool = governor::Pool::unified();
    let memory = pool.sense()?;
    let managed = State::load_checked(root)?;
    // Sleeping models count at their sleeping footprint (M3).
    let committed = crate::states::committed_gb(root, &registry, &managed, &pool);
    let models = collect_model_states(
        root,
        &registry,
        &managed,
        crate::states::read_gateway_counts(gateway_addr).as_ref(),
    );
    let reserved = pool.reserved_gb(&registry);
    let reserved_unused = pool.reserved_unused_gb(&registry);
    let configured_budget = governor::scene_budget_gb(
        &pool,
        root,
        managed
            .active_scene
            .as_deref()
            .or(managed.desired_scene.as_deref()),
    )?;
    let budget = pool.effective_budget_gb(configured_budget, Some(memory.total_gb), reserved);

    let running_ids = governor::running_ids(&registry);
    let gateway = gateway_status(gateway_addr, &running_ids);
    let running = running_ids
        .into_iter()
        .map(|id| {
            let model = &registry.models[&id];
            RunningModel {
                name: model.name.clone(),
                footprint_gb: model.footprint_gb,
                port: model.serve.port,
                compute_owned: managed
                    .procs
                    .get(&id)
                    .is_some_and(|process| process.owned_alive()),
                id,
            }
        })
        .collect();

    Ok(StatusReport {
        memory_total_gb: memory.total_gb,
        memory_available_gb: memory.available_gb,
        budget_gb: budget,
        safety_margin_gb: pool.safety_margin_gb,
        watchdog_floor_gb: pool.floor_gb,
        // `+ 0.0` normalises the negative zero an empty sum can produce, which serialises as
        // "-0.0" and reads as a bug to anyone consuming this JSON.
        committed_gb: committed + 0.0,
        // The BINDING constraint, not the declared one. This used to be
        // `budget - committed - margin`, which ignores how much memory the box actually has —
        // so on a 62 GB machine with a 100 GB budget it reported 92 GB of headroom that the
        // next admission would refuse. cmd_status was fixed when admission started consulting
        // MemAvailable; this path was missed, which meant every agent driving the node over
        // MCP got the optimistic number.
        headroom_gb: pool
            .headroom_gb(
                committed,
                budget,
                Some(memory.available_gb - reserved_unused),
            )
            .max(0.0),
        reserved_gb: reserved + 0.0,
        reserved_unused_gb: reserved_unused + 0.0,
        active_scene: managed.active_scene,
        desired_scene: managed.desired_scene,
        last_known_good_scene: managed.last_known_good_scene,
        last_error: managed.last_error,
        running,
        gateway,
        models,
    })
}

/// The `models` block of `status`: one row per desired-Scene model plus any
/// other owned process, with state, idle time and sleeping footprint.
pub(crate) fn collect_model_states(
    root: &Path,
    registry: &Registry,
    managed: &State,
    counts: Option<&crate::states::GatewayCounts>,
) -> Vec<ModelStateReport> {
    use crate::states::ModelState;
    let scene = managed
        .desired_scene
        .as_deref()
        .and_then(|name| find_scene(root, name).ok());
    let mut ids: Vec<String> = scene
        .as_ref()
        .map(|scene| scene.models.clone())
        .unwrap_or_default();
    for (id, process) in &managed.procs {
        if !ids.contains(id) && process.owned_alive() {
            ids.push(id.clone());
        }
    }
    let now = crate::states::unix_now_ms();
    ids.into_iter()
        .filter_map(|id| {
            let model = registry.models.get(&id)?;
            let owned = managed.procs.get(&id).filter(|p| p.owned_alive());
            let state = match owned {
                Some(process) => process.model_state(),
                None if governor::is_running(model) => ModelState::Ready,
                None => ModelState::Stopped,
            };
            let idle_sec = (state == ModelState::Ready)
                .then(|| {
                    counts
                        .and_then(|c| c.last_finished_ms.get(&id).copied())
                        .into_iter()
                        .chain(owned.and_then(|p| p.state_since_ms))
                        .max()
                })
                .flatten()
                .map(|since| now.saturating_sub(since) / 1000);
            let sleeping_footprint_gb = crate::engines::sleep::supports_sleep(model)
                .then(|| crate::states::sleeping_gb(root, registry, &id, owned));
            Some(ModelStateReport {
                mode: scene
                    .as_ref()
                    .filter(|scene| scene.models.contains(&id))
                    .map(|scene| scene.mode_of(&id).mode.as_str().to_string()),
                state: state.as_str().to_string(),
                sleep_level: match state {
                    ModelState::Sleeping { level } => Some(level),
                    _ => None,
                },
                idle_sec,
                sleeping_footprint_gb,
                counted_gb: crate::states::counted_gb(root, registry, managed, &id) + 0.0,
                id,
            })
        })
        .collect()
}

/// Short on purpose: the agent answers from memory in microseconds, and `status` must stay
/// instant when nothing is listening or something unrelated holds the port.
const GATEWAY_CONNECT_TIMEOUT: Duration = Duration::from_millis(300);
const GATEWAY_READ_TIMEOUT: Duration = Duration::from_millis(700);

pub(crate) fn gateway_status(addr: &str, running: &[String]) -> Option<GatewayReport> {
    read_gateway(addr).map(|counts| gateway_report(addr, counts, running))
}

impl GatewayReport {
    /// One line for the human `status`: `brain 2 · embed 0`.
    pub(crate) fn summary(&self) -> String {
        if self.in_flight.is_empty() {
            return "nothing routed yet".to_string();
        }
        self.in_flight
            .iter()
            .map(|(id, n)| format!("{id} {n}"))
            .collect::<Vec<_>>()
            .join(" · ")
    }
}

fn read_gateway(addr: &str) -> Option<BTreeMap<String, u32>> {
    let body = gateway::fetch_inflight(addr, GATEWAY_CONNECT_TIMEOUT, GATEWAY_READ_TIMEOUT)?;
    parse_inflight(&body)
}

fn parse_inflight(body: &str) -> Option<BTreeMap<String, u32>> {
    #[derive(Deserialize)]
    struct Reply {
        in_flight: BTreeMap<String, u32>,
    }
    serde_json::from_str::<Reply>(body)
        .ok()
        .map(|reply| reply.in_flight)
}

/// The gateway only knows ids it has routed to; fill in running models at 0 so a reader can
/// tell "idle" from "not reported".
fn gateway_report(
    addr: &str,
    mut in_flight: BTreeMap<String, u32>,
    running: &[String],
) -> GatewayReport {
    for id in running {
        in_flight.entry(id.clone()).or_insert(0);
    }
    GatewayReport {
        addr: addr.to_string(),
        total_in_flight: in_flight.values().sum(),
        in_flight,
    }
}

pub(crate) fn collect_models(root: &Path) -> Result<ModelList> {
    let registry = Registry::load(root)?;
    Ok(ModelList {
        models: registry
            .models
            .iter()
            .map(|(id, model)| ModelEntry {
                id: id.clone(),
                name: model.name.clone(),
                footprint_gb: model.footprint_gb,
                port: model.serve.port,
                running: governor::is_running(model),
            })
            .collect(),
    })
}

pub(crate) fn collect_scenes(root: &Path) -> Result<SceneList> {
    let registry = Registry::load(root)?;
    Ok(SceneList {
        scenes: load_scenes(root)?
            .into_iter()
            .map(|scene| SceneEntry {
                name: scene.metadata.name,
                version: scene.metadata.version,
                description: scene.metadata.description,
                footprint_gb: scene
                    .models
                    .iter()
                    .filter_map(|id| registry.models.get(id))
                    .map(|model| model.footprint_gb)
                    .sum(),
                models: scene.models,
            })
            .collect(),
    })
}

pub(crate) fn collect_scene(root: &Path, name: &str) -> Result<SceneDetail> {
    let registry = Registry::load(root)?;
    let scene = find_scene(root, name)?;
    let pool = governor::Pool::unified();
    let budget = pool.effective_budget_gb(
        pool.scene_budget_gb(scene.budget_gb),
        pool.sense().ok().map(|memory| memory.total_gb),
        pool.reserved_gb(&registry),
    );

    let mut footprint = 0.0;
    let mut missing = Vec::new();
    let mut models = Vec::new();
    for id in &scene.models {
        match registry.models.get(id) {
            Some(model) => {
                footprint += model.footprint_gb;
                models.push(SceneModel {
                    id: id.clone(),
                    mode: scene.mode_of(id).mode.as_str().to_string(),
                    name: Some(model.name.clone()),
                    footprint_gb: Some(model.footprint_gb),
                    running: governor::is_running(model),
                });
            }
            None => {
                missing.push(id.clone());
                models.push(SceneModel {
                    id: id.clone(),
                    mode: scene.mode_of(id).mode.as_str().to_string(),
                    name: None,
                    footprint_gb: None,
                    running: false,
                });
            }
        }
    }

    // Same three-way verdict the CLI prints: a missing id is fatal regardless
    // of arithmetic, because the Scene cannot be resolved at all.
    // With `modes:`, sleeping models count at their sleeping footprint plus room
    // for one awake (M3); without, this is `footprint`.
    let required = crate::states::scene_need_gb(&registry, &scene, &pool) + pool.safety_margin_gb;
    let (admits, verdict) = if !missing.is_empty() {
        (
            false,
            format!("{} model(s) missing from the registry", missing.len()),
        )
    } else if required <= budget {
        (
            true,
            format!("fits with {:.1} GB to spare", budget - required),
        )
    } else {
        (
            false,
            format!("exceeds the budget by {:.1} GB", required - budget),
        )
    };

    // Never take the transition lock here: it writes under the root, and a read that
    // collides with the agent's own transition would make that transition fail.
    let (plan, plan_error) = if missing.is_empty() {
        match lifecycle::plan_read_only(root, name) {
            Ok(plan) => (
                Some(ScenePlan {
                    stop: plan.stop,
                    start: plan.start,
                }),
                None,
            ),
            Err(error) => (None, Some(format!("{error:#}"))),
        }
    } else {
        (None, Some(verdict.clone()))
    };

    Ok(SceneDetail {
        name: scene.metadata.name,
        version: scene.metadata.version,
        description: scene.metadata.description,
        budget_gb: budget,
        models,
        missing_models: missing,
        footprint_gb: footprint,
        required_gb: required,
        admits,
        verdict,
        plan,
        plan_error,
    })
}

pub(crate) fn collect_evals(root: &Path) -> Result<EvalList> {
    Ok(EvalList {
        suites: eval::catalogue(root)?
            .into_iter()
            .map(|suite| EvalSuite {
                name: suite.name,
                version: suite.version,
                cases: suite.cases,
                description: suite.description,
            })
            .collect(),
    })
}

fn activate_scene(root: &Path, name: &str, dry_run: bool) -> Result<ActivationReport> {
    // Plan first either way. On a real transition this is the plan we report;
    // `activate` re-derives and re-validates its own under the transition lock,
    // so the report describes intent and the lock still owns the truth.
    let plan = lifecycle::plan(root, name)?;
    if !dry_run {
        lifecycle::activate(root, name, false)?;
    }
    Ok(ActivationReport {
        scene: plan.scene,
        budget_gb: plan.budget_gb,
        stop: plan.stop,
        start: plan.start,
        applied: !dry_run,
    })
}

/// Run a synchronous Governor call off the transport's thread.
///
/// Every read here touches `/proc` and probes serving ports with blocking
/// socket timeouts, and a real transition can sit for minutes waiting on model
/// health. Doing that inline would stall the JSON-RPC reader for the duration,
/// so the synchronous core stays on the blocking pool where it belongs.
async fn offload<T, F>(work: F) -> Result<T, ErrorData>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(work).await {
        Ok(Ok(value)) => Ok(value),
        // `{:#}` keeps anyhow's context chain, which is where the actionable
        // half of a Lumbridge Computer error lives ("adopt the active Scene first").
        Ok(Err(error)) => Err(ErrorData::internal_error(format!("{error:#}"), None)),
        Err(join) => Err(ErrorData::internal_error(
            format!("Lumbridge Computer worker task failed: {join}"),
            None,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem;

    #[test]
    fn gateway_report_lists_idle_running_models_at_zero_and_sums_the_rest() {
        let counts = parse_inflight(r#"{"in_flight":{"brain":2,"embed":0,"old":1}}"#).unwrap();
        let report = gateway_report(
            "127.0.0.1:8011",
            counts,
            &["brain".to_string(), "voice".to_string()],
        );
        assert_eq!(report.total_in_flight, 3);
        assert_eq!(
            serde_json::to_value(&report).unwrap(),
            serde_json::json!({
                "addr": "127.0.0.1:8011",
                "in_flight": {"brain": 2, "embed": 0, "old": 1, "voice": 0},
                "total_in_flight": 3
            })
        );
    }

    #[test]
    fn a_running_gateway_is_read_over_http() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 1024];
            let n = stream.read(&mut request).unwrap();
            assert!(String::from_utf8_lossy(&request[..n])
                .starts_with(&format!("GET {} ", gateway::INFLIGHT_PATH)));
            let body = r#"{"in_flight":{"brain":1}}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        });
        let report = gateway_status(&addr, &["brain".to_string()]).unwrap();
        server.join().unwrap();
        assert_eq!(report.total_in_flight, 1);
        assert_eq!(report.summary(), "brain 1");
    }

    fn scene_root(port: u16) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("lumbridge-computer-mcp-{unique}"));
        std::fs::create_dir_all(root.join("registry")).unwrap();
        std::fs::create_dir_all(root.join("scenes")).unwrap();
        std::fs::write(
            root.join("registry/models.yaml"),
            format!(
                "apiVersion: lumbridge/v1\nmodels:\n  big:\n    name: Big\n    footprint_gb: 4\n    serve:\n      kind: exec\n      port: {port}\n      command: [/usr/bin/false]\n  small:\n    name: Small\n    footprint_gb: 1\n    serve:\n      kind: exec\n      command: [/usr/bin/false]\n"
            ),
        )
        .unwrap();
        std::fs::write(
            root.join("scenes/pair.scene.yaml"),
            "apiVersion: lumbridge/v1\nmetadata:\n  name: pair\n  version: 1\nmodels: [big, small]\nbudget_gb: 100\n",
        )
        .unwrap();
        root
    }

    #[test]
    fn scene_detail_carries_the_activation_plan_smallest_first() {
        let _memory = mem::test_memory(128.0, 100.0);
        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = free.local_addr().unwrap().port();
        drop(free);
        let root = scene_root(port);
        let detail = serde_json::to_value(collect_scene(&root, "pair").unwrap()).unwrap();
        assert_eq!(detail["admits"], true);
        assert_eq!(
            detail["plan"],
            serde_json::json!({"stop": [], "start": ["small", "big"]})
        );
        assert!(detail["plan_error"].is_null());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_model_serving_outside_lumbridge_computer_is_a_plan_error_not_a_failure() {
        let _memory = mem::test_memory(128.0, 100.0);
        // Something Lumbridge Computer does not own already holds big's port.
        let squatter = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let root = scene_root(squatter.local_addr().unwrap().port());
        let detail = serde_json::to_value(collect_scene(&root, "pair").unwrap()).unwrap();
        assert_eq!(detail["admits"], true, "the budget still fits");
        assert!(detail["plan"].is_null());
        assert!(detail["plan_error"]
            .as_str()
            .unwrap()
            .contains("not identity-owned"));
        assert!(
            !root.join(".compute").exists(),
            "showing a scene must not write under the root"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn an_unrelated_reply_on_the_gateway_port_is_no_gateway() {
        assert!(parse_inflight(r#"{"status":"ok"}"#).is_none());
        assert!(parse_inflight("<html>").is_none());
    }

    #[test]
    fn nothing_listening_reads_as_no_gateway_quickly() {
        // Bind then drop: the port is free, so the connect is refused at once.
        let addr = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let started = std::time::Instant::now();
        assert!(read_gateway(&addr.to_string()).is_none());
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}
