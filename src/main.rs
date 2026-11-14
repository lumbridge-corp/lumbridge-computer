//! Lumbridge Computer — safe AI workload orchestration for accelerator nodes.

mod agent;
mod box_;
mod config;
mod docker;
mod engines;
mod eval;
mod footprint;
mod gateway;
mod governor;
mod health;
mod http;
mod lifecycle;
mod mcp;
mod mem;
mod pool;
mod pressure;
mod proc;
mod scope;
mod snapshot;
mod states;
mod telemetry;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Duration;

use config::{find_scene, load_scenes, Registry, Scene};
use proc::State;

#[derive(Parser)]
#[command(
    name = "lumbridge-computer",
    version,
    about = "Lumbridge Computer — safe AI workload orchestration for accelerator nodes."
)]
struct Cli {
    /// Root dir containing registry/ and scenes/ (default: $LUMBRIDGE_COMPUTER_ROOT, or current dir)
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Governor status: memory, budget, running set, headroom, gateway in-flight
    Status {
        /// Print the same JSON the MCP `governor_status` tool returns
        #[arg(long)]
        json: bool,
        /// Where the resident agent's gateway listens (for in-flight counts)
        #[arg(long, default_value = gateway::DEFAULT_LISTEN)]
        gateway: String,
    },
    /// Sanitized read-only JSON for a status publisher
    ///
    /// One compact line, keys sorted. Always exits 0: a down model is `unavailable`, and a load
    /// failure is `error_code: runtime_error` with no detail.
    Snapshot,
    /// Manage the model registry
    Model {
        #[command(subcommand)]
        cmd: ModelCmd,
    },
    /// Manage scenes (bundles of models)
    Scene {
        #[command(subcommand)]
        cmd: SceneCmd,
    },
    /// Run reproducible capability and serving-performance evaluations
    Eval {
        #[command(subcommand)]
        cmd: EvalCmd,
    },
    /// Inspect durable, privacy-safe model usage and concurrency history
    Usage {
        #[command(subcommand)]
        cmd: UsageCmd,
    },
    /// Measured memory per model vs its declared footprint, with a suggested footprint_gb
    Footprints {
        /// Window for peak / p99 (s, m, h, d, w)
        #[arg(long, default_value = "7d")]
        since: String,
        #[arg(long)]
        json: bool,
    },
    /// Run the memory watchdog in the foreground (kills the newest model before OOM-wedge)
    Watchdog {
        /// Kill the newest model if MemAvailable dips below this many GB
        #[arg(long, default_value_t = governor::WATCHDOG_FLOOR_GB)]
        floor: f64,
    },
    /// Run the stable streaming gateway in the foreground
    Gateway {
        #[arg(long, default_value = gateway::DEFAULT_LISTEN)]
        listen: String,
        /// tcp: the only upstream. http: where requests without a `model` go.
        #[arg(long, default_value = "127.0.0.1:8001")]
        upstream: String,
        #[command(flatten)]
        gateway: GatewayArgs,
    },
    /// Serve the Governor as a read-only JSON API over HTTP
    ///
    /// A second transport over the same operations the MCP server exposes, for a web UI. It
    /// never mutates: no activation, no stop, no eval run.
    Api {
        /// Loopback by default. This API has no authentication unless --token is set, and it
        /// reveals which models this node runs, so widening the bind is an explicit act.
        #[arg(long, default_value = "127.0.0.1:8012")]
        listen: String,
        /// Browser origin permitted to read this API. Repeatable. Empty means no browser may
        /// read it; `*` is deliberately not supported, because any page you visit would then
        /// be able to fingerprint this machine.
        #[arg(long = "allow-origin")]
        allow_origin: Vec<String>,
        /// Require `Authorization: Bearer <token>` on every request.
        #[arg(long)]
        token: Option<String>,
    },
    /// Serve the Governor to AI agents as MCP tools over stdio
    Mcp {
        /// Let an agent perform a real Scene transition, not just plan one.
        /// Off by default: the agent writes its own tool arguments, so the only
        /// meaningful gate on a production switch is one the operator sets here.
        #[arg(long)]
        allow_activate: bool,
    },
    /// Run the resident supervisor: resume desired Scene, gateway, and memory floor
    Agent {
        #[arg(long, default_value = gateway::DEFAULT_LISTEN)]
        listen: String,
        /// tcp: the only upstream. http: where requests without a `model` go.
        #[arg(long, default_value = "127.0.0.1:8001")]
        upstream: String,
        #[command(flatten)]
        gateway: GatewayArgs,
        #[arg(long, default_value_t = governor::WATCHDOG_FLOOR_GB)]
        floor: f64,
        /// Scrape only this Prometheus endpoint (legacy override). By default
        /// every running model with a known engine is scraped on its own port.
        #[arg(long)]
        metrics_url: Option<String>,
        /// Registry model id for metrics, or auto (all running models; with
        /// --metrics-url, the active Scene's model on that port).
        #[arg(long, default_value = "auto")]
        metrics_model: String,
        /// Sampling interval for time-weighted C0-C4 and queue history.
        #[arg(long, default_value_t = 5)]
        telemetry_interval_sec: u64,
        /// Per-scrape timeout in milliseconds (capped at 2000).
        #[arg(long, default_value_t = 2000)]
        telemetry_timeout_ms: u64,
        /// Disable resident metrics collection while keeping supervision and the gateway.
        #[arg(long)]
        no_telemetry: bool,
    },
    /// Read-only health checks of the box (Spark, Supabase, Tailscale, models) against lumbridge-computer.toml
    Doctor {
        /// lumbridge-computer.toml (default: $LUMBRIDGE_COMPUTER_CONFIG, then ./lumbridge-computer.toml)
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Diff lumbridge-computer.toml against the box (--dry-run), or apply one item (--only)
    Apply {
        /// lumbridge-computer.toml (default: $LUMBRIDGE_COMPUTER_CONFIG, then ./lumbridge-computer.toml)
        #[arg(long)]
        config: Option<PathBuf>,
        /// Print what would change; change nothing
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        json: bool,
        /// Apply (or with --dry-run, show) only this item
        #[arg(long, value_enum)]
        only: Option<box_::ApplyOnly>,
        /// Treat this directory as `/` (testing): files are written under it and
        /// systemd is not reloaded
        #[arg(long, hide = true)]
        host_root: Option<PathBuf>,
    },
}

/// Gateway flags shared by `gateway` and `agent`.
#[derive(clap::Args)]
struct GatewayArgs {
    /// `http` routes by the request's `model`; `tcp` is the pre-M1 byte pump to
    /// `--upstream`, kept for rollback.
    #[arg(long, value_enum, default_value_t = gateway::Mode::Http)]
    gateway_mode: gateway::Mode,
    /// Largest request body (MiB) the http gateway buffers to read `model`.
    #[arg(long, default_value_t = gateway::DEFAULT_MAX_BODY_MIB)]
    gateway_max_body_mib: usize,
    /// Seconds a request for a stopped model is held while it cold-starts (a
    /// 35B brain takes 4–6 min). A sleeping model is held for its own
    /// `serve.sleep.wake_timeout_sec` (default 60) instead.
    #[arg(long, default_value_t = gateway::DEFAULT_WAKE_TIMEOUT_SEC)]
    wake_timeout_sec: u64,
    /// Requests held per model while it wakes; the next one gets a 503.
    #[arg(long, default_value_t = gateway::DEFAULT_MAX_HELD_PER_MODEL)]
    max_held_per_model: u32,
}

impl GatewayArgs {
    fn options(&self, root: &Path, listen: &str, upstream: &str) -> gateway::Options {
        gateway::Options {
            mode: self.gateway_mode,
            listen: listen.to_string(),
            upstream: upstream.to_string(),
            root: root.to_path_buf(),
            max_body_bytes: self.gateway_max_body_mib.saturating_mul(1 << 20),
            wake_timeout: Duration::from_secs(self.wake_timeout_sec.max(1)),
            max_held_per_model: self.max_held_per_model,
        }
    }
}

#[derive(Subcommand)]
enum ModelCmd {
    /// List registered models and their live state
    Ls {
        #[arg(long)]
        json: bool,
    },
    /// Print one model fully resolved (extends + macros) and the argv it launches with
    Show {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// Drain then sleep a sleepable/on_demand model (stops it if its engine can't sleep)
    Sleep {
        id: String,
        #[command(flatten)]
        lifecycle: LifecycleArgs,
    },
    /// Wake a sleeping model and wait until it is ready
    Wake {
        id: String,
        #[command(flatten)]
        lifecycle: LifecycleArgs,
    },
    /// Start (or wake) a model of the desired Scene and wait until it is ready
    Start {
        id: String,
        #[command(flatten)]
        lifecycle: LifecycleArgs,
    },
    /// Drain then stop a sleepable/on_demand model
    Stop {
        id: String,
        #[command(flatten)]
        lifecycle: LifecycleArgs,
    },
}

/// Flags shared by `model sleep|wake|start|stop`.
#[derive(clap::Args)]
struct LifecycleArgs {
    /// Where the resident agent's gateway listens (in-flight counts for draining)
    #[arg(long, default_value = gateway::DEFAULT_LISTEN)]
    gateway: String,
    /// How long `wake`/`start` wait for the model to be ready
    #[arg(long, default_value_t = 900)]
    timeout_sec: u64,
}

#[derive(Subcommand)]
enum SceneCmd {
    /// List scenes with total footprint
    Ls {
        #[arg(long)]
        json: bool,
    },
    /// Show a scene: models, footprints, and the Governor's admission verdict
    Show {
        #[arg(default_value = config::DEFAULT_SCENE)]
        name: String,
        #[arg(long)]
        json: bool,
    },
    /// Activate a scene: stop what's not in it, admit + start what is
    Activate {
        #[arg(default_value = config::DEFAULT_SCENE)]
        name: String,
        /// Print the plan without changing anything
        #[arg(long)]
        dry_run: bool,
    },
    /// Adopt an already-running exact Scene and bind legacy PIDs to process identities
    Adopt { name: String },
    /// Resume the persisted desired Scene, falling back to the previous known-good Scene
    Resume,
    /// Stop all Lumbridge Computer-managed models
    Deactivate,
}

#[derive(Subcommand)]
enum EvalCmd {
    /// List available evaluation suites
    Ls,
    /// Run a suite against an OpenAI-compatible endpoint
    Run {
        suite: String,
        #[arg(long, default_value = "http://127.0.0.1:8001/v1")]
        base_url: String,
        #[arg(long, default_value = "brain")]
        model: String,
        /// Override the suite's repetitions per case
        #[arg(long)]
        repeat: Option<u32>,
    },
}

#[derive(Subcommand)]
enum UsageCmd {
    /// Scrape running models once and append durable snapshots
    Collect {
        /// Scrape only this endpoint instead of every running model.
        #[arg(long)]
        metrics_url: Option<String>,
        /// Registry model id, or auto for all.
        #[arg(long, default_value = "auto")]
        model: String,
        /// Per-scrape timeout in milliseconds (capped at 2000).
        #[arg(long, default_value_t = 2000)]
        timeout_ms: u64,
    },
    /// Report calls, tokens, vision, concurrency, queueing, and agent labels
    Summary {
        /// Window ending now, such as 24h, 7d, or 4w.
        #[arg(long, default_value = "24h")]
        since: String,
        #[arg(long)]
        json: bool,
    },
    /// Report bounded client/agent/workload usage labels
    Agents {
        #[arg(long, default_value = "7d")]
        since: String,
        #[arg(long)]
        json: bool,
    },
    /// Report the time-sampled C0-C4+ distribution and queue pressure
    Concurrency {
        #[arg(long, default_value = "7d")]
        since: String,
        #[arg(long)]
        json: bool,
    },
}

fn root_dir(cli: &Cli) -> PathBuf {
    cli.root
        .clone()
        .or_else(|| env_var("ROOT").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// `LUMBRIDGE_COMPUTER_<suffix>`; an empty value counts as unset.
pub(crate) fn env_var(suffix: &str) -> Option<String> {
    std::env::var(format!("LUMBRIDGE_COMPUTER_{suffix}"))
        .ok()
        .filter(|value| !value.is_empty())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = root_dir(&cli);
    match &cli.cmd {
        Cmd::Status {
            json: true,
            gateway,
        } => print_json(&mcp::collect_status_at(&root, gateway)?),
        Cmd::Status {
            json: false,
            gateway,
        } => cmd_status(&root, gateway),
        Cmd::Snapshot => {
            println!("{}", snapshot::collect(&root).to_line());
            Ok(())
        }
        Cmd::Model { cmd } => match cmd {
            ModelCmd::Ls { json: true } => print_json(&mcp::collect_models(&root)?),
            ModelCmd::Ls { json: false } => cmd_model_ls(&root),
            ModelCmd::Show { id, json } => config::show_model(&root, id, *json),
            ModelCmd::Sleep { id, lifecycle } => cmd_model_lifecycle(&root, id, "sleep", lifecycle),
            ModelCmd::Wake { id, lifecycle } => cmd_model_lifecycle(&root, id, "wake", lifecycle),
            ModelCmd::Start { id, lifecycle } => cmd_model_lifecycle(&root, id, "start", lifecycle),
            ModelCmd::Stop { id, lifecycle } => cmd_model_lifecycle(&root, id, "stop", lifecycle),
        },
        Cmd::Scene { cmd } => match cmd {
            SceneCmd::Ls { json: true } => print_json(&mcp::collect_scenes(&root)?),
            SceneCmd::Ls { json: false } => cmd_scene_ls(&root),
            SceneCmd::Show { name, json: true } => print_json(&mcp::collect_scene(&root, name)?),
            SceneCmd::Show { name, json: false } => cmd_scene_show(&root, name),
            SceneCmd::Activate { name, dry_run } => lifecycle::activate(&root, name, *dry_run),
            SceneCmd::Adopt { name } => lifecycle::adopt(&root, name),
            SceneCmd::Resume => lifecycle::resume(&root),
            SceneCmd::Deactivate => lifecycle::deactivate(&root),
        },
        Cmd::Eval { cmd } => match cmd {
            EvalCmd::Ls => eval::list(&root),
            EvalCmd::Run {
                suite,
                base_url,
                model,
                repeat,
            } => eval::run(&root, suite, base_url, model, *repeat),
        },
        Cmd::Usage { cmd } => cmd_usage(&root, cmd),
        Cmd::Footprints { since, json } => {
            let rows = footprint::report(&root, since)?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
            } else {
                print!("{}", footprint::render_text(&rows, since));
            }
            Ok(())
        }
        Cmd::Watchdog { floor } => {
            cmd_watchdog(&root, &governor::Pool::unified().with_floor(*floor))
        }
        Cmd::Gateway {
            listen,
            upstream,
            gateway,
        } => gateway::run(
            &gateway.options(&root, listen, upstream),
            gateway::InFlight::default(),
            // The states lane provides the Lifecycle; until then, pre-M3 routing.
            None,
        ),
        Cmd::Api {
            listen,
            allow_origin,
            token,
        } => http::run(
            listen,
            http::Config {
                root: root.clone(),
                allowed_origins: allow_origin.clone(),
                token: token.clone(),
            },
        ),
        Cmd::Mcp { allow_activate } => mcp::run(&root, *allow_activate),
        Cmd::Agent {
            listen,
            upstream,
            gateway,
            floor,
            metrics_url,
            metrics_model,
            telemetry_interval_sec,
            telemetry_timeout_ms,
            no_telemetry,
        } => agent::run(
            &root,
            gateway.options(&root, listen, upstream),
            governor::Pool::unified().with_floor(*floor),
            if *no_telemetry {
                None
            } else {
                Some(agent::TelemetryConfig {
                    metrics_url: metrics_url.clone(),
                    model: metrics_model.clone(),
                    interval: Duration::from_secs((*telemetry_interval_sec).max(1)),
                    timeout: Duration::from_millis(*telemetry_timeout_ms),
                })
            },
        ),
        Cmd::Doctor { config, json } => std::process::exit(box_::doctor(config.as_deref(), *json)?),
        Cmd::Apply {
            config,
            dry_run,
            json,
            only,
            host_root,
        } => std::process::exit(box_::apply(
            config.as_deref(),
            *dry_run,
            *json,
            *only,
            host_root.as_deref(),
        )?),
    }
}

/// `--json` prints exactly what the MCP tool of the same name returns: one collector,
/// two transports, so a script and an agent can never disagree about a number.
fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn cmd_usage(root: &Path, command: &UsageCmd) -> Result<()> {
    match command {
        UsageCmd::Collect {
            metrics_url,
            model,
            timeout_ms,
        } => {
            let result = telemetry::collect_once(
                root,
                metrics_url.as_deref(),
                model,
                Duration::from_millis(*timeout_ms),
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        UsageCmd::Summary { since, json } => {
            let report = telemetry::report(root, since)?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("{}", telemetry::render_text(&report));
            }
        }
        UsageCmd::Agents { since, json } => {
            let report = telemetry::report(root, since)?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&report.agents)?);
            } else if report.agents.is_empty() {
                println!("no agent-labelled usage in this window");
            } else {
                for row in report.agents {
                    println!(
                        "{}/{}/{}  {:.0} requests · {:.0} prompt · {:.0} generated tokens",
                        row.client,
                        row.agent,
                        row.workload,
                        row.requests,
                        row.prompt_tokens,
                        row.generation_tokens
                    );
                }
            }
        }
        UsageCmd::Concurrency { since, json } => {
            let report = telemetry::report(root, since)?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&report.concurrency)?);
            } else if report.concurrency.samples == 0 {
                println!("no concurrency samples in this window");
            } else {
                for bucket in report.concurrency.distribution {
                    println!(
                        "{}  {} samples · {:.1}%",
                        bucket.concurrency,
                        bucket.samples,
                        bucket.share * 100.0
                    );
                }
                println!(
                    "peak running {} · peak queued {} · queue observed in {} sample(s)",
                    report.concurrency.peak_running,
                    report.concurrency.peak_queued,
                    report.concurrency.queued_samples
                );
            }
        }
    }
    Ok(())
}

fn cmd_status(root: &Path, gateway_addr: &str) -> Result<()> {
    let reg = Registry::load(root)?;
    let pool = governor::Pool::unified();
    let m = pool.sense()?;
    let running = governor::running_ids(&reg);
    let reserved = pool.reserved_gb(&reg);
    let reserved_unused = pool.reserved_unused_gb(&reg);
    let managed = State::load_checked(root)?;
    // Sleeping models count at their sleeping footprint (M3).
    let committed = states::committed_gb(root, &reg, &managed, &pool);
    let configured_budget = governor::scene_budget_gb(
        &pool,
        root,
        managed
            .active_scene
            .as_deref()
            .or(managed.desired_scene.as_deref()),
    )?;
    let budget = pool.effective_budget_gb(configured_budget, Some(m.total_gb), reserved);
    let model_available = m.available_gb - reserved_unused;
    // The binding constraint, not the generous one: admission enforces the declared
    // budget AND observed memory, so reporting only the declared headroom would promise
    // room the next `scene activate` is going to refuse.
    let headroom = pool
        .headroom_gb(committed, budget, Some(model_available))
        .max(0.0);
    let declared_headroom = pool.declared_headroom_gb(committed, budget).max(0.0);

    println!("Lumbridge Computer · governor");
    println!(
        "  memory     {:.1} GB total · {:.1} GB available",
        m.total_gb, m.available_gb
    );
    if let Some(pressure) = pressure::read() {
        println!(
            "  pressure   {:.1}% of the last 10 s stalled on memory (all tasks {:.1}%){}",
            pressure.some_avg10,
            pressure.full_avg10,
            if pressure.elevated() {
                "  ← elevated"
            } else {
                ""
            }
        );
    }
    for (id, r) in &reg.reservations {
        println!(
            "  reserved   {:.1} GB  {id}{}",
            r.gb,
            if r.description.is_empty() {
                String::new()
            } else {
                format!(" — {}", r.description)
            }
        );
    }
    if reserved > 0.0 {
        println!(
            "             {:.1} GB of it in use, {reserved_unused:.1} GB held back from models",
            reserved - reserved_unused
        );
    }
    println!(
        "  budget     {:.0} GB  (safety margin {:.0} · watchdog floor {:.0})",
        budget, pool.safety_margin_gb, pool.floor_gb
    );
    println!(
        "  committed  {committed:.1} GB across {} model(s)",
        running.len()
    );
    if headroom < declared_headroom {
        println!(
            "  headroom   {headroom:.1} GB admittable  (budget allows {declared_headroom:.1}; \
             MemAvailable is the tighter limit)"
        );
    } else {
        println!("  headroom   {headroom:.1} GB admittable");
    }
    println!();
    println!(
        "  scene      active={} desired={} fallback={}",
        managed.active_scene.as_deref().unwrap_or("-"),
        managed.desired_scene.as_deref().unwrap_or("-"),
        managed.last_known_good_scene.as_deref().unwrap_or("-")
    );
    if let Some(error) = &managed.last_error {
        println!("  last error {error}");
    }
    match mcp::gateway_status(gateway_addr, &running) {
        Some(report) => println!(
            "  in flight  {}  (gateway {gateway_addr})",
            report.summary()
        ),
        None => println!("  in flight  - (no agent gateway at {gateway_addr})"),
    }

    let model_states = mcp::collect_model_states(
        root,
        &reg,
        &managed,
        states::read_gateway_counts(gateway_addr).as_ref(),
    );
    if model_states.iter().any(|row| {
        row.mode.as_deref().is_some_and(|mode| mode != "resident") || row.state != "ready"
    }) {
        println!("  states:");
        for row in &model_states {
            let state = match row.sleep_level {
                Some(level) => format!("sleeping/{level}"),
                None => row.state.clone(),
            };
            let idle = row
                .idle_sec
                .map_or(String::new(), |sec| format!("  idle {sec}s"));
            let asleep = row
                .sleeping_footprint_gb
                .map_or(String::new(), |gb| format!("  asleep {gb:.1} GB"));
            println!(
                "    {:<18} {:<10} {:<11} counts {:>5.1} GB{idle}{asleep}",
                row.id,
                row.mode.as_deref().unwrap_or("-"),
                state,
                row.counted_gb
            );
        }
    }

    if running.is_empty() {
        println!("  (no registered models currently serving)");
    } else {
        // Measured next to declared (host + GPU + swap); one nvidia-smi call.
        let measured = footprint::measure_owned(
            &footprint::Roots::default(),
            &managed,
            &footprint::NvidiaSmi,
        );
        println!("  running:");
        for id in &running {
            let mdl = &reg.models[id];
            let tag = if managed
                .procs
                .get(id)
                .is_some_and(|process| process.owned_alive())
            {
                "owned"
            } else {
                "ext  "
            };
            let real = measured.get(id).map_or(String::new(), |m| {
                format!(
                    "  measured {:.1} GB{} [{}]",
                    m.demand_gb(),
                    if m.swap_gb >= 0.05 {
                        format!(" ({:.1} swap)", m.swap_gb)
                    } else {
                        String::new()
                    },
                    m.source
                )
            });
            println!(
                "    ● [{tag}] {:<12} {:>5.0} GB  :{:<5} {}{real}",
                id,
                mdl.footprint_gb,
                port_str(mdl.serve.port),
                mdl.name
            );
        }
    }
    Ok(())
}

/// `lumbridge-computer model sleep|wake|start|stop <id>`: each goes through the same
/// `Lifecycle` the agent uses, so the CLI and the agent serialise on the
/// transition lock and agree on every state.
fn cmd_model_lifecycle(root: &Path, id: &str, verb: &str, args: &LifecycleArgs) -> Result<()> {
    use states::Lifecycle;
    let lifecycle = states::ModelLifecycle::new(
        root,
        std::sync::Arc::new(states::CliHost::new(&args.gateway)),
    );
    let deadline = std::time::Instant::now() + Duration::from_secs(args.timeout_sec.max(1));
    let before = lifecycle.state(id);
    match verb {
        "sleep" => lifecycle.sleep(id, "operator: lumbridge-computer model sleep")?,
        "stop" => lifecycle.stop(id, "operator: lumbridge-computer model stop")?,
        "wake" => {
            if before == states::ModelState::Stopped {
                anyhow::bail!(
                    "'{id}' is stopped, not asleep; use `lumbridge-computer model start {id}`"
                );
            }
            lifecycle.ensure_ready(id, deadline)?
        }
        _ => lifecycle.ensure_ready(id, deadline)?,
    }
    println!("{id}: {before} -> {}", lifecycle.state(id));
    Ok(())
}

fn cmd_model_ls(root: &Path) -> Result<()> {
    let reg = Registry::load(root)?;
    println!(
        "{:<12} {:>6}  {:<6} {:<6} MODEL",
        "ID", "GB", "STATE", "PORT"
    );
    for (id, m) in &reg.models {
        println!(
            "{:<12} {:>6.0}  {:<6} {:<6} {}",
            id,
            m.footprint_gb,
            if governor::is_running(m) {
                "up"
            } else {
                "down"
            },
            port_str(m.serve.port),
            m.name
        );
    }
    Ok(())
}

fn cmd_scene_ls(root: &Path) -> Result<()> {
    let reg = Registry::load(root)?;
    let scenes = load_scenes(root)?;
    if scenes.is_empty() {
        println!("no scenes in {}/scenes", root.display());
        return Ok(());
    }
    println!("{:<12} {:>6}  {:<30} MODELS", "SCENE", "GB", "DESCRIPTION");
    for s in &scenes {
        let total = scene_footprint(s, &reg);
        println!(
            "{:<12} {:>6.0}  {:<30} {}",
            s.metadata.name,
            total,
            truncate(&s.metadata.description, 30),
            s.models.join(", ")
        );
    }
    Ok(())
}

fn cmd_scene_show(root: &Path, name: &str) -> Result<()> {
    let reg = Registry::load(root)?;
    let scene = find_scene(root, name)?;
    let pool = governor::Pool::unified();
    let reserved = pool.reserved_gb(&reg);
    let budget = pool.effective_budget_gb(
        pool.scene_budget_gb(scene.budget_gb),
        pool.sense().ok().map(|m| m.total_gb),
        reserved,
    );

    println!(
        "scene {} (v{})",
        scene.metadata.name, scene.metadata.version
    );
    if !scene.metadata.description.is_empty() {
        println!("  {}", scene.metadata.description);
    }
    if reserved > 0.0 {
        println!("  budget {budget:.0} GB  (after {reserved:.1} GB reserved)\n  models:");
    } else {
        println!("  budget {budget:.0} GB\n  models:");
    }

    let mut total = 0.0;
    let mut missing = Vec::new();
    for id in &scene.models {
        match reg.models.get(id) {
            Some(m) => {
                total += m.footprint_gb;
                let state = if governor::is_running(m) {
                    "up"
                } else {
                    "down"
                };
                let mode = scene.mode_of(id);
                let knobs = match mode.mode {
                    config::Mode::Resident => String::new(),
                    config::Mode::Sleepable => format!(
                        "  idle {}",
                        mode.idle_sec()
                            .map_or("never".to_string(), |s| format!("{s}s"))
                    ),
                    config::Mode::OnDemand => format!(
                        "  idle {} · stop after {}s",
                        mode.idle_sec()
                            .map_or("never".to_string(), |s| format!("{s}s")),
                        mode.stop_after_sec()
                    ),
                };
                println!(
                    "    {:<12} {:>5.0} GB  [{:<4}] {:<9} {}{knobs}",
                    id,
                    m.footprint_gb,
                    state,
                    mode.mode.as_str(),
                    m.name
                );
            }
            None => {
                missing.push(id.clone());
                println!("    {id:<12}     ?     [MISSING from registry]");
            }
        }
    }

    let need = states::scene_need_gb(&reg, &scene, &pool);
    let needed = need + pool.safety_margin_gb;
    println!(
        "\n  total footprint  {total:.1} GB  (+{:.0} safety = {:.1} GB)",
        pool.safety_margin_gb,
        total + pool.safety_margin_gb
    );
    if !scene.modes.is_empty() {
        println!(
            "  with modes       {need:.1} GB  (+{:.0} safety = {needed:.1} GB): residents in full, \
             the rest asleep, room for one awake",
            pool.safety_margin_gb
        );
    }
    let verdict = if !missing.is_empty() {
        format!("✗ {} model(s) missing from registry", missing.len())
    } else if needed <= budget {
        format!("✓ fits — {:.1} GB to spare", budget - needed)
    } else {
        format!("✗ exceeds budget by {:.1} GB", needed - budget)
    };
    println!("  admission        {verdict}");
    Ok(())
}

fn cmd_watchdog(root: &Path, pool: &governor::Pool) -> Result<()> {
    let floor = pool.floor_gb;
    println!(
        "Lumbridge Computer watchdog — killing the newest model if MemAvailable < {floor:.1} GB. Ctrl-C to stop."
    );
    loop {
        let mut m = pool.sense()?;
        let reg = Registry::load(root).ok();
        if let Some(reg) = reg.as_ref() {
            m.available_gb = pool.model_available_gb(reg, m.available_gb);
        }
        if pool.below_floor(m.available_gb) {
            let mut state = State::load_checked(root)?;
            let overage = reg.as_ref().map_or_else(Default::default, |reg| {
                let measured = footprint::measure_owned(
                    &footprint::Roots::default(),
                    &state,
                    &footprint::NvidiaSmi,
                );
                footprint::overage_gb(reg, &measured)
            });
            match state.shed_candidate_by(reg.as_ref(), &overage) {
                Some((id, p)) => {
                    eprintln!(
                        "watchdog: memory available to models {:.1} GB < floor {:.1} — killing newest '{id}' (pid {})",
                        m.available_gb, floor, p.pid
                    );
                    proc::shed_owned(&p)?;
                    state.procs.remove(&id);
                    state.save(root)?;
                }
                None => eprintln!(
                    "watchdog: memory available to models {:.1} GB < floor {:.1} but no Lumbridge Computer-managed model to kill!",
                    m.available_gb, floor
                ),
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

// ---- helpers ---------------------------------------------------------------

fn scene_footprint(s: &Scene, reg: &Registry) -> f64 {
    s.models
        .iter()
        .filter_map(|id| reg.models.get(id))
        .map(|m| m.footprint_gb)
        .sum()
}

fn port_str(p: Option<u16>) -> String {
    p.map(|p| p.to_string()).unwrap_or_else(|| "-".to_string())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!(
            "{}…",
            s.chars().take(n.saturating_sub(1)).collect::<String>()
        )
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn default_scene_is_used_without_losing_dry_run_or_explicit_names() {
        let parsed = Cli::try_parse_from(["lumbridge-computer", "scene", "show"]).unwrap();
        assert!(
            matches!(parsed.cmd, Cmd::Scene { cmd: SceneCmd::Show { name, json: false } }
            if name == config::DEFAULT_SCENE)
        );
        let parsed =
            Cli::try_parse_from(["lumbridge-computer", "scene", "activate", "--dry-run"]).unwrap();
        assert!(matches!(parsed.cmd, Cmd::Scene {
            cmd: SceneCmd::Activate { name, dry_run: true }
        } if name == config::DEFAULT_SCENE));
        let parsed = Cli::try_parse_from([
            "lumbridge-computer",
            "scene",
            "activate",
            "custom",
            "--dry-run",
        ])
        .unwrap();
        assert!(matches!(parsed.cmd, Cmd::Scene {
            cmd: SceneCmd::Activate { name, dry_run: true }
        } if name == "custom"));
    }

    #[test]
    fn model_lifecycle_verbs_parse() {
        for verb in ["sleep", "wake", "start", "stop"] {
            let parsed =
                Cli::try_parse_from(["lumbridge-computer", "model", verb, "brain-alt"]).unwrap();
            let Cmd::Model { cmd } = parsed.cmd else {
                panic!("{verb}")
            };
            let (id, args) = match cmd {
                ModelCmd::Sleep { id, lifecycle }
                | ModelCmd::Wake { id, lifecycle }
                | ModelCmd::Start { id, lifecycle }
                | ModelCmd::Stop { id, lifecycle } => (id, lifecycle),
                _ => panic!("{verb}"),
            };
            assert_eq!(id, "brain-alt");
            assert_eq!(args.gateway, gateway::DEFAULT_LISTEN);
            assert_eq!(args.timeout_sec, 900);
        }
        assert!(Cli::try_parse_from(["lumbridge-computer", "model", "wake"]).is_err());
    }

    #[test]
    fn json_flags_parse_on_every_read_command() {
        for argv in [
            &["lumbridge-computer", "status", "--json"][..],
            &["lumbridge-computer", "model", "ls", "--json"],
            &["lumbridge-computer", "scene", "ls", "--json"],
            &["lumbridge-computer", "scene", "show", "custom", "--json"],
        ] {
            assert!(Cli::try_parse_from(argv).is_ok(), "{argv:?}");
        }
        let parsed = Cli::try_parse_from([
            "lumbridge-computer",
            "status",
            "--json",
            "--gateway",
            "127.0.0.1:1",
        ])
        .unwrap();
        assert!(
            matches!(parsed.cmd, Cmd::Status { json: true, gateway } if gateway == "127.0.0.1:1")
        );
        let parsed =
            Cli::try_parse_from(["lumbridge-computer", "scene", "show", "--json"]).unwrap();
        assert!(matches!(parsed.cmd, Cmd::Scene {
            cmd: SceneCmd::Show { name, json: true }
        } if name == config::DEFAULT_SCENE));
    }
}
