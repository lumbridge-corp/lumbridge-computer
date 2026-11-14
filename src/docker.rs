//! Models that run as `docker run` containers (M4).
//!
//! A registry entry like `brain-dock` launches
//! `docker run --rm --init --name=brain-dock … image …`. Lumbridge Computer spawns and
//! records the **docker CLI client**, but the model itself runs under
//! containerd in `system.slice/docker-<cid>.scope`, with its own pids. Tracking
//! only the CLI meant: footprints of ~0 GB (the CLI holds nothing), a stop that
//! SIGKILLed the CLI and left the container (and its port, memory and
//! `--name`) behind, and liveness that followed the CLI, not the engine.
//!
//! This module gives Lumbridge Computer the container:
//!
//! - **start**: `--cidfile <root>/.compute/cid/<id>.cid` is injected right
//!   after `run`; a stale cidfile's container and any container already holding
//!   the same `--name` are removed first, so a start never hits a name conflict.
//!   The id read back from the cidfile is stored as `Proc::container_id`.
//! - **measure**: the container's cgroup (from its main pid's
//!   `/proc/<pid>/cgroup`, else the systemd driver's `system.slice/docker-<cid>.scope`)
//!   and every nvidia-smi compute pid inside it (see `footprint`).
//! - **stop**: `docker stop -t <grace>` before the CLI's process group is
//!   signalled, then `docker rm -f` if the container is still there.
//! - **legacy records** (state written before M4, or a start whose cidfile
//!   never appeared, e.g. a long image pull) are resolved by the `--name` in
//!   the CLI's own `/proc/<pid>/cmdline`, so a deploy adopts running containers
//!   without bouncing them.
//!
//! Every docker call goes through the `Docker` trait, so unit tests never need
//! a daemon.

use anyhow::{bail, Context, Result};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// `docker stop` grace when the caller has no better number: the same 30 s a
/// drain gets (`config::DEFAULT_DRAIN_TIMEOUT_SEC`).
pub const DEFAULT_STOP_GRACE: Duration =
    Duration::from_secs(crate::config::DEFAULT_DRAIN_TIMEOUT_SEC);
/// How long a start waits for the CLI to write the cidfile. Past this (an image
/// pull, say) the record has no id and is resolved by name, like a legacy one.
pub const CIDFILE_WAIT: Duration = Duration::from_secs(30);
/// Bound on every short docker call (`ps`, `inspect`, `rm`): under memory
/// pressure a wedged daemon must not stall the watchdog.
const CALL_TIMEOUT: Duration = Duration::from_secs(10);

/// The docker CLI, as a seam: tests script it.
pub trait Docker: Sync {
    /// Run `docker <args>` with a deadline; stdout on success, an error
    /// carrying stderr on a non-zero exit.
    fn run(&self, args: &[&str], timeout: Duration) -> Result<String>;
}

/// The real `docker` on `PATH`.
#[cfg_attr(test, allow(dead_code))]
pub struct DockerCli;

impl Docker for DockerCli {
    fn run(&self, args: &[&str], timeout: Duration) -> Result<String> {
        let mut child = Command::new("docker")
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("running docker")?;
        let deadline = Instant::now() + timeout;
        let status = loop {
            if let Some(status) = child.try_wait().context("waiting for docker")? {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                bail!(
                    "docker {} timed out after {}s",
                    args.join(" "),
                    timeout.as_secs()
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let mut out = String::new();
        let mut err = String::new();
        if let Some(mut stdout) = child.stdout.take() {
            stdout.read_to_string(&mut out).ok();
        }
        if let Some(mut stderr) = child.stderr.take() {
            stderr.read_to_string(&mut err).ok();
        }
        if !status.success() {
            bail!("docker {}: {}", args.join(" "), err.trim());
        }
        Ok(out)
    }
}

/// The docker the agent uses. Unit tests get one that always fails, so no test
/// can reach a daemon by accident; tests that exercise docker pass a fake.
#[cfg(not(test))]
pub fn system() -> &'static dyn Docker {
    &DockerCli
}

#[cfg(test)]
pub fn system() -> &'static dyn Docker {
    struct NoDocker;
    impl Docker for NoDocker {
        fn run(&self, args: &[&str], _: Duration) -> Result<String> {
            bail!("docker {} is not available in unit tests", args.join(" "))
        }
    }
    &NoDocker
}

// ---- argv ---------------------------------------------------------------------

/// docker's global options that take a value (`docker --context x run …`).
const GLOBAL_WITH_VALUE: &[&str] = &[
    "-c",
    "--context",
    "--config",
    "-H",
    "--host",
    "-l",
    "--log-level",
    "--tlscacert",
    "--tlscert",
    "--tlskey",
];

/// `docker run` flags that take no value. Anything else that starts with `-`
/// and has no `=` is assumed to take the next token, which is the safe guess
/// for finding the image: a wrong guess only scans further, never shorter.
const RUN_BOOL_FLAGS: &[&str] = &[
    "--rm",
    "--init",
    "-d",
    "--detach",
    "-i",
    "--interactive",
    "-t",
    "--tty",
    "--privileged",
    "--read-only",
    "-P",
    "--publish-all",
    "--oom-kill-disable",
    "--no-healthcheck",
    "--sig-proxy",
    "-q",
    "--quiet",
    "--disable-content-trust",
    "--use-api-socket",
    "--help",
];

/// Index of the `run` subcommand when `argv` is `docker [global opts] run …`
/// (or `docker container run …`); `None` for anything else.
pub fn run_index(argv: &[String]) -> Option<usize> {
    let program = Path::new(argv.first()?).file_name()?;
    if program != "docker" {
        return None;
    }
    let mut i = 1;
    while let Some(arg) = argv.get(i) {
        if arg.starts_with('-') {
            i += if GLOBAL_WITH_VALUE.contains(&arg.as_str()) {
                2
            } else {
                1
            };
            continue;
        }
        return match arg.as_str() {
            "run" => Some(i),
            "container" if argv.get(i + 1).map(String::as_str) == Some("run") => Some(i + 1),
            _ => None,
        };
    }
    None
}

/// True for a model whose launch argv is a `docker run`.
pub fn launches_container(argv: &[String]) -> bool {
    run_index(argv).is_some()
}

/// The `docker run` options between `run` and the image, as `(index, flag,
/// inline value, next-token value)` per option.
fn run_options(argv: &[String]) -> Vec<(usize, &str, Option<&str>, Option<&str>)> {
    let Some(run) = run_index(argv) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut i = run + 1;
    while let Some(arg) = argv.get(i) {
        if !arg.starts_with('-') || arg == "-" {
            break; // the image
        }
        if let Some((flag, value)) = arg.split_once('=') {
            out.push((i, flag, Some(value), None));
            i += 1;
        } else if is_bool_flag(arg) {
            out.push((i, arg.as_str(), None, None));
            i += 1;
        } else {
            out.push((i, arg.as_str(), None, argv.get(i + 1).map(String::as_str)));
            i += 2;
        }
    }
    out
}

fn is_bool_flag(arg: &str) -> bool {
    if RUN_BOOL_FLAGS.contains(&arg) {
        return true;
    }
    // Combined short booleans: `-it`, `-dit`.
    arg.len() > 2
        && !arg.starts_with("--")
        && arg[1..]
            .chars()
            .all(|c| matches!(c, 'd' | 'i' | 't' | 'P' | 'q'))
}

/// The container's `--name`, if the argv sets one.
pub fn container_name(argv: &[String]) -> Option<String> {
    run_options(argv)
        .into_iter()
        .filter(|(_, flag, _, _)| *flag == "--name")
        .filter_map(|(_, _, inline, next)| inline.or(next))
        .next_back()
        .map(str::to_string)
}

/// `argv` with exactly one `--cidfile <path>`, right after `run`. Any
/// `--cidfile` already among the run options (a registry's own, or a previous
/// injection) is dropped: docker would keep only the last, and the agent must
/// know where its file is. Idempotent; a non-docker argv comes back unchanged.
pub fn inject_cidfile(argv: &[String], cidfile: &Path) -> Vec<String> {
    let Some(run) = run_index(argv) else {
        return argv.to_vec();
    };
    let mut drop = Vec::new();
    for (i, flag, inline, _) in run_options(argv) {
        if flag == "--cidfile" {
            drop.push(i);
            if inline.is_none() {
                drop.push(i + 1);
            }
        }
    }
    let mut out: Vec<String> = argv
        .iter()
        .enumerate()
        .filter(|(i, _)| !drop.contains(i))
        .map(|(_, a)| a.clone())
        .collect();
    out.insert(run + 1, "--cidfile".to_string());
    out.insert(run + 2, cidfile.display().to_string());
    out
}

/// Where the CLI writes `id`'s container id.
pub fn cidfile_path(root: &Path, id: &str) -> PathBuf {
    root.join(".compute").join("cid").join(format!("{id}.cid"))
}

/// A full container id: 64 lowercase hex characters.
fn valid_id(text: &str) -> Option<String> {
    let id = text.trim();
    (id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit())).then(|| id.to_string())
}

/// A `name` filter that matches the name exactly. Docker treats it as a
/// regex; names may contain `.`.
fn name_filter(name: &str) -> String {
    format!("name=^{}$", name.replace('.', "\\."))
}

/// What a start must clear before `docker run` (the caller injects the
/// cidfile with `inject_cidfile`).
///
/// Removes the container a stale cidfile names and any container (running or
/// not) already holding the same `--name` (a container orphaned by a CLI that
/// was killed, or by a stop from before M4), logging each to `log`. Docker
/// errors are logged, not fatal: if the name is really taken, `docker run`
/// says so in the model's log.
pub fn clear_for_start(
    docker: &dyn Docker,
    root: &Path,
    id: &str,
    argv: &[String],
    log: &mut dyn Write,
) -> Result<()> {
    let cidfile = cidfile_path(root, id);
    if let Some(dir) = cidfile.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    if let Ok(text) = fs::read_to_string(&cidfile) {
        if let Some(stale) = valid_id(&text) {
            if matches!(status(docker, &stale), Ok(Some(_))) {
                writeln!(
                    log,
                    "lumbridge-computer: removing container {stale} left by the previous run"
                )?;
                if let Err(error) = remove(docker, &stale) {
                    writeln!(log, "lumbridge-computer: {error:#}")?;
                }
            }
        }
    }
    // docker refuses to start when the cidfile already exists.
    match fs::remove_file(&cidfile) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("removing {}", cidfile.display())),
    }
    if let Some(name) = container_name(argv) {
        let filter = name_filter(&name);
        match docker.run(&["ps", "-aq", "--no-trunc", "-f", &filter], CALL_TIMEOUT) {
            Ok(out) => {
                for existing in out.lines().filter_map(valid_id) {
                    writeln!(
                        log,
                        "lumbridge-computer: removing existing container {existing} named '{name}'"
                    )?;
                    if let Err(error) = remove(docker, &existing) {
                        writeln!(log, "lumbridge-computer: {error:#}")?;
                    }
                }
            }
            Err(error) => writeln!(
                log,
                "lumbridge-computer: could not list containers named '{name}': {error:#}"
            )?,
        }
    }
    Ok(())
}

/// Poll `cidfile` until it holds a container id, `alive` turns false, or
/// `timeout` passes.
pub fn wait_cidfile(
    cidfile: &Path,
    alive: &dyn Fn() -> bool,
    timeout: Duration,
    poll: Duration,
) -> Option<String> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(id) = fs::read_to_string(cidfile)
            .ok()
            .as_deref()
            .and_then(valid_id)
        {
            return Some(id);
        }
        if !alive() || Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(poll);
    }
}

// ---- one container --------------------------------------------------------------

/// How a record names its container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Handle {
    /// From the cidfile: exact.
    Id(String),
    /// From the CLI's `--name` (legacy record, or no cidfile yet).
    Name(String),
}

/// The CLI's argv from `/proc/<pid>/cmdline`.
pub fn cmdline(proc_root: &Path, pid: i32) -> Option<Vec<String>> {
    let raw = fs::read(proc_root.join(pid.to_string()).join("cmdline")).ok()?;
    let argv: Vec<String> = raw
        .split(|b| *b == 0)
        .filter(|part| !part.is_empty())
        .map(|part| String::from_utf8_lossy(part).into_owned())
        .collect();
    (!argv.is_empty()).then_some(argv)
}

/// The container a record stands for: its stored id, else the `--name` of the
/// `docker run` its pid is running. `None` for a non-docker process.
pub fn handle(proc_root: &Path, container_id: Option<&str>, pid: i32) -> Option<Handle> {
    if let Some(id) = container_id {
        return Some(Handle::Id(id.to_string()));
    }
    let argv = cmdline(proc_root, pid)?;
    launches_container(&argv).then_some(())?;
    container_name(&argv).map(Handle::Name)
}

/// The full id behind a handle. `Ok(None)` when a name matches no container.
pub fn resolve(docker: &dyn Docker, handle: &Handle) -> Result<Option<String>> {
    match handle {
        Handle::Id(id) => Ok(Some(id.clone())),
        Handle::Name(name) => {
            let filter = name_filter(name);
            let out = docker.run(&["ps", "-aq", "--no-trunc", "-f", &filter], CALL_TIMEOUT)?;
            Ok(out.lines().find_map(valid_id))
        }
    }
}

fn no_such(error: &anyhow::Error) -> bool {
    format!("{error:#}")
        .to_ascii_lowercase()
        .contains("no such")
}

/// `State.Status` (`created`, `running`, `exited`, …), or `Ok(None)` when the
/// container does not exist.
pub fn status(docker: &dyn Docker, id: &str) -> Result<Option<String>> {
    match docker.run(&["inspect", "-f", "{{.State.Status}}", id], CALL_TIMEOUT) {
        Ok(out) => Ok(Some(out.trim().to_string())),
        Err(error) if no_such(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

/// True only on positive evidence the container is finished: it is gone, or
/// docker reports it exited, dead or being removed. `created` counts as alive
/// (the cidfile is written before the container starts).
pub fn finished(status: Option<&str>) -> bool {
    matches!(status, None | Some("exited" | "dead" | "removing"))
}

/// The cgroup v2 path (relative to the cgroup root) the container runs in.
///
/// The main pid's `/proc/<pid>/cgroup` works for either cgroup driver; the
/// systemd driver's `system.slice/docker-<id>.scope` is the fallback when the
/// pid cannot be read.
pub fn cgroup(docker: &dyn Docker, roots: &crate::footprint::Roots, id: &str) -> Option<String> {
    let pid = docker
        .run(&["inspect", "-f", "{{.State.Pid}}", id], CALL_TIMEOUT)
        .ok()
        .and_then(|out| out.trim().parse::<i32>().ok())
        .filter(|pid| *pid > 0);
    if let Some(path) = pid.and_then(|pid| crate::footprint::cgroup_of(roots, pid)) {
        if !path.is_empty() {
            return Some(path);
        }
    }
    let systemd = format!("system.slice/docker-{id}.scope");
    roots.cgroup.join(&systemd).is_dir().then_some(systemd)
}

/// `docker stop -t <grace>`: SIGTERM to the container's init, SIGKILL after
/// `grace`. A container that is already gone is not an error.
pub fn stop(docker: &dyn Docker, id: &str, grace: Duration) -> Result<()> {
    let secs = grace.as_secs().to_string();
    match docker.run(
        &["stop", "-t", &secs, id],
        grace + CALL_TIMEOUT + Duration::from_secs(5),
    ) {
        Ok(_) => Ok(()),
        Err(error) if no_such(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

/// `docker rm -f`. Gone already, or `--rm` already removing it, is success.
pub fn remove(docker: &dyn Docker, id: &str) -> Result<()> {
    match docker.run(&["rm", "-f", id], CALL_TIMEOUT) {
        Ok(_) => Ok(()),
        Err(error) if no_such(&error) => Ok(()),
        Err(error) if format!("{error:#}").contains("already in progress") => Ok(()),
        Err(error) => Err(error).with_context(|| format!("removing container {id}")),
    }
}

#[cfg(test)]
pub mod fake {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A scripted docker: answers by the joined argv's first words, records
    /// every call. Unscripted calls fail with "no such object".
    #[derive(Default)]
    pub struct FakeDocker {
        pub answers: Mutex<HashMap<String, Vec<Result<String, String>>>>,
        pub calls: Mutex<Vec<String>>,
    }

    impl FakeDocker {
        /// Queue `answer` for calls starting with `prefix`. The last queued
        /// answer for a prefix repeats.
        pub fn on(&self, prefix: &str, answer: Result<&str, &str>) -> &Self {
            self.answers
                .lock()
                .unwrap()
                .entry(prefix.to_string())
                .or_default()
                .push(answer.map(str::to_string).map_err(str::to_string));
            self
        }

        pub fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl Docker for FakeDocker {
        fn run(&self, args: &[&str], _: Duration) -> Result<String> {
            let line = args.join(" ");
            self.calls.lock().unwrap().push(line.clone());
            let mut answers = self.answers.lock().unwrap();
            let key = answers
                .keys()
                .filter(|prefix| line.starts_with(prefix.as_str()))
                .max_by_key(|prefix| prefix.len())
                .cloned();
            let Some(queue) = key.and_then(|key| answers.get_mut(&key)) else {
                bail!("Error: No such object: {line}");
            };
            let answer = if queue.len() > 1 {
                queue.remove(0)
            } else {
                queue[0].clone()
            };
            answer.map_err(|e| anyhow::anyhow!(e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::FakeDocker;
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    const CID: &str = "31c245bd349d50351b56efd802d73d2eba1f05263f0b36a6a6c055af8d1da986";
    const OLD: &str = "1be159322466b720589f5a21d0f2bc6ae84d437dd49139fc27efc7d7e8546284";

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lumbridge-computer-docker-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn only_a_docker_run_argv_is_a_container_launch() {
        assert_eq!(run_index(&argv("docker run --rm img")), Some(1));
        assert_eq!(run_index(&argv("/usr/bin/docker run img")), Some(1));
        assert_eq!(run_index(&argv("docker --context spark run img")), Some(3));
        assert_eq!(run_index(&argv("docker -D container run img")), Some(3));
        assert_eq!(run_index(&argv("docker compose up")), None);
        assert_eq!(run_index(&argv("docker ps")), None);
        assert_eq!(run_index(&argv("vllm serve run")), None);
        assert_eq!(run_index(&argv("dockerd run")), None);
        assert_eq!(run_index(&[]), None);
    }

    #[test]
    fn the_name_is_read_from_the_run_options_in_either_spelling_and_never_past_the_image() {
        // Both spellings of --name, as registries write them.
        assert_eq!(
            container_name(&argv("docker run --rm --init --name=brain-dock --gpus all --network host img --port 8001")),
            Some("brain-dock".into())
        );
        assert_eq!(
            container_name(&argv("docker run -it --name embed -e A=1 -v /a:/b img")),
            Some("embed".into())
        );
        // `--name` after the image belongs to the containerised program.
        assert_eq!(
            container_name(&argv("docker run --rm img --name nope")),
            None
        );
        assert_eq!(container_name(&argv("docker run --rm img")), None);
        assert_eq!(container_name(&argv("python --name x")), None);
    }

    #[test]
    fn the_cidfile_goes_right_after_run_exactly_once_and_other_argvs_are_untouched() {
        let cid = Path::new("/k/.compute/cid/embed.cid");
        let original = argv("docker run --rm --init --name=embed --gpus all img serve --port 8012");
        let once = inject_cidfile(&original, cid);
        assert_eq!(
            once,
            argv("docker run --cidfile /k/.compute/cid/embed.cid --rm --init --name=embed --gpus all img serve --port 8012")
        );
        assert_eq!(inject_cidfile(&once, cid), once, "idempotent");
        // A registry's own --cidfile (either spelling) is replaced, not doubled;
        // one after the image is the program's and stays.
        let own = argv("docker run --cidfile=/tmp/x --rm --cidfile /tmp/y img --cidfile z");
        assert_eq!(
            inject_cidfile(&own, cid),
            argv("docker run --cidfile /k/.compute/cid/embed.cid --rm img --cidfile z")
        );
        let plain = argv("vllm serve --cidfile x");
        assert_eq!(inject_cidfile(&plain, cid), plain);
    }

    #[test]
    fn a_start_removes_the_stale_cidfile_container_and_any_same_named_container() {
        let root = temp_root("prepare");
        let cidfile = cidfile_path(&root, "embed");
        fs::create_dir_all(cidfile.parent().unwrap()).unwrap();
        fs::write(&cidfile, OLD).unwrap();
        let docker = FakeDocker::default();
        docker
            .on(
                &format!("inspect -f {{{{.State.Status}}}} {OLD}"),
                Ok("exited\n"),
            )
            .on("ps -aq --no-trunc -f name=^embed$", Ok(&format!("{CID}\n")))
            .on("rm -f", Ok(""));
        let mut log = Vec::new();
        clear_for_start(
            &docker,
            &root,
            "embed",
            &argv("docker run --rm --name=embed img"),
            &mut log,
        )
        .unwrap();
        assert!(!cidfile.exists(), "docker refuses an existing cidfile");
        assert!(cidfile.parent().unwrap().is_dir());
        assert_eq!(
            docker.calls(),
            [
                format!("inspect -f {{{{.State.Status}}}} {OLD}"),
                format!("rm -f {OLD}"),
                "ps -aq --no-trunc -f name=^embed$".to_string(),
                format!("rm -f {CID}"),
            ]
        );
        let log = String::from_utf8(log).unwrap();
        assert!(
            log.contains(&format!("removing existing container {CID} named 'embed'")),
            "{log}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_start_with_no_leftovers_and_an_unreachable_daemon_still_proceeds() {
        let root = temp_root("clean");
        let docker = FakeDocker::default();
        docker.on("ps", Err("Cannot connect to the Docker daemon"));
        let mut log = Vec::new();
        clear_for_start(
            &docker,
            &root,
            "m",
            &argv("docker run --name=m.v2 img"),
            &mut log,
        )
        .unwrap();
        assert_eq!(docker.calls(), ["ps -aq --no-trunc -f name=^m\\.v2$"]);
        assert!(String::from_utf8(log).unwrap().contains("could not list"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_cidfile_wait_returns_the_id_and_gives_up_when_the_cli_dies() {
        let root = temp_root("wait");
        let cidfile = root.join("x.cid");
        let fast = Duration::from_millis(5);
        assert_eq!(
            wait_cidfile(&cidfile, &|| false, Duration::from_secs(5), fast),
            None
        );
        assert_eq!(
            wait_cidfile(&cidfile, &|| true, Duration::from_millis(30), fast),
            None
        );
        fs::write(&cidfile, "half").unwrap();
        assert_eq!(
            wait_cidfile(&cidfile, &|| true, Duration::from_millis(30), fast),
            None
        );
        fs::write(&cidfile, CID).unwrap();
        assert_eq!(
            wait_cidfile(&cidfile, &|| true, Duration::from_secs(5), fast),
            Some(CID.to_string())
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_legacy_record_is_resolved_by_the_name_in_its_cli_cmdline() {
        let root = temp_root("legacy");
        let pid_dir = root.join("4242");
        fs::create_dir_all(&pid_dir).unwrap();
        fs::write(
            pid_dir.join("cmdline"),
            "docker\0run\0--rm\0--init\0--name=brain-dock\0img\0--port\08001\0",
        )
        .unwrap();
        assert_eq!(
            handle(&root, None, 4242),
            Some(Handle::Name("brain-dock".into()))
        );
        assert_eq!(handle(&root, Some(CID), 4242), Some(Handle::Id(CID.into())));
        // Not docker, or no such pid: not a container.
        fs::create_dir_all(root.join("7")).unwrap();
        fs::write(root.join("7/cmdline"), "vllm\0serve\0").unwrap();
        assert_eq!(handle(&root, None, 7), None);
        assert_eq!(handle(&root, None, 8), None);

        let docker = FakeDocker::default();
        docker.on(
            "ps -aq --no-trunc -f name=^brain-dock$",
            Ok(&format!("{CID}\n")),
        );
        assert_eq!(
            resolve(&docker, &Handle::Name("brain-dock".into())).unwrap(),
            Some(CID.to_string())
        );
        docker.on("ps -aq --no-trunc -f name=^gone$", Ok(""));
        assert_eq!(
            resolve(&docker, &Handle::Name("gone".into())).unwrap(),
            None
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn only_a_gone_exited_dead_or_removing_container_counts_as_finished() {
        let docker = FakeDocker::default();
        docker.on("inspect -f {{.State.Status}} a", Ok("running\n"));
        docker.on("inspect -f {{.State.Status}} b", Err("permission denied"));
        assert_eq!(status(&docker, "a").unwrap().as_deref(), Some("running"));
        assert_eq!(status(&docker, "zzz").unwrap(), None, "no such object");
        assert!(status(&docker, "b").is_err());
        assert!(finished(None));
        for s in ["exited", "dead", "removing"] {
            assert!(finished(Some(s)), "{s}");
        }
        for s in ["created", "running", "restarting", "paused"] {
            assert!(!finished(Some(s)), "{s}");
        }
    }

    #[test]
    fn the_container_cgroup_comes_from_its_main_pid_with_the_systemd_path_as_fallback() {
        let base = temp_root("cg");
        let roots = crate::footprint::Roots {
            proc: base.join("proc"),
            cgroup: base.join("cgroup"),
        };
        let scope = format!("system.slice/docker-{CID}.scope");
        fs::create_dir_all(roots.proc.join("1533285")).unwrap();
        fs::write(roots.proc.join("1533285/cgroup"), format!("0::/{scope}\n")).unwrap();
        let docker = FakeDocker::default();
        docker.on(
            &format!("inspect -f {{{{.State.Pid}}}} {CID}"),
            Ok("1533285\n"),
        );
        assert_eq!(cgroup(&docker, &roots, CID), Some(scope.clone()));

        // Pid 0 (not running) and no systemd scope dir: unknown.
        let docker = FakeDocker::default();
        docker.on("inspect", Ok("0\n"));
        assert_eq!(cgroup(&docker, &roots, CID), None);
        fs::create_dir_all(roots.cgroup.join(&scope)).unwrap();
        assert_eq!(cgroup(&docker, &roots, CID), Some(scope));
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn stop_and_remove_treat_a_vanished_container_as_done() {
        let docker = FakeDocker::default();
        assert!(stop(&docker, CID, Duration::from_secs(30)).is_ok());
        assert!(remove(&docker, CID).is_ok());
        assert_eq!(
            docker.calls(),
            [format!("stop -t 30 {CID}"), format!("rm -f {CID}")]
        );
        docker.on(
            "rm",
            Err("Error response from daemon: removal of container x is already in progress"),
        );
        assert!(remove(&docker, CID).is_ok());
        docker.on("stop", Err("Cannot connect to the Docker daemon"));
        assert!(stop(&docker, CID, Duration::from_secs(1)).is_err());
    }
}
