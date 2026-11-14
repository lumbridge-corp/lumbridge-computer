use super::*;
use std::io::Write;
use std::net::TcpListener;
use std::thread;

// Captures of real engines' /metrics (see each file's first line).
const VLLM_GENERATE: &str = include_str!("../testdata/vllm-0.27-generate.prom");
const VLLM_EMBED: &str = include_str!("../testdata/vllm-pooling-embed.prom");
const VLLM_RERANK: &str = include_str!("../testdata/vllm-pooling-rerank.prom");
const VLLM_OMNI: &str = include_str!("../testdata/vllm-omni-tts.prom");
const NEMO_SPEECH: &str = include_str!("../testdata/nemo-speech-asr.prom");
// SYNTHETIC, hand-written in each engine's metric names.
const SGLANG: &str = include_str!("../testdata/sglang-synthetic.prom");
const LLAMA_CPP: &str = include_str!("../testdata/llamacpp-synthetic.prom");
const VLLM_LEGACY: &str = include_str!("../testdata/vllm-legacy-synthetic.prom");

fn model(yaml: &str) -> Model {
    serde_yaml::from_str(yaml).unwrap()
}

#[test]
fn real_vllm_generate_capture_normalises_to_idle_with_token_totals() {
    assert_eq!(
        parse_load(Engine::Vllm, VLLM_GENERATE),
        Some(EngineLoad {
            running: 0,
            waiting: 0,
            kv_usage: Some(0.0),
            prompt_tokens_total: Some(754_197.0),
            generation_tokens_total: Some(18_883.0),
        })
    );
}

#[test]
fn real_vllm_pooling_captures_report_prompt_tokens_and_zero_generation() {
    let embed = parse_load(Engine::Vllm, VLLM_EMBED).unwrap();
    assert_eq!(embed.running, 0);
    assert_eq!(embed.prompt_tokens_total, Some(30_991_928.0));
    assert_eq!(embed.generation_tokens_total, Some(0.0));
    let rerank = parse_load(Engine::Vllm, VLLM_RERANK).unwrap();
    assert_eq!(rerank.prompt_tokens_total, Some(731_109.0));
    assert_eq!(rerank.kv_usage, Some(0.0));
}

#[test]
fn real_vllm_omni_capture_with_replica_and_stage_labels_is_busy() {
    let load = parse_load(Engine::Vllm, VLLM_OMNI).unwrap();
    assert_eq!(load.running, 1);
    assert_eq!(load.waiting, 0);
    assert!((load.kv_usage.unwrap() - 0.003_298_515_667_949_475_4).abs() < 1e-12);
    assert_eq!(load.prompt_tokens_total, Some(140_001.0));
    assert_eq!(load.generation_tokens_total, Some(11_700.0));
}

#[test]
fn real_nemo_speech_capture_without_an_in_flight_gauge_is_unknown_not_idle() {
    // It exports `speech_stream_active 0`, but HTTP transcriptions in
    // progress are not counted anywhere, so 0 would be a guess.
    assert_eq!(parse_load(Engine::NemoSpeech, NEMO_SPEECH), None);
    assert_eq!(parse_load(Engine::Generic, NEMO_SPEECH), None);
}

#[test]
fn nemo_speech_with_an_in_flight_gauge_adds_open_streams() {
    let text = format!("{NEMO_SPEECH}speech_requests_in_flight{{model=\"speech-asr\"}} 2\n");
    let load = parse_load(Engine::NemoSpeech, &text).unwrap();
    assert_eq!(load.running, 2);
    assert_eq!(load.kv_usage, None);
    assert_eq!(load.prompt_tokens_total, None);
}

#[test]
fn synthetic_sglang_sums_dp_ranks_skips_priority_series_and_takes_fullest_cache() {
    assert_eq!(
        parse_load(Engine::Sglang, SGLANG),
        Some(EngineLoad {
            running: 5,
            waiting: 1,
            kv_usage: Some(0.5),
            prompt_tokens_total: Some(166_238.0),
            generation_tokens_total: Some(7_500.0),
        })
    );
}

#[test]
fn synthetic_llama_cpp_maps_processing_deferred_and_kv_ratio() {
    assert_eq!(
        parse_load(Engine::LlamaCpp, LLAMA_CPP),
        Some(EngineLoad {
            running: 2,
            waiting: 1,
            kv_usage: Some(0.42),
            prompt_tokens_total: Some(51_234.0),
            generation_tokens_total: Some(2_048.0),
        })
    );
}

#[test]
fn older_vllm_cache_gauge_name_is_still_read() {
    let load = parse_load(Engine::Vllm, VLLM_LEGACY).unwrap();
    assert_eq!(load.running, 2, "summed across model_name label sets");
    assert_eq!(load.waiting, 4);
    assert_eq!(load.kv_usage, Some(0.875));
}

#[test]
fn generic_engine_is_resolved_from_the_metric_names() {
    assert_eq!(
        parse_load(Engine::Generic, VLLM_GENERATE),
        parse_load(Engine::Vllm, VLLM_GENERATE)
    );
    assert_eq!(
        parse_load(Engine::Generic, SGLANG),
        parse_load(Engine::Sglang, SGLANG)
    );
    assert_eq!(
        parse_load(Engine::Generic, LLAMA_CPP),
        parse_load(Engine::LlamaCpp, LLAMA_CPP)
    );
    assert_eq!(sniff(&prom::parse(VLLM_OMNI)), Some(Engine::Vllm));
}

#[test]
fn the_wrong_engine_or_missing_running_gauge_is_unknown_never_idle() {
    assert_eq!(parse_load(Engine::Sglang, VLLM_GENERATE), None);
    assert_eq!(parse_load(Engine::Vllm, LLAMA_CPP), None);
    assert_eq!(parse_load(Engine::Vllm, ""), None);
    assert_eq!(parse_load(Engine::Generic, "up 1\n"), None);
    assert_eq!(
        parse_load(
            Engine::Vllm,
            "vllm:num_requests_running{engine=\"0\"} NaN\n"
        ),
        None
    );
}

#[test]
fn missing_token_counters_stay_none() {
    let load = parse_load(Engine::Vllm, "vllm:num_requests_running 3\n").unwrap();
    assert_eq!(load.running, 3);
    assert_eq!(load.waiting, 0);
    assert_eq!(load.kv_usage, None);
    assert_eq!(load.prompt_tokens_total, None);
    assert_eq!(load.generation_tokens_total, None);
}

/// The pre-`engines` liveness parser, verbatim except that it uses the new
/// Prometheus parser. `vllm_progress` must match it exactly.
fn old_vllm_progress(metrics: &str) -> Option<EngineProgress> {
    let samples = prom::parse(metrics);
    let sums: Vec<Option<f64>> = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
    ]
    .iter()
    .map(|name| prom::sum(&samples, name))
    .collect();
    let (running, generation) = (sums[0]?, sums[3]?);
    let waiting = sums[1].unwrap_or(0.0);
    Some(EngineProgress {
        running,
        waiting,
        in_flight: running + waiting,
        tokens: generation + sums[2].unwrap_or(0.0),
    })
}

#[test]
fn vllm_progress_is_unchanged_on_every_fixture() {
    for fixture in [
        VLLM_GENERATE,
        VLLM_EMBED,
        VLLM_RERANK,
        VLLM_OMNI,
        VLLM_LEGACY,
        NEMO_SPEECH,
        SGLANG,
        LLAMA_CPP,
        "",
        "up 1\n",
        "vllm:num_requests_running{engine=\"0\"} 1.0\n",
        "vllm:num_requests_running 2.5\nvllm:generation_tokens_total 1\n",
    ] {
        assert_eq!(vllm_progress(fixture), old_vllm_progress(fixture));
    }
    assert_eq!(
        vllm_progress(VLLM_GENERATE),
        Some(EngineProgress {
            running: 0.0,
            waiting: 0.0,
            in_flight: 0.0,
            tokens: 754_197.0 + 18_883.0,
        })
    );
    assert_eq!(
        vllm_progress(VLLM_OMNI),
        Some(EngineProgress {
            running: 1.0,
            waiting: 0.0,
            in_flight: 1.0,
            tokens: 140_001.0 + 11_700.0,
        })
    );
}

#[test]
fn explicit_engine_field_wins_and_registries_without_it_still_parse() {
    let plain = model("name: p\nfootprint_gb: 1\nserve:\n  kind: exec\n  port: 9000\n");
    assert_eq!(plain.serve.engine, None);
    assert_eq!(detect(&plain), Engine::Generic);
    let explicit = model(
        "name: asr\nfootprint_gb: 1\nserve:\n  kind: python\n  engine: nemo-speech\n  port: 8791\n  command: [\"~/env/bin/python\", \"~/asr_api.py\"]\n",
    );
    assert_eq!(detect(&explicit), Engine::NemoSpeech);
    assert!(serde_yaml::from_str::<Model>(
        "name: x\nfootprint_gb: 1\nserve:\n  kind: exec\n  engine: tgi\n"
    )
    .is_err());
}

#[test]
fn engine_is_detected_from_kind_and_from_real_world_argv() {
    let cases = [
        // A brain on the kind-based vLLM builder.
        ("kind: vllm\n  executable: ~/vllm27-env/bin/vllm", Engine::Vllm),
        // A TTS model: python wrapper running `vllm serve --omni`.
        (
            "kind: exec\n  command: [/home/sam/tts/venv/bin/python, /home/sam/vllm-omni-env/bin/vllm, serve, /m]",
            Engine::Vllm,
        ),
        (
            "kind: exec\n  command: [python3, -m, vllm.entrypoints.openai.api_server]",
            Engine::Vllm,
        ),
        // A pinned SGLang container running `sglang serve`.
        (
            "kind: docker\n  command: [docker, run, -v, /home/sam/.cache/vllm:/root/.cache/vllm, \"lmsysorg/sglang@sha256:3c0a\", sglang, serve]",
            Engine::Sglang,
        ),
        (
            "kind: exec\n  command: [python, -m, sglang.launch_server, --port, '30000']",
            Engine::Sglang,
        ),
        // A custom image: the engine is only in the image name.
        (
            "kind: docker\n  command: [docker, run, --name=brain-dock, -v, /home/sam/.cache/vllm:/root/.cache/vllm, \"example/vllm-custom:77154ca6\", org/brain-74b]",
            Engine::Vllm,
        ),
        // A llama-server built from source.
        (
            "kind: exec\n  command: [/home/sam/src/llama.cpp/build/bin/llama-server, -m, /m.gguf, --metrics]",
            Engine::LlamaCpp,
        ),
        (
            "kind: exec\n  command: [/home/sam/vendor/NeMo-Speech.cpp/build/cuda-server/bin/nemo-speech, serve]",
            Engine::NemoSpeech,
        ),
        // vision: a vLLM shim launcher, not recognisable from argv; set `engine:`.
        (
            "kind: exec\n  command: [~/vllm27-env/bin/python3, ~/vision_launch.py, serve]",
            Engine::Generic,
        ),
        // brain-ds4: opaque container image.
        (
            "kind: docker\n  command: [docker, run, -v, /x/vllm:/y, 0xsero/ds4-spark:latest]",
            Engine::Generic,
        ),
        ("kind: diffusers\n  command: [~/flux-env/bin/python, ~/flux_server.py]", Engine::Generic),
    ];
    for (serve, expected) in cases {
        let m = model(&format!(
            "name: m\nfootprint_gb: 1\nserve:\n  port: 8001\n  {serve}\n"
        ));
        assert_eq!(detect(&m), expected, "{serve}");
    }
}

#[test]
fn configured_concurrency_reads_each_engines_flag() {
    let llama = model(
        "name: m\nfootprint_gb: 1\nserve:\n  kind: exec\n  command: [llama-server, -np, '2']\n",
    );
    assert_eq!(configured_concurrency(&llama), 2);
    let vllm =
        model("name: m\nfootprint_gb: 1\nserve:\n  kind: vllm\n  args:\n    max-num-seqs: 4\n");
    assert_eq!(configured_concurrency(&vllm), 4);
    let none = model("name: m\nfootprint_gb: 1\nserve:\n  kind: exec\n");
    assert_eq!(configured_concurrency(&none), 0);
}

/// A one-request fake engine. Other tests probe ports concurrently, so a
/// connection that does not send our GET is ignored rather than answered.
fn serve_metrics(body: &'static str) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            if !http::tests::read_request_head(&mut stream).starts_with("GET /metrics ") {
                continue;
            }
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            return;
        }
    });
    port
}

#[test]
fn scrape_reads_the_models_own_loopback_port() {
    let port = serve_metrics(VLLM_OMNI);
    let m = model(&format!(
        "name: tts\nfootprint_gb: 1\nserve:\n  kind: vllm\n  port: {port}\n"
    ));
    let load = scrape(&m).unwrap().unwrap();
    assert_eq!(load.running, 1);
}

#[test]
fn scrape_of_an_endpoint_without_load_metrics_is_ok_none() {
    let port = serve_metrics("python_gc_objects_collected_total{generation=\"0\"} 1\n");
    let m = model(&format!(
        "name: tts\nfootprint_gb: 1\nserve:\n  kind: vllm\n  port: {port}\n"
    ));
    assert_eq!(scrape(&m).unwrap(), None);
}

#[test]
fn scrape_fails_fast_on_a_closed_port_or_a_silent_server() {
    // Port 1 (tcpmux) is privileged and unused: refused at once. A freshly
    // released ephemeral port could be reused by a parallel test's server.
    let m = model("name: gone\nfootprint_gb: 1\nserve:\n  kind: vllm\n  port: 1\n");
    assert!(scrape(&m).is_err());

    // Accepts but never answers: bounded by the timeout, not hung.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let holder = thread::spawn(move || {
        let held = listener.accept().unwrap();
        thread::sleep(Duration::from_millis(600));
        drop(held);
    });
    let m = model(&format!(
        "name: stuck\nfootprint_gb: 1\nserve:\n  kind: vllm\n  port: {port}\n"
    ));
    let started = std::time::Instant::now();
    assert!(scrape_within(&m, Duration::from_millis(200)).is_err());
    assert!(started.elapsed() < Duration::from_millis(550));
    holder.join().unwrap();

    let portless = model("name: p\nfootprint_gb: 1\nserve:\n  kind: vllm\n");
    assert!(scrape(&portless).is_err());
}
