"""Lumbridge Computer's vLLM worker extension: restore the weights after a level-2 sleep.

Loaded by every vLLM that has `serve.sleep` in the Lumbridge Computer registry:
`--worker-extension-cls lumbridge_computer_sleep_ext.LumbridgeComputerSleepExt`, with this
directory first on the engine's PYTHONPATH (`proc::launch_env`). Lumbridge Computer calls
it as step two of the level-2 wake:

    POST /wake_up?tags=weights                          # map the weight pool back
    POST /collective_rpc {"method":"reload_all_fast"}   # this file
    POST /wake_up?tags=kv_cache                         # map KV, resume scheduling

Why it exists (measured on GB10 with vLLM 0.27.1; see docs/sleep.md):

- Level 2 releases every `weights` allocation. Model *buffers* are saved and
  restored by vLLM, *parameters* are not, so a bare `/wake_up` serves garbage.
- vLLM's stock `reload_weights` RPC reloads only the target model:
  `Qwen3_5*.load_weights` skips the `mtp.` prefix, yet the MTP drafter lived in
  the same pool and was discarded with it. Acceptance falls from 0.85 to 0.00
  and decode gets 3-4x slower, with no error and correct-looking output.
- It is also slow on GB10: a device copy straight from a file-backed safetensors
  mmap runs at ~0.18 GB/s (42-52 s for 8 GB, ~3 min for a 22 GB brain).
  Cloning each tensor into anonymous memory first runs at ~6.7 GB/s, at the cost
  of one tensor's worth of transient host RAM. Reading whole shards eagerly
  (`--safetensors-load-strategy eager`) is not an option: it spiked host RAM by
  the shard size and nearly hit the box's memory floor.

`reload_all_fast` reloads the target through vLLM's layerwise reload path (which
re-runs quantization post-processing), then the drafter's `mtp.*` tensors with a
plain `load_weights` (the MTP layer is unquantized bf16 and shares embed_tokens
and lm_head with the target). Result on Qwen3.5-4B FP8 + MTP: 1.05-1.45 s,
byte-identical output, identical acceptance.

The reply is `results[0]` of `/collective_rpc`:
`{"target_secs", "has_draft", "draft_secs", "draft_tensors", "draft_loaded"}`.
Lumbridge Computer treats a wake of a model with a speculative config as failed unless
`draft_loaded` is true, and then stops the engine rather than serve from it.

Arguments arrive as strings (vLLM's dev RPC route only passes string args).
"""

import time

from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import get_model_loader


class LumbridgeComputerSleepExt:
    def reload_all_fast(self, draft: str = "1"):
        """Reload target (and, unless draft == "0", drafter) weights after a
        level-2 wake, cloning each mmap-backed tensor before the device copy."""
        mr = self.model_runner
        loader = get_model_loader(mr.load_config)
        out = {}

        t = time.perf_counter()
        weights = (
            (name, tensor.clone())
            for name, tensor in loader.get_all_weights(mr.model_config, mr.get_model())
        )
        self.reload_weights(weights_iterator=weights, is_checkpoint_format=True)
        out["target_secs"] = round(time.perf_counter() - t, 2)

        drafter = self.get_draft_model()
        out["has_draft"] = drafter is not None
        out["draft_loaded"] = False
        if drafter is not None and draft == "1":
            t = time.perf_counter()
            cfg = mr.vllm_config.speculative_config.draft_model_config
            with set_current_vllm_config(mr.vllm_config):
                weights = (
                    (name, tensor.clone())
                    for name, tensor in loader.get_all_weights(cfg, drafter)
                    if name.startswith("mtp.")
                )
                loaded = drafter.load_weights(weights)
            count = len(loaded or ())
            out["draft_secs"] = round(time.perf_counter() - t, 2)
            out["draft_tensors"] = count
            out["draft_loaded"] = count > 0
        return out
