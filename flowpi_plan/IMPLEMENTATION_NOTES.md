# FlowPi Implementation Notes

> Records file changes, deviations from ARCHITECTURE.md, and known limitations per milestone.

## M1 — SEA-RAFT Wrapper

**Files changed/added:**
- `SEA-RAFT/core/raft.py`: Added `return_low_res` parameter to `forward()`
- `src/openpi/training/sea_raft.py`: `SeaRaftFlowExtractor` class
- `src/openpi/training/sea_raft_test.py`: 3 tests

**No deviations from architecture.**

## M2 — Data Pipeline

**Files changed/added:**
- `src/openpi/training/lerobot_v3_dataset.py`: v3.0 parquet reader shim
- `src/openpi/transforms.py`: `ComputeFlow`, `LoadFlowCache`, `DelaySlowImage`, `normalize_flow`, `compute_image_frame_offsets`
- `src/openpi/training/config.py`: `FlowDataConfig`, `LeRobotAlohaDataConfig.flow`
- `src/openpi/training/data_loader.py`: v3 support, delta_timestamps for flow cache
- `src/openpi/policies/aloha_policy.py`: `AlohaInputs` flow/vlm_delay passthrough
- `scripts/precompute_flow_cache.py`: offline cache precomputation
- `src/openpi/training/data_loader_flow_test.py`: 4 tests

**Deviation**: `lerobot_v3_dataset.py` is a shim because the installed lerobot (0.1.0, v2.1 codebase) does not support v3.0. Upgrade lerobot or continue using this shim for formal training.

## M3 — Model Core

**Files changed/added:**
- `src/openpi/models/gemma.py`: `RMSNorm` per-position cond, `FlowGeom`, Block flow CA injection, 3-slot params in Module.setup
- `src/openpi/models/flow_tokenizer.py`: CNN + positional/lag/camera embeddings
- `src/openpi/models/model.py`: `Observation` fields `flow`, `flow_masks`, `vlm_delay`
- `src/openpi/models/pi0.py`: `embed_suffix` per-position tau, `embed_flow`, πR² `compute_loss`, `StreamingState`, `warm_start`, `denoise_step`, `refresh_prefix`
- `src/openpi/models/pi0_config.py`: `FlowConfig`, `get_freeze_filter`
- `src/openpi/models/flowpi_test.py`: 8 tests

**No deviations from architecture.** Key design decisions:
- Only `flow_gate` is zero-init; tokenizer and Q/K/V/O have normal initialization
- 3-slot flow CA params are in `Module.setup` (scan outer), not inside scanned Block
- `jax.lax.cond` prevents CA matmul execution on non-injection layers
- State token and flow tokens go through separate pathways into the Action Expert

## M4 — Training Configs & Smoke

**Files changed/added:**
- `src/openpi/training/config.py`: `debug_flowpi`, `flowpi_aloha` configs
- `src/openpi/training/weight_loaders.py`: `FlowPiWeightLoader`

**Deviation**: Dummy SigLIP is not available; `paligemma_variant="dummy"` only affects the LLM component. The vision tower always loads the real `So400m/14`. Dummy configs therefore still require GPU.

## M5 — Streaming Runtime

**Files changed/added:**
- `src/openpi/policies/flowpi_runtime.py`: `FlowPiRuntime` with frame ring buffer, online SEA-RAFT flow per tick, background prefix refresh
- `src/openpi/policies/flowpi_runtime_test.py`: 1 test (CPU, passes ~7 min on GPU)
- `scripts/flowpi_infer.py`: offline replay CLI
- `src/openpi/models/gemma.py`: fixed `flow=None` fallback path (batch_size from first non-None embedded)
- `flowpi_plan/HANDOVER.md`: agent handover document

**Bugs fixed during M5 completion:**
1. `ValueError: axis 3 is out of bounds` — flow data from `_compute_flow()` lacked batch dim. Fixed by adding `[None, ...]` when constructing `obs_with_flow`.
2. Duplicate `preprocess_observation` — `_prefix_forward` had preprocessing added but `warm_start`/`refresh_prefix` already preprocessed. Reverted in `_prefix_forward`, moved to runtime's `refresh_prefix`.
3. Test assertion bug — test expected `_prefix_age == i` regardless of refresh resets. Fixed with `expected_age = i - ((i-1)//5)*5`.

**Known limitation**: The runtime reads KV cache via `_slow_lock` in `tick()` but does not yet swap it into the streaming state (the background refresh thread KV cache is cached but not propagated before `denoise_step`). This does not affect the offline test but should be addressed before real deployment.

## M6 — Documentation

**Files added:**
- `flowpi_plan/README_USAGE.md`: complete usage guide
- `flowpi_plan/IMPLEMENTATION_NOTES.md`: this document

## Ruff Status

Ruff auto-fix was applied to M5 files. Remaining warnings/errors in `src/` and `scripts/` are pre-existing from M1-M4 (unused imports, `# noqa` directives for disabled rules, PERF401 suggestions, etc.). A full cleanup pass was not completed.

## Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `flowpi_test.py` (M3) | 8 | All passing |
| `pi0_test.py` (baseline) | 4 | All passing |
| `flowpi_runtime_test.py` (M5) | 1 | Passing |
| `sea_raft_test.py` (M1) | 3 | All passing (per M1 commit) |
| `data_loader_flow_test.py` (M2) | 4 | All passing (per M2 commit) |

## Known Limitations

1. **V1 fixed at 50Hz**: Control frequency must match dataset FPS. No variable-frequency support.
2. **PyTorch path unsupported**: `models_pytorch` training path does not support FlowPi features.
3. **lerobot v3.0 shim**: The installed lerobot version (0.1.0) does not support v3.0 datasets natively. The `lerobot_v3_dataset.py` shim works but should be replaced when lerobot is upgraded.
4. **Dummy variant requires GPU**: The dummy SigLIP (`paligemma_variant="dummy"`) still loads the real vision tower.
5. **Ruff not fully clean**: Pre-existing warnings from M1-M4 remain. No functional impact.
6. **Runtime KV cache swap**: Background refresh KV cache is stored but not propagated to the streaming state during ticks (see M5 known limitation above).
", "filePath": "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/flowPi/flowpi_plan/IMPLEMENTATION_NOTES.md"}