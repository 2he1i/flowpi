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
- `src/openpi/policies/flowpi_runtime_test.py`: 1 test
- `scripts/flowpi_infer.py`: offline replay CLI
- `src/openpi/models/gemma.py`: fixed `flow=None` fallback path (batch_size from first non-None embedded)
- `flowpi_plan/HANDOVER.md`: agent handover document

**Bugs fixed during M5 completion:**
1. `ValueError: axis 3 is out of bounds` — flow data from `_compute_flow()` lacked batch dim. Fixed by adding `[None, ...]` when constructing `obs_with_flow`.
2. Duplicate `preprocess_observation` — `_prefix_forward` had preprocessing added but `warm_start`/`refresh_prefix` already preprocessed. Reverted in `_prefix_forward`, moved to runtime's `refresh_prefix`.
3. Test assertion bug — test expected `_prefix_age == i` regardless of refresh resets. Fixed with `expected_age = i - ((i-1)//5)*5`.

## M6 — Documentation

**Files added:**
- `flowpi_plan/README_USAGE.md`: complete usage guide
- `flowpi_plan/IMPLEMENTATION_NOTES.md`: this document

## M7 — Three-channel training freshness

FlowPI training now uses the same freshness contract as the intended asynchronous runtime:

```text
Fast channel:
    state[t] and actions[t:] at the current dataset/control tick

Flow channel:
    cached or online flow targeted at s = t - flow_delay
    flow[s] still contains F_(s-k*stride -> s) for each internal lag k
    flow_delay is sampled independently from vlm_delay

Slow channel:
    VLM image/prefix observation[t-vlm_delay]
    vlm_delay is sampled independently from flow_delay
```

`FlowConfig.flow_delay_max` and `flow_delay_distribution` mirror the existing VLM delay
interface. Delay support is clamped and renormalized at episode start, so neither channel can
read across an episode boundary. `Observation` carries both ages, and the flow tokenizer adds a
normally initialized global flow-age embedding in addition to its existing internal lag
embedding. The Flow cross-attention gate remains zero-initialized.

The precomputed cache has a strict row convention: cache row `s` is SEA-RAFT flow ending at
episode tick `s`. Training simulates asynchronous channel freshness by selecting historical
cached observations; future online runtime will replace this sampling with actual worker
completion timestamps while preserving exactly the same model input semantics.

Training keeps the dataset/control tick at the dataset FPS (V1: 50 Hz). The online runtime still
publishes the synchronous flow observation with `flow_delay=0`, but its telemetry now exposes the
future contract fields `flow_source_tick`, `flow_delay_ticks`, and `flow_delay_ms`.

## M8 — Flow-required training comparison

The baseline remains exactly compatible with the previous zero-gate recipe:
`FlowConfig.flow_gate_init=0.0`, `FlowDataConfig.flow_required_prob=0.0`, and
`flow_required_vlm_delay_min=0` preserve the prior sampling and identity initialization.

The opt-in comparison launcher `scripts/train_flowpi_flow_required_8xh200.sh` adds two targeted
changes for the modality-laziness experiment:

```text
Flow-required samples:
    with probability p, sample VLM delay from [flow_required_vlm_delay_min, vlm_delay_max]
    while sampling flow_delay independently from the configured flow distribution

Flow gate:
    initialize raw gate parameters to atanh(flow_gate_init), so tanh(gate) starts at the
    requested small magnitude (default comparison value: 0.01)
```

This is the slow-prefix dropout variant: the competing VLM prefix is deliberately stale on a
subset of samples, while the cached Flow branch remains a real historical observation with its
own independently sampled age. It does not add a Flow reconstruction loss, alter πR², or change
the internal `K × stride` meaning of a cache row. The comparison run uses a separate log and
checkpoint namespace, leaving the original `flowpi_8xh200` and its step-2000 checkpoint intact.

Training simulates asynchronous channel freshness by selecting historical cached observations.
Future online runtime will replace this sampling with actual worker completion timestamps while
preserving exactly the same model input semantics.

## Review Fixes (GPT-review pass, main branch)

### P0 — Offline cache correctness
- `src/openpi/training/lerobot_v3_dataset.py`: rewritten indexing — the parquet shim now maps global dataset indices to `(chunk, file, row)` via per-file episode metadata (`episode_frames`, `episodes`) instead of assuming a single contiguous file, and validates that episodes sharing a parquet file tile the global index range contiguously (`itertools.pairwise` in `_file_index_range`).
- `scripts/precompute_flow_cache.py`: rewritten around `episode_frames` — the cache no longer indexes flow rows by an episode-relative offset computed from a single shared `frame_offsets`; it slices each episode's frames by its own flow-frame range so the cache rows line up exactly with the data loader's `LoadFlowCache` reads.
- NEW `src/openpi/policies/flowpi_e2e_test.py` (2 tests): proves the offline cache and the streaming runtime produce bit-identical flow inputs (`test_cache_flow_equals_runtime_flow`) and that `DelaySlowImage` produces exactly the delayed frame the runtime's prefix refresh would (delay semantics equivalence, including clamping at episode start). Both run the real SEA-RAFT precompute once per session (module-scoped fixture).
  - Encoding detail: the runtime's `(img + 1) * 127.5 → clip → uint8` round-trip fails for 63/256 u8 values after `u8/127.5 - 1`; `_item_to_obs` bumps one ulp via `np.nextafter` so the round-trip is exact for all 256 levels (verified empirically).
- `src/openpi/policies/flowpi_runtime_test.py`: now asserts against `runtime._ring.base_index` (was `_frame_index`) and passes with the E2E tolerance.

### P1 — Delay distribution & training telemetry
- `src/openpi/models/pi0_config.py`: `FlowConfig.vlm_delay_distribution: tuple[float, ...] | None` — optional training-time histogram over `[0, vlm_delay_max]`, validated at construction (length `vlm_delay_max + 1`, non-negative weights, positive mass).
- `src/openpi/transforms.py`: `DelaySlowImage(distribution=...)` samples the delay categorically from the fitted histogram, renormalized over the delays reachable at each frame (`[0, min(vlm_delay_max, frame_index)]`, matching the runtime's clamping); falls back to uniform when the fitted mass lies entirely beyond the reachable range (early episode).
- NEW `scripts/fit_vlm_delay.py`: fits the histogram from `flowpi_infer.py --realtime --telemetry-json` output — prints the delay histogram, P99 → recommended `vlm_delay_max`, and the `0.8 · P̂ + 0.2 · U` smoothed weights to paste into `FlowConfig`.
- `scripts/flowpi_infer.py`: new `--telemetry-json PATH` option dumps per-tick `vlm_delay_max` / `telemetry` / `stats` for the fitter.
- `src/openpi/models/gemma.py`: `Block` now also returns the per-layer gated cross-attention residual ratio `r_l = |g_l C_l| / |h_l|` (float32 scalar; 0.0 for non-injection layers), stacked by the layer scan; `Module.__call__(return_flow_stats=True)` returns `(outputs, kv_cache, flow_stats)` with the ratios sliced to the injection layers. The scan's return stays `(carry, ys)` — the ratio rides in the `ys` tree.
- `src/openpi/models/pi0.py`: `compute_loss_and_metrics` — same forward as `compute_loss` plus a flat telemetry dict (empty for baseline configs): `frac_pir2`, `loss_standard`/`loss_pir2` (per-row masks dividing by `n_rows · H` so the weighted combination reproduces the outer renormalized loss mean exactly), `loss_{front,mid,tail}_third`, `loss_tau_{low,mid,high}`, and per-injection-layer `flow_ca_residual_ratio_layer{l}`. `compute_loss` delegates.
- `scripts/train.py`: `loss_fn` returns `(mean_loss, metrics)` through `nnx.value_and_grad(..., has_aux=True)`; `train_step` merges the forward metrics with parameter-level telemetry from the post-update param tree (`flow_gate_tanh_abs_layer{l}` = mean `|tanh(gate)|` per injection layer, `flow_delay_emb_norm` = delay-embedding RMS) via `nnx_utils.PathRegex`.
- `src/openpi/training/config.py`: NEW `debug_flow` training config (dummy variants + flow), exercised by `scripts/train_test.py` which is now parametrized over `["debug", "debug_flow"]` and no longer forces `JAX_PLATFORMS=cpu` (tests run on GPU).

### P2 — Documentation (this file)
- Removed the "KV cache swap is not implemented" limitation: `_slow_lock` read-back + `refresh_prefix` KV swap into the streaming state **is** implemented and covered by `test_prefix_refresh_swaps_kv_and_carries_source_tick` (flowpi_test.py) and the E2E delay-semantics test.

## Ruff Status

All `src/openpi` and `scripts` files touched by the review fixes pass `ruff check` and `ruff format --check`. Pre-existing warnings from M1-M4 in untouched files remain.

## Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `flowpi_test.py` (M3) | 16 | All passing |
| `pi0_test.py` (baseline) | 4 | All passing |
| `flowpi_runtime_test.py` (M5) | 1 | Passing |
| `sea_raft_test.py` (M1) | 3 | All passing (per M1 commit) |
| `data_loader_flow_test.py` (M2) | 9 | All passing |
| `lerobot_v3_dataset_test.py` (P0) | 8 | All passing |
| `flowpi_e2e_test.py` (P0) | 2 | All passing (~4 min, real SEA-RAFT) |
| `train_test.py` | 2 (`debug`, `debug_flow`) | All passing |

## Known Limitations

1. **V1 fixed at 50Hz**: Control frequency must match dataset FPS. No variable-frequency support.
2. **PyTorch path unsupported**: `models_pytorch` training path does not support FlowPi features.
3. **lerobot v3.0 shim**: The installed lerobot version (0.1.0) does not support v3.0 datasets natively. The `lerobot_v3_dataset.py` shim works but should be replaced when lerobot is upgraded.
4. **Dummy variant requires GPU**: The dummy SigLIP (`paligemma_variant="dummy"`) still loads the real vision tower.
