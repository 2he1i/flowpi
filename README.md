# FlowPI

**FlowPI is an experimental high-frequency vision-feedback policy built on π0.5 / πR²-style streaming action generation, with optical flow injected into the fast Action Expert path.**

The project studies a simple question: **can motion cues be refreshed more frequently than the expensive VLM prefix, and used to correct an action trajectory before the next full visual-language update arrives?**

FlowPI therefore separates visual reasoning into a slow semantic path and a lightweight motion-feedback path. SEA-RAFT provides short-term optical flow; a small tokenizer converts flow fields into motion tokens; and gated cross-attention injects those tokens into selected Action Expert layers during streaming denoising.

> **Research status:** experimental. The repository contains the training pipeline, flow cache generation, FlowPI model, ablation switches, and a three-device reference runtime. The current runtime is intentionally sequential and is **not yet a production asynchronous controller**. Initial rollout behavior is encouraging, but controlled baseline and ablation evaluation is still required before attributing improvements to the flow branch.

---

## Motivation

A standard VLA policy typically couples new visual information to a relatively expensive vision-language forward pass. This creates a mismatch between:

- the frequency at which the environment changes,
- the frequency at which visual semantics can be recomputed, and
- the frequency at which actions can be regenerated.

FlowPI explores a two-timescale design:

1. **Slow semantic channel** — image + language are encoded into the VLM prefix / KV cache.
2. **Fast feedback channel** — fresh robot state and optical flow are injected while the Action Expert performs streaming NFEs.

The intended behavior is not to replace semantic perception with optical flow. Optical flow only supplies **short-horizon motion evidence** that can help the action generator react between expensive semantic refreshes.

---

## Architecture

```text
                 ┌─────────────────────────────────────┐
 RGB + language ─►        Slow VLM / prefix path       │
                 │      SigLIP + PaliGemma prefix      │
                 └────────────────┬────────────────────┘
                                  │ cached KV
                                  ▼
                           ┌───────────────┐
 fresh robot state ───────►│ Action Expert │──────► actions
                           │   fast NFE     │
                           └───────▲───────┘
                                   │ gated cross-attention
                                   │
 RGB history ─► SEA-RAFT ─► Flow tokenizer ─► flow tokens
```

### 1. π0.5 backbone

FlowPI keeps the π0.5-style VLM + Action Expert decomposition and uses the Action Expert as the fast action-generation path.

When `Pi0Config.flow is None`, the model graph remains the baseline π0.5 graph.

### 2. Optical-flow fast path

For each configured camera, SEA-RAFT estimates motion between the current frame and several historical frames. The default FlowPI configuration uses:

- `K = 2` flow history steps,
- frame stride `Δ = 3`,
- full-resolution flow input size `480 × 640`,
- SEA-RAFT low-resolution flow at `H/8 × W/8`,
- normalized and clamped flow before tokenization.

The flow cache is generated offline during training so SEA-RAFT is not part of the policy backward pass.

### 3. Flow tokenizer

Low-resolution 2-D flow fields are converted into a compact sequence of motion tokens by `FlowTokenizer`.

The tokenizer also receives flow age / delay information so the model can distinguish fresh motion estimates from stale ones.

### 4. Gated flow cross-attention

Flow tokens are injected into selected Action Expert layers through dedicated cross-attention blocks.

Default injection layers:

```text
7, 12, 16
```

Each injected residual is gated, allowing the pretrained Action Expert path to remain close to its original behavior while the flow branch learns how much motion correction is useful.

### 5. Fresh-state fast channel

Robot state can be re-encoded into the Action Expert suffix at every NFE through `flow_state_proj`.

This avoids forcing the fast path to rely only on the potentially stale state information embedded in the cached slow prefix.

### 6. Delay conditioning

FlowPI explicitly models two different forms of staleness:

- **`vlm_delay`** — age of the cached slow VLM prefix,
- **`flow_delay`** — age of the available optical-flow observation.

The Action Expert receives VLM delay through an adaRMS conditioning embedding. Flow delay is provided to the flow tokenizer.

Training can sample either delay uniformly or from an empirical histogram fitted from runtime telemetry.

---

## Streaming action generation

FlowPI includes a πR²-style staircase noise schedule for streaming action updates.

For an action horizon of length `H`, a sampled width `d` divides the action chunk into three regions:

```text
[ clean / already committed ][ partially denoised ][ noise / future ]
<--------- d ----------->                     <--- d --->
```

The clean action prefix is inpainted and excluded from the training loss, while the remaining positions use position-dependent noise levels.

The current training recipe mixes:

- **πR² staircase samples** for streaming behavior, and
- **standard scalar-time flow-matching samples** to retain the original denoising distribution.

Default values are:

```text
d_max       = 5
p_standard  = 0.2
tau_jitter  = 0.01
```

The implementation also logs loss by schedule type, horizon region and noise bucket, together with flow cross-attention residual ratios.

---

## Ablations

FlowPI keeps a common parameter layout across its ablations, so the same checkpoint architecture can be used while disabling individual fast-path signals.

```python
FlowConfig(
    use_fresh_state=True,
    use_delay=True,
    use_flow=True,
    use_pir2=True,
)
```

Available switches:

| Switch | Meaning |
| --- | --- |
| `use_flow` | Flow tokenizer + gated flow cross-attention |
| `use_fresh_state` | Fresh robot-state token in every fast NFE |
| `use_delay` | Slow-prefix age conditioning |
| `use_pir2` | πR² staircase schedule instead of standard scalar-time FM |

These switches are intended for controlled attribution rather than defining separate model architectures.

---

## Data and optical-flow cache

FlowPI currently targets LeRobot v3 datasets with three RGB cameras:

```text
base_0_rgb
left_wrist_0_rgb
right_wrist_0_rgb
```

Training uses an **offline SEA-RAFT cache**. This keeps policy training deterministic with respect to the flow model and avoids repeatedly running a Torch optical-flow network inside the JAX training loop.

Because cached flow is computed in the coordinate system of the raw image, FlowPI disables geometric image augmentation by default. Photometric augmentation can still be used without breaking image/flow spatial alignment.

---

## Setup

```bash
uv sync --group dev
git submodule update --init SEA-RAFT
uv run python scripts/setup_sea_raft.py
```

The pinned SEA-RAFT submodule is patched by:

```text
third_party/sea_raft/flowpi_return_low_res.patch
```

The patch adds the low-resolution flow API used by FlowPI, avoids unnecessary ImageNet initialization when restoring a trained SEA-RAFT model, and makes large correlation sampling safe.

SEA-RAFT weights are external and are not stored in this repository.

---

## Training

### 1. Compute normalization statistics

```bash
uv run python scripts/compute_norm_stats.py flowpi_aloha \
  --data.repo-id /path/to/lerobot-dataset \
  --data.assets.assets-dir /path/to/assets \
  --data.assets.asset-id flowpi \
  --data.flow.enabled false
```

### 2. Precompute SEA-RAFT flow

```bash
uv run python scripts/precompute_flow_cache.py flowpi_aloha \
  --data.repo-id /path/to/lerobot-dataset \
  --data.assets.assets-dir /path/to/assets \
  --data.assets.asset-id flowpi \
  --data.flow.flow-cache-dir /path/to/flow-cache \
  --data.flow.sea-raft-ckpt /path/to/sea-raft.pth \
  --data.flow.sea-raft-device cuda:0
```

### 3. Train FlowPI

```bash
uv run python scripts/train.py flowpi_aloha \
  --exp-name flowpi-run \
  --checkpoint-base-dir /path/to/checkpoints \
  --data.repo-id /path/to/lerobot-dataset \
  --data.assets.assets-dir /path/to/assets \
  --data.assets.asset-id flowpi \
  --data.flow.flow-cache-dir /path/to/flow-cache \
  --data.flow.sea-raft-ckpt /path/to/sea-raft.pth \
  --weight-loader.params-path gs://openpi-assets/checkpoints/pi05_base/params
```

`--checkpoint-base-dir`, `--data.repo-id`, `--data.flow.flow-cache-dir`, `--data.flow.sea-raft-ckpt`, and `--weight-loader.params-path` are independent paths. The repository does not assume local datasets, caches, checkpoints, or SEA-RAFT weights.

---

## Reference inference runtime

The repository currently provides a **sequential three-device reference runner**:

```bash
uv run python scripts/flowpi_infer.py \
  --config-name flowpi_aloha \
  --checkpoint /path/to/checkpoint/step \
  --dataset /path/to/lerobot-dataset \
  --output /path/to/results/actions.npz \
  --sea-raft-ckpt /path/to/sea-raft.pth \
  --slow-jax-device gpu:0 \
  --fast-jax-device gpu:1 \
  --sea-raft-device cuda:2
```

The three logical devices are:

| Device | Role |
| --- | --- |
| GPU 0 | Slow VLM prefix / KV-cache computation |
| GPU 1 | Fast Action Expert streaming NFE |
| GPU 2 | SEA-RAFT optical flow |

For each input frame, the current reference runtime executes:

```text
flow → slow prefix refresh → one fast NFE
```

It then writes the requested action array to `.npz`.

### Important limitation

This runner is deliberately minimal. It currently has no:

- simulator adapter,
- asynchronous producer/consumer channels,
- wall-clock frequency controller,
- dynamic action-width scheduler,
- automatic slow-prefix refresh policy,
- production robot interface.

The architectural separation is designed to support a later asynchronous implementation, but the current runner should be interpreted as a correctness / integration reference rather than a measured high-frequency deployment stack.

---

## Verification

CPU/static checks are kept separate from hardware validation.

```bash
uv run ruff check \
  src/openpi/models/pi0.py \
  src/openpi/policies/flowpi_runtime.py \
  src/openpi/training/sea_raft.py \
  scripts/flowpi_infer.py \
  scripts/flowpi_checkpoint.py \
  scripts/compute_norm_stats.py

uv run python -m compileall -q src scripts
```

GPU training, SEA-RAFT validation and rollout evaluation are intentionally not part of the basic verification command.

---

## Current research questions

The repository is being used to evaluate several hypotheses rather than presenting a finished benchmark result:

1. Does optical-flow feedback improve recovery from target or object motion between slow VLM updates?
2. How much of any improvement comes from flow itself versus fresh proprioceptive state or πR² streaming?
3. How sensitive is the fast path to VLM-prefix and flow latency distributions?
4. What flow refresh rate is sufficient before SEA-RAFT latency becomes the system bottleneck?
5. Can the slow VLM, fast Action Expert and optical-flow estimator be scheduled asynchronously without destabilizing action generation?

The intended evaluation therefore requires matched baselines and ablations, not only successful rollout examples.

---

## Repository layout

```text
src/openpi/models/pi0.py                 FlowPI model and πR² training / sampling
src/openpi/models/pi0_config.py          FlowPI configuration and ablation switches
src/openpi/models/flow_tokenizer.py      Optical-flow tokenizer
src/openpi/models/gemma.py               Flow cross-attention inside the Action Expert
src/openpi/policies/flowpi_runtime.py    Three-device reference runtime
src/openpi/training/sea_raft.py          SEA-RAFT integration
scripts/precompute_flow_cache.py         Offline flow-cache generation
scripts/flowpi_infer.py                  Minimal sequential inference entry point
scripts/fit_vlm_delay.py                 Runtime-delay distribution fitting
```

---

## Acknowledgements

FlowPI is built on the OpenPI / π0.5 codebase and uses SEA-RAFT for optical-flow estimation. The streaming action-generation design is inspired by πR²-style fast/slow policy execution.

Please refer to the upstream projects for their original implementations, licenses, checkpoints and citations.
