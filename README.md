# FlowPI

FlowPI explores high-frequency visuomotor feedback for Vision-Language-Action policies by augmenting a π0.5-style Action Expert with optical-flow motion cues, fresh robot state, and delay-aware conditioning. The method is designed around a slow semantic pathway and lightweight fast feedback signals while preserving the pretrained VLA backbone as much as possible.

## 📚 Contents

- [Method Overview](#method-overview)
- [Architecture](#architecture)
  - [Base policy](#base-policy)
  - [Optical-flow pathway](#optical-flow-pathway)
  - [Flow tokenizer](#flow-tokenizer)
  - [Gated flow cross-attention](#gated-flow-cross-attention)
  - [Fresh robot state](#fresh-robot-state)
  - [Delay-aware conditioning](#delay-aware-conditioning)
- [πR²-Style Training Objective](#πr²-style-training-objective)
- [Training Data](#training-data)
- [Training Configuration](#training-configuration)
- [Setup](#setup)
- [Training](#training)
- [Inference Partitioning](#inference-partitioning)
- [Repository Structure](#repository-structure)
- [Acknowledgements](#acknowledgements)

<a id="method-overview"></a>
## 🧠 Method Overview

FlowPI extends the π0.5 architecture with four coupled components:

1. **Optical-flow feedback** from SEA-RAFT, represented as compact motion tokens.
2. **Gated cross-attention** that injects motion information into selected Action Expert layers.
3. **Fresh-state conditioning** that re-encodes the latest robot state in the fast suffix.
4. **πR²-style staircase flow matching** with explicit modeling of stale semantic and motion observations.

The resulting policy keeps semantic image-language processing in the VLM prefix while giving the Action Expert direct access to higher-frequency motion and state feedback.

<a id="architecture"></a>
## 🏗️ Architecture

<a id="base-policy"></a>
### Base policy

FlowPI is built on the π0.5 / PaliGemma architecture:

- SigLIP vision encoder for multi-view RGB observations.
- PaliGemma VLM prefix for image-language semantic context.
- A separate Action Expert conditioned through adaRMSNorm.
- Flow-matching action generation over a fixed action horizon.

When `Pi0Config.flow is None`, the model graph remains identical to the baseline π0.5 implementation.

<a id="optical-flow-pathway"></a>
### Optical-flow pathway

For each camera, SEA-RAFT computes motion between the current frame and multiple historical frames. The default configuration uses:

- `num_flow_steps = 2`
- `flow_stride_frames = 3`
- `flow_image_size = (480, 640)`
- low-resolution flow at `H/8 × W/8`

The policy receives normalized 2-D flow fields rather than RGB frame differences. Multiple temporal offsets provide a short motion history while keeping the fast representation compact.

Flow is normalized before entering the policy using:

- `flow_scale = 4.0`
- `flow_clamp = 8.0`

<a id="flow-tokenizer"></a>
### Flow tokenizer

Raw flow fields are converted into motion tokens by a lightweight convolutional tokenizer followed by positional encoding and projection into the Action Expert width.

Default tokenizer configuration:

```text
channels = (32, 64, 128)
MLP hidden = 512
```

Each camera and temporal offset contributes spatial motion features. Invalid historical flow slots are masked instead of replaced with learnable dummy information.

<a id="gated-flow-cross-attention"></a>
### Gated flow cross-attention

Flow tokens are injected directly into selected Action Expert layers through dedicated cross-attention modules.

Default configuration:

```text
injection layers = (7, 12, 16)
attention heads = 8
head dimension = 128
```

At each injection layer, the Action Expert hidden state attends to the flow tokens. The cross-attention residual is controlled by a learned gate:

```text
h <- h + tanh(g) * CrossAttention(h, flow)
```

The gated residual lets the model learn how strongly motion information should modify the pretrained action representation. When no valid flow is available, the flow branch reduces to an identity update.

<a id="fresh-robot-state"></a>
### Fresh robot state

The semantic prefix may represent an older observation than the current control tick. FlowPI therefore optionally inserts a freshly encoded robot-state token into the Action Expert suffix at every denoising step.

This pathway is controlled by:

```text
use_fresh_state = True
```

The state token is treated as conditioning information rather than as an action denoising target.

<a id="delay-aware-conditioning"></a>
### Delay-aware conditioning

FlowPI explicitly distinguishes two forms of observation age.

#### VLM delay

`vlm_delay` measures the age of the semantic VLM prefix relative to the current control tick. A learned delay embedding is added to the Action Expert adaRMS conditioning.

Default maximum:

```text
vlm_delay_max = 10
```

The delay embedding is zero-initialized so introducing the mechanism does not perturb the pretrained Action Expert before training.

#### Flow delay

`flow_delay` represents the age of the available optical-flow signal independently of the semantic-prefix delay.

Default maximum:

```text
flow_delay_max = 2
```

Both delay variables can be sampled from empirical distributions during training. This allows the training distribution to approximate the latency profile of a deployed multi-rate system instead of assuming perfectly synchronous observations.

<a id="πr²-style-training-objective"></a>
## 🧮 πR²-Style Training Objective

FlowPI supports a staircase flow-matching schedule inspired by πR².

For an action horizon `H`, a value `d` is sampled and the horizon is divided into three regions:

- the first `d` actions are treated as already clean,
- the middle segment receives progressively increasing noise,
- the final `d` actions are fully noisy.

The default configuration uses:

```text
d_max = 5
p_standard = 0.2
tau_jitter = 0.01
```

Most training samples therefore use the staircase schedule, while a fraction `p_standard` retain the standard scalar-time flow-matching objective. This mixture preserves coverage of the original π0.5 denoising distribution while training the Action Expert for streaming-style partial action refinement.

The clean prefix positions are excluded from the flow-matching loss for staircase samples, and the remaining loss is renormalized so its scale remains comparable to standard π0.5 training.

<a id="training-data"></a>
## 📦 Training Data

FlowPI currently uses LeRobot v3 datasets together with an offline SEA-RAFT flow cache.

The training pipeline is:

```text
RGB trajectories
    ↓
normalization statistics
    ↓
SEA-RAFT flow precomputation
    ↓
offline flow cache
    ↓
FlowPI policy training
```

The flow cache is computed from the original image geometry and loaded together with RGB observations and robot actions during policy training.

### Image augmentation

Geometric image augmentation is disabled by default for FlowPI:

```text
image_geometric_aug = False
```

The reason is geometric consistency. Optical flow is precomputed offline in the coordinate system of the raw frames; independently cropping or rotating the RGB image would place the RGB observation and cached flow in different coordinate systems.

Photometric augmentation can still be applied because it does not alter the spatial correspondence between RGB and flow.

<a id="training-configuration"></a>
## ⚙️ Training Configuration

The main FlowPI controls are defined in `FlowConfig`.

```python
FlowConfig(
    enabled=True,
    num_flow_steps=2,
    flow_stride_frames=3,
    flow_scale=4.0,
    flow_clamp=8.0,
    flow_image_size=(480, 640),
    tokenizer_channels=(32, 64, 128),
    tokenizer_mlp_hidden=512,
    num_cross_heads=8,
    cross_head_dim=128,
    injection_layers=(7, 12, 16),
    d_max=5,
    p_standard=0.2,
    tau_jitter=0.01,
    vlm_delay_max=10,
    flow_delay_max=2,
    use_fresh_state=True,
    use_delay=True,
    use_flow=True,
    use_pir2=True,
    image_geometric_aug=False,
)
```

### Ablations

The main method components can be disabled independently:

| Option | Function |
| --- | --- |
| `use_flow` | Flow tokenizer and gated cross-attention |
| `use_fresh_state` | Fresh robot-state token in the Action Expert suffix |
| `use_delay` | VLM-delay conditioning |
| `use_pir2` | πR² staircase noise schedule |

These switches gate the forward use of each component without changing the FlowPI parameter layout. This allows the same model architecture and checkpoint format to be used across ablations.

<a id="setup"></a>
## 🛠️ Setup

```bash
uv sync --group dev
git submodule update --init SEA-RAFT
uv run python scripts/setup_sea_raft.py
```

The pinned SEA-RAFT submodule is patched by:

```text
third_party/sea_raft/flowpi_return_low_res.patch
```

The patch exposes low-resolution flow output, avoids unnecessary ImageNet initialization when restoring a checkpoint, and makes large correlation sampling safe. SEA-RAFT weights are external and are not stored in this repository.

<a id="training"></a>
## 🚂 Training

### 1. Compute normalization statistics

```bash
uv run python scripts/compute_norm_stats.py flowpi_aloha \
  --data.repo-id /path/to/lerobot-dataset \
  --data.assets.assets-dir /path/to/assets \
  --data.assets.asset-id flowpi \
  --data.flow.enabled false
```

### 2. Precompute optical flow

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

The dataset, flow cache, SEA-RAFT checkpoint, π0.5 initialization, and FlowPI checkpoints are independent paths and do not need to live inside the repository.

<a id="inference-partitioning"></a>
## 🧩 Inference Partitioning

The intended FlowPI system separates inference into three logical compute paths:

| Path | Responsibility |
| --- | --- |
| **Slow semantic path** | VLM image-language prefix / semantic context |
| **Fast action path** | Action Expert streaming update using fresh state and flow tokens |
| **Optical-flow path** | SEA-RAFT motion estimation |

A natural deployment maps these paths to three GPUs so semantic inference, fast action updates, and optical-flow extraction can be isolated computationally. The README intentionally does not specify a particular runtime scheduler or evaluation implementation.

<a id="repository-structure"></a>
## 🗂️ Repository Structure

```text
src/openpi/models/pi0.py
    FlowPI forward pass, πR² objective, streaming Action Expert logic

src/openpi/models/pi0_config.py
    FlowConfig and model configuration

src/openpi/models/flow_tokenizer.py
    Optical-flow tokenization

src/openpi/models/gemma.py
    Gated flow cross-attention inside the Action Expert

src/openpi/training/sea_raft.py
    SEA-RAFT integration

scripts/precompute_flow_cache.py
    Offline optical-flow cache generation

scripts/fit_vlm_delay.py
    Delay-distribution fitting utilities
```

<a id="acknowledgements"></a>
## 🙏 Acknowledgements

FlowPI builds on the OpenPI / π0.5 codebase, πR²-style streaming action generation, and SEA-RAFT optical-flow estimation.