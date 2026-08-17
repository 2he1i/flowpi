# flowPi 架构方案（Architecture Spec）— v3 最终版

> 本文档是 flowPi 的完整技术方案，基于对当前仓库（openpi / π0.5 JAX）、`SEA-RAFT/`、`test_data/` 的真实代码分析得出。
> 所有文件路径、类名、维度、层号均为已核实的事实，可直接照此实现。
> 实现由 Kimi-2.7 Code 执行，执行指令见 `KIMI_PROMPT.md`。

## 0. 定位

$$\boxed{\text{FlowPi} = \pi0.5 + \pi R^2\ \text{streaming} + \text{Fresh State} + \text{Optical Flow fast channel}}$$

- **πR² 解决**实时 action regeneration（per-position noise + 单 NFE streaming）；
- **FlowPi 进一步解决** slow VLM 无法提供高频视觉运动反馈的问题：在 πR² 的实时闭环框架中增加 SEA-RAFT 光流快通道。
- 研究归因保持单一：验证"显式光流能否改善 π0.5 系的动态闭环能力"，πR² 结构是前提而非变量。

```
                         Slow Channel（异步，低频刷新）
I_{t-d_vlm} + Language ──→ PaliGemma ──→ cached prefix KV（含 d_vlm delay embedding）
                                      │
                                      ▼   每个 NFE 共享 joint self-attn
                              Action Expert（suffix）
                                      ▲
                                      │
                       fresh proprioception s_t（每个 NFE 重新编码）
                                      │
I_{t-Δ}, I_t → SEA-RAFT(frozen) → Flow Tokenizer
                                      │
                                      └─→ AE @ 7/12/16 层末尾 gated Cross-Attn（仅 3 套参数）
                                      ▼
                        πR² per-position Flow Matching（单 NFE）
                                      ▼
                    emit d 个动作 → buffer 左移 d → 尾部补 d 个 fresh noise
```

---

## 1. 仓库事实（已核实，实现时以此为准）

### 1.1 π0.5 JAX 模型（本仓库 = openpi）

| 事实 | 值 | 出处 |
|---|---|---|
| 模型类 | `src/openpi/models/pi0.py::Pi0`，`pi05=True` 走 adaRMS 分支 | pi0.py:67 |
| 双专家 Gemma | `src/openpi/models/gemma.py::Module`，`configs=[gemma_2b, gemma_300m]`，专家 0=PaliGemma(2048d)，专家 1=Action Expert(1024d) | pi0.py:70-80 |
| **AE 层数 L** | **18**（`gemma_300m`: width=1024, depth=18, mlp_dim=4096, heads=8, kv_heads=1, head_dim=256） | gemma.py:69-78 |
| AE 参数命名 | 专家 1 子模块带 `_1` 后缀（`attn_1`, `mlp_1`...）；专家 0 无后缀 | gemma.py:443-450 |
| scanned Block | `nn.scan(Block, variable_axes={"params": 0}, length=depth)`；**Block 内创建的参数会沿 depth 堆叠 18 份** | gemma.py:365-381 |
| scan 签名 | `Block.__call__(xs, kv_cache, positions, attn_mask, adarms_cond, deterministic)`；`xs` 是 carry，`in_axes` 逐参数指定 | gemma.py:293, 369-375 |
| remat | `nn.remat(Block, static_argnums=(5,), ...)`（含 self 计数） | gemma.py:359-364 |
| adaRMS | `RMSNorm` 自适应分支：`nn.Dense(3D, kernel_init=zeros)(cond)` → per-token scale/shift/gate；**cond 目前仅支持 `[b, D]`（整 chunk 共享一个 time）** | gemma.py:126-131 |
| time 约定 | **t=1 是噪声，t=0 是干净动作**；`x_t = t·ε + (1−t)·a`，`u = ε − a`，Euler 步 `x ← x + dt·v`，`dt = −1/N` | pi0.py:196-199, 228-278 |
| pi05 的 state 入口 | **pi05 默认把 state 离散化进 prompt（`discrete_state_input=True`），suffix 中没有 state token**（`if not self.pi05` 才加 state token）；`discrete_state_input=False` 是被支持的配置（见 `pi05_libero`） | pi0.py:151-158, config.py:745 |
| 推理 KV cache | `sample_actions` 先跑 prefix 得 `kv_cache`，再 `while_loop` 内仅跑 suffix | pi0.py:233-278 |
| Observation | `struct.dataclass`，`from_dict` 消费 `data["image"]` 等键；新字段默认 None 即向后兼容 | model.py:81-129 |
| 冻结机制 | `freeze_filter` + `trainable_filter`；frozen 参数转 bfloat16 | config.py:549-552, train.py:102-111 |
| 权重加载 | `CheckpointWeightLoader` 用 `missing_regex=".*lora.*"` 合并新参数 | weight_loaders.py:50-54 |
| 数据管道 | `LeRobotDataset(delta_timestamps=...)` → repack → data_transforms(`AlohaInputs`) → `Normalize`(quantile) → model_transforms | data_loader.py:130-191 |
| aloha 变换 | `AlohaInputs`：CHW→HWC，`cam_high→base_0_rgb` 等映射 | aloha_policy.py:42-87 |

### 1.2 SEA-RAFT（`SEA-RAFT/`，Tartan-M 配置）

| 事实 | 值 |
|---|---|
| 模型 | `SEA-RAFT/core/raft.py::RAFT`（torch）；M 配置 `dim=128, iters=4, radius=4, block_dims=[64,128,256]`（`config/train/Tartan480x640-M.json`） |
| 输入 | uint8 `[B,3,H,W]`（内部 `2x/255−1` 归一化；`InputPadder` pad 到 8 倍数） |
| 输出 | **`flow_8x`（1/8 分辨率、最后一次 refinement 后、upsample 前）目前不返回，需给 raft.py 加 `return_low_res` 暴露**（默认行为不变） |
| 权重 | `core/utils/utils.py::load_ckpt`；用户已微调好自己的权重 |

### 1.3 数据（`test_data/adjust_bottle_ep0`）

- LeRobot **v3.0**，aloha，**fps=50**，1 episode，424 帧；state/action 14 维；3 相机 `3×480×640`，像素内嵌 parquet。
- **数据集中无光流**：训练读 SEA-RAFT 离线缓存（§4），推理在线计算。
- 480/640 恰被 8 整除 → SEA-RAFT 无 padding，flow 网格恒 `60×80`。

### 1.4 环境

无 GPU（测试全 CPU）；uv 未装（`uv sync --group dev`）。`/tmp/opencode/baseline_*.py` 是改动前基线快照，可 diff 对照。

---

## 2. πR² per-position Flow Matching（核心算法修改）

**放弃"标准 π0.5 scalar-time FM + 每 tick 单步 denoise"**（数学上不成立：标量 time 下中间 Euler 结果仍是 noisy latent，不可直接下发）。改为 πR² 的 per-position noise level。

### 2.1 记号与约定

沿用**本仓库** time 约定：`t=1` 纯噪声、`t=0` 干净动作。πR² 论文记号 τ（1=clean, 0=noise）与本文 `t = 1 − τ` 一一对应。H = action_horizon = 50。

### 2.2 Per-position conditioning（取代 scalar time）

```text
time: [B]        →    tau: [B, H]
adarms_cond: [B, D]  →  [B, H+1, D]      # H 个 action token + 1 个 state token
```

- `embed_suffix(obs, noisy_actions, tau)`：`tau` 接受 `[B]`（广播为整 chunk 共享，标准路径）或 `[B,H]`（per-position，πR² 路径）。
- time embedding：对每个位置独立 `posemb_sincos(tau_p, 1024, min_period=4e-3, max_period=4.0)` → `time_mlp_in/out`（结构复用，逐位置应用）→ `[B,H,1024]`。
- **RMSNorm 自适应分支泛化**：cond 支持 `[b,D]`（保持旧行为，checkpoint 兼容）或 `[b,s,D]`（modulation 逐位置：`Dense(3D)` 输出 `[b,s,3D]`，去掉现有 `[:, None, :]`）。专家 1 的 `adarms_cond = [B, H+1, 1024]`；**state token（suffix 第 0 个）的 cond 固定用 `t=0` 的 embedding**（它是条件 token，不是去噪目标）。专家 0 cond 恒 None。

### 2.3 训练：staircase schedule + 混合标准样本

每 batch 元素（行内可混合，用逐行 mask 向量化）：

1. **πR² 样本（概率 `p_pir2 = 0.8`）**：采 `d ~ U{1..d_max}`（默认 `d_max=5`，断言 `d_max < H/2`），构造 staircase（本仓库约定）：
   $$t_p = \begin{cases} 0 & 0 \le p < d & \text{（已执行/in-flight，clean inpaint，不计 loss）}\\ \frac{p-d}{H-2d} & d \le p < H-d & \text{（渐进加噪的未来）}\\ 1 & H-d \le p < H & \text{（fresh noise）} \end{cases}$$
   - `[0,d)`：`x_p = a_p`（GT 干净动作，inpainting 条件），**loss mask = 0**；
   - `[d,H)`：`x_p = t_p ε_p + (1−t_p) a_p`，target `u_p = ε_p − a_p`，**loss mask = 1**（含 fresh-noise 尾部）；
   - τ jitter：对中间段 `t_p ← clip(t_p + U(−j, j), 0, 1)`，`tau_jitter=0.01`；两端（0 与 1）保持精确。
2. **标准 FM 样本（概率 `p_standard = 0.2`，支撑 warm-start 的完整去噪能力）**：`t ~ Beta(1.5,1)·0.999+0.001` 标量共享所有位置，全部位置计 loss、无 inpainting（即现 π0.5 行为）。

Velocity target 与 MSE 形式不变，只是 schedule 与 loss mask 改变。

### 2.4 推理：streaming runtime（取代 `time→0→整条重置`）

状态（`StreamingState`）：
```python
action_buffer: [B, H, D]     # 含 in-flight 段（前端 t=0）
tau:           [B, H]        # 本仓库约定的 staircase
prefix_cache                  # 慢通道 KV
prefix_age: int               # 距上次 prefill 的 tick 数
```

每个控制 tick（50Hz，**每 tick 恰一次 NFE**）：
1. 读取 **fresh state**、**fresh flow**（最新帧对在线算 SEA-RAFT → tokenizer）、cached prefix；
2. 一次 NFE：对所有 `t_p > 0` 的位置 `x_p ← x_p − dt·v_p`、`t_p ← max(t_p − dt, 0)`，其中 **`dt = d/(H−2d)`**；
3. buffer 左移 `d` 位，尾部 append `d` 个 fresh Gaussian（`t=1`）；
4. **下发 shift 后 buffer 的 `[0,d)`**（即本 tick 刚 clean 的 d 个动作）。
   自相似性：所有位置 t 同减 `d/(H−2d)`，左移 d 后严格复原 staircase——这是上式的推导依据。

**Warm-start（仅 episode 开始 / reset 时，正常运行永不整条重置）**：
1. 从纯 Gaussian chunk 出发，用**标准路径**（`sample_actions`，10 步完整去噪，可用 flow）得到第一个干净 chunk `a`；
2. 按 staircase re-noise：`x_p = t_p ε_p + (1−t_p) a_p`（fresh ε）；
3. 先下发 `buffer[0:d)`，随后进入 streaming loop。

---

## 3. 模块详设

### 3.1 SEA-RAFT 封装（新增 `src/openpi/training/sea_raft.py`，torch，冻结）

- `SeaRaftFlowExtractor(ckpt_path, variant="M", device, iters=4).compute(prev,curr)`：`[B,n_cam,3,H,W] uint8 → [B,n_cam,2,H//8,W//8] float32`。
- `raft.py` 最小增量：`forward(..., return_low_res=False)`，True 时返回 dict 增加 `"flow_8x"`。
- 完全冻结、不进 JAX 参数树、不反传；**训练读离线缓存（§4），仅预计算/推理/测试在线跑**。

### 3.2 Flow Tokenizer（新增 `src/openpi/models/flow_tokenizer.py`，nnx）

输入 `obs.flow: dict[key, [B,K,2,60,80]]`（已归一化）+ `obs.flow_masks: dict[key, [B,K]]`（**per-lag**）。

1. 相机间共享 CNN：`Conv(2→32,k3,s2)+GN+SiLU` ×3（60×80→30×40→15×20→8×10，末通道 128）
2. flatten → `[B, n_cam·K·80, 128]`
3. `+` 固定 2D sincos PosEmb（复用 `pi0.posemb_sincos`，行 64 + 列 64，min/max_period=1/1000，按网格缓存）`+` learned 时间步 emb（`Embed(K,128)`，normal(0.02)）`+` learned 相机 emb（`Embed(3,128)`，normal(0.02)）
4. `LayerNorm(128)` → `Linear(128→512)` → SiLU → `Linear(512→1024)`

**初始化（修正：取消双重 zero-init）**：
> **Tokenizer 全部 Linear/Conv 正常初始化；Cross-Attn Q/K/V/O 正常初始化；唯一 zero-init 的是 `flow_gate γ=0`。**

初始 `tanh(0)=0` ⇒ `h'=h` 仍严格等价 π0.5，但 CA 输出非零 ⇒ step 0 即有 `∂L/∂γ ≠ 0`，一步之后梯度进入 Tokenizer/QKV。

**Per-lag validity mask**：`flow_masks[key]: [B,K]`（episode 开头不足 `kΔ` 历史帧的 lag 为 False；相机缺失则该相机全部 lag False）。Token mask = 每个 `(cam,lag)` 的标量广播到其 80 个空间 token → `[B, 480]`。模型可区分"真实静止（flow=0, mask=1）"与"历史不存在（flow=0, mask=0）"。

输出 tokens `[B, 480, 1024]` float32，进入 gemma 前统一 cast `embed_dtype`。

### 3.3 Flow Cross-Attention：3 套参数、scan 外层（修正：参数膨胀）

**禁止**在 scanned Block 内创建 FlowCrossAttn（那会沿 depth 堆 18 份 ≈75M 死参数）。改为：

- 外层 `gemma.Module.setup` 创建 **slot 堆叠参数**（linen raw params，仅 3 份 ≈12.6M）：
  ```text
  flow_q      [3, 8, 1024, 128]   lecun_normal
  flow_kv     [3, 2, 8, 1024, 128] lecun_normal
  flow_out    [3, 8, 128, 1024]   lecun_normal
  flow_gate   [3, 1024]           zeros
  flow_pre_norm_scale [3, 1024]   zeros（RMSNorm 的 1+scale 形式）
  ```
- 构造 `flow_slot: [depth]`（int 数组：全 −1，层 7→0、12→1、16→2），与 `flow_params`（上述堆叠，broadcast）一起传入 scan。
- Block 内（每层）：
  ```python
  def inject(xs1):   # RMSNorm(q 无，见下) → CA → gated add
      h = xs[1]
      hn = h * rmsnorm(h, flow_pre_norm_scale[slot])
      ca = cross_attn(hn, flow, flow_mask, slot_params)      # 8 heads × 128，float32 logits，
                                                             # big_neg=-2.3819763e38 掩码，无 RoPE
      return h + tanh(flow_gate[slot]) * ca
  xs[1] = jax.lax.cond(flow_slot >= 0, inject, lambda x: x, xs[1])
  ```
  `lax.cond` 保证非注入层**不执行 CA matmul**（计算与参数都只在 3 层发生）。
- **Block 新签名**（新增 4 个参数，位于 carry `xs` 与 `kv_cache` 之后）：
  ```python
  __call__(self, xs, kv_cache, flow, flow_mask, flow_params, flow_slot,
           positions, attn_mask, adarms_cond, deterministic=True)
  ```
  - scan `in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, 0, nn.broadcast, nn.broadcast, nn.broadcast)`，按序 `(kv_cache, flow, flow_mask, flow_params, flow_slot, positions, attn_mask, adarms_cond)`
  - `nn.remat(..., static_argnums=(10,))`（0=self … 10=deterministic）
  - 注入位置：Block 末尾（FFN 残差后），作用于专家 1 全部 suffix token（state token + action tokens）。
- Q=AE hidden（norm 后），K=V=flow tokens；head 8×128=1024；`scale=128^-0.5`。
- 注入层 `(7,12,16)` = `round(0.4/0.65/0.9 × 18)`，可配（`injection_layers=None` 自动算）。

### 3.4 Fresh State 快通道（新增，πR² 核心 invariant）

> **不变式：`denoise_step()` 每次调用必须重新读取并编码当前 `observation.state`，不得只从 cached prefix 获取 state。**

- 实现：`flow_state_proj: Linear(32→1024)`（state 先经现有 `PadStatesAndActions` 到 32 维），suffix = `[state_token, action_tokens×H]`，即 `embed_suffix` 在 flow 启用时前置 state token（等价于 pi0 非 pi05 分支的做法，复用其 ar_mask 方案 `[True, True, False×(H−1)]`）。
- flowpi 配置设 **`discrete_state_input=False`**（state 不进 prefix/prompt，只走快通道；`pi05_libero` 已有此先例，微调可适配）。
- State 与 Flow **保持两条独立通路**（state=token 注入 suffix，flow=Cross-Attn），不合并成一个 token 序列。

### 3.5 慢通道 d_vlm delay embedding（新增）

- `flow_vlm_delay: nnx.Embed(vlm_delay_max+1, 2048)`，**zeros init**（纯加性，安全）。
- 训练：`d_vlm ~ U{0..vlm_delay_max}`（默认 `vlm_delay_max=10` ticks），prefix 图像取 `I_{t-d_vlm}`，且
  $$z_{slow} \leftarrow z_{slow} + E_{delay}(d_{vlm})$$
  （在 `embed_prefix` 中加到 prefix token 上，随后才做 prefill 前向，因此 KV cache 天然编码了本次刷新的 age）。
- 部署：用实测 `prefix_age`（clip 到 `vlm_delay_max`）作索引。
- `vlm_delay_max=0` 时等价无延迟（不加载历史帧窗口）。

### 3.6 Pi0 模型接线（`src/openpi/models/pi0.py`）

- `Observation` 新增可选字段（默认 None，全向后兼容）：
  ```python
  flow: dict[key, [B,K,2,60,80]] | None      # 已归一化
  flow_masks: dict[key, [B,K]] | None        # per-lag
  vlm_delay: [B] int | None                  # 训练随机 / 部署实测
  ```
- `FlowConfig`（`Pi0Config.flow: FlowConfig | None = None`；`None` ⇒ 与基线 π0.5 完全一致）：
  ```python
  @dataclasses.dataclass(frozen=True)
  class FlowConfig:
      enabled: bool = True
      # 光流
      num_flow_steps: int = 2        # K
      flow_stride_frames: int = 3    # Δ（数据集帧，50Hz 下 Δt=60ms）
      flow_scale: float = 4.0
      flow_clamp: float = 8.0
      flow_image_size: tuple[int,int] = (480, 640)
      tokenizer_channels: tuple[int,...] = (32, 64, 128)
      tokenizer_mlp_hidden: int = 512
      # Cross-Attn
      num_cross_heads: int = 8
      cross_head_dim: int = 128
      injection_layers: tuple[int,...] | None = None    # None ⇒ (7,12,16)
      # πR²
      d_max: int = 5                # 断言 d_max < action_horizon/2
      p_standard: float = 0.2
      tau_jitter: float = 0.01
      # 慢通道延迟
      vlm_delay_max: int = 10
  ```
- `compute_loss`：按 §2.3 实现（πR² staircase 样本 80% + 标准 FM 样本 20%，逐行混合）。
- `sample_actions`：保留为**标准完整去噪**（scalar time；flow 在 while_loop 外算一次）——用于 warm-start、对照实验与回归测试。
- **流式 API**：
  ```python
  def warm_start(self, rng, observation, num_steps=10) -> StreamingState
      # 标准完整去噪 → staircase re-noise → 返回初始 buffer/tau/prefix（含 d_vlm=age emb）
  def denoise_step(self, state: StreamingState, observation, d: int) -> (actions [B,d,D], StreamingState)
      # §2.4 的单 NFE：fresh state token + fresh flow tokens + cached prefix；shift+append；返回 [0,d)
  def refresh_prefix(self, state, observation) -> StreamingState   # 慢通道刷新，age 清零
  ```
  flow tokens 在 `denoise_step` **每次调用重新计算**（`embed_flow(latest obs)`）——这正是 FlowPi 的意义；不得在 streaming 开始时算一次闭包捕获。

---

## 4. 数据管道与时间对齐

### 4.1 光流缓存（训练；修正：含 per-lag validity）

- **预计算脚本** `scripts/precompute_flow_cache.py --config-name <cfg>`：遍历 episode×帧×相机，对 `k=1..K` 算 `flow_8x=RAFT(I_{t-kΔ}, I_t)`，写：
  - `{flow_cache_dir}/episode-{ep:06d}/{cam_key}.npy`：`[T,K,2,60,80]` **raw float16**（不做 scale/clamp，归一化推迟到加载，改参数无需重算缓存）；
  - `.../valid.npy`：`[T,K]` bool（episode 开头 `t < kΔ` 的 lag 为 False）；
  - `meta.json`：K/Δ/分辨率/权重路径摘要，加载时校验。
- DataLoader 用 `np.load(mmap_mode="r")` 随机访问。
- **加载 transform** `LoadFlowCache`（data_transforms 最前）：按样本 `episode_index/frame_index`（repack 透传）读 raw flow + validity → fp32 → `clamp(f/flow_scale, ±flow_clamp)` → `data["flow"]`、`data["flow_masks"]`（per-lag）。
- **在线 transform** `ComputeFlow`（`mode="online"`）：`delta_timestamps` 加载相机历史帧堆叠，两两与当前帧配对 RAFT（validity 同规则：不足历史 → 零 flow + mask False）。仅用于预计算脚本内部、推理 runtime、测试。
- **缓存模式下 `delta_timestamps` 对 action keys 照旧**（当前锚定 t）。

### 4.2 慢通道延迟窗口（d_vlm 增广）

- `vlm_delay_max>0` 时，图像 keys 的 `delta_timestamps = [(-i)/fps for i in vlm_delay_max..0]`（升序，共 `vlm_delay_max+1` 帧）。
- transform `DelaySlowImage`：采 `d_vlm ~ U{0..max}`，取 `frames[-1-d_vlm]` 作为 prefix 图像，`data["vlm_delay"]=d_vlm`；flow/action 仍锚定当前帧 t。
- 代价说明：每样本多解码 `vlm_delay_max` 帧/相机；`--data.flow.enabled false` 同时关闭延迟窗口（norm stats 等场景）。

### 4.3 频率约定（修正：不宣称任意解耦）

$$\boxed{T_{control}=20\,\mathrm{ms}=50\,\mathrm{Hz}\ \text{（与数据 fps 对齐）},\quad \Delta t_{flow}=\Delta/fps=60\,\mathrm{ms}\ (\Delta=3)}$$

- 光流是**位移**（≈motion×Δt），与物理时间强耦合；V1 固定 50Hz、每 tick `d=1`（`d>1` 仅用于 latency-adaptive 补偿，训练 `d_max=5` 已覆盖）。
- 未来支持变频率时再加真实 Δt embedding 或时间归一化 flow；V1 不做。K/Δ 仍为可配参数（改 K/Δ 需重算缓存）。

### 4.4 推理运行时（新增 `src/openpi/policies/flowpi_runtime.py`）

`FlowPiRuntime`（离线回放与真机共用）：
- 状态：帧 ring buffer（长 `K·Δ+1`）、`StreamingState`、SEA-RAFT extractor。
- 慢通道：后台线程周期 `refresh_prefix`（atomic 引用替换 KV cache），`prefix_age` 每 tick +1 供 delay embedding。
- 快通道：每 tick fresh state + 在线 flow → `denoise_step(state, obs, d)`。
- Episode 开始调 `warm_start`；正常运行 buffer 永不整条重置。
- 交付 `scripts/flowpi_infer.py`：加载 checkpoint，在 `test_data` 上离线回放（`--slow-every-n` 等），输出动作 + 计时（RAFT/prefill/NFE）。

---

## 5. 训练策略（保持不变：VT+SEA-RAFT 冻结，其余全量微调）

| 部分 | 参数量级 | 策略 |
|---|---|---|
| SigLIP Vision Tower（`PaliGemma/img.*`） | ~400M | **冻结** |
| π0.5 VLM Transformer（专家 0） | ~2.2B | **全量微调** |
| Action Expert（专家 1） | ~300M | **全量微调** |
| `action_in/out_proj`、`time_mlp_*`、`flow_state_proj` | ~4M | 可训练 |
| Flow 分支（Tokenizer ~1M + 3-slot CA ~12.6M + gate/norm + delay emb） | ~13.7M | 全量微调（唯一 zero-init 是 gate） |
| SEA-RAFT | ~25M | **冻结**（管道外；训练只读缓存） |

- `freeze_filter = nnx_utils.PathRegex(r"PaliGemma/img.*")`（唯一冻结项；`get_freeze_filter` 在 flow 启用且无 lora 时返回此式）。
- 超参默认：AdamW peak_lr=2.5e-5、cosine decay（warmup 1000, 30k→2.5e-6）、batch 32、20k steps、EMA 0.99、clip 1.0；**可选 `--optimizer-flow-lr`**（flow 分支独立 LR，经 optax `multi_transform` 按 `.*flow.*` 路径分组，默认 None 共享全局 LR）。
- 显存：trainable ~2.9B，可 `--fsdp-devices N`。
- TrainConfig：`flowpi_aloha`（正式）与 `debug_flowpi`（dummy + FakeData，CPU 冒烟）。

---

## 6. 命名规范与 checkpoint 兼容

- **所有新增参数路径必须含小写子串 `flow`**：`flow_tokenizer`、`flow_q/flow_kv/flow_out/flow_gate/flow_pre_norm_scale`（gemma 外层）、`flow_state_proj`、`flow_vlm_delay`。
- `FlowPiWeightLoader(params_path)`：`missing_regex = r".*(lora|flow).*"`——从 `pi05_base` 加载时 VLM+AE 全命中，flow 分支保留新初始化。
- adaRMS 的 `Dense(3D)` kernel 形状不变（只是 cond 变 per-position）→ pi05_base 的 adaRMS 权重**直接兼容**。
- `flow=None` 时模型图与基线 π0.5 完全一致；旧 checkpoint 与 flowPi 双向可加载（intersect/merge 规则同前）。
- `discrete_state_input` 差异：flowpi 设 False（pi05_base 用 True 训练），属微调可适配范围（先例 `pi05_libero`），在 README 注明。
- PyTorch 训练路径（models_pytorch）v1 不支持 flow。

---

## 7. 测试与验收（全部 CPU 可跑）

1. **零门控等价**：gate 全零时，flowPi（任意 flow/d_vlm/vlm_delay 输入）输出 == 基线 π0.5 输出（dummy 变体逐元素比对；注意基线需同样 `discrete_state_input=False` 且给 state token 才严格可比——对照模型用 `pi05=True` + 手动加 state token 的等价构造，或对 dummy 直接比对 `flow=None` 分支）。
2. **初始化梯度不变式（修正）**：step 0：`flow_gate` 梯度非零（CA 输出非零），Tokenizer/QKV 梯度**允许为零**；optimizer 一步后 gate≠0 且 Tokenizer/QKV 梯度非零。
3. **参数预算**：flow CA 参数只存在 3-slot 堆叠（形状 `[3,...]`），scanned 参数树中**无** flow 参数；flow 分支总量 ≈13.7M。
4. **staircase**：构造正确（三段）；`t` 同减 `d/(H−2d)` 后左移 d 位与原 staircase 严格相等（自相似）；loss mask 排除 `[0,d)`；`p_standard` 路径在 gate=0 时退化为基线 loss。
5. **per-position cond**：RMSNorm cond `[b,s,D]` 通路 shape 正确；`[b,D]` 旧通路行为不变（回归）。
6. **streaming**：warm-start 后逐 tick：tau 剖面单调且循环复原；每 tick 恰 emit `d` 个动作；buffer 左移正确、尾部为 fresh noise（与 N(0,1) 统计一致）；`refresh_prefix` 触发计数与 `prefix_age`/delay 索引正确。
7. **数据**：缓存 roundtrip（fp16 写→读→归一化）与 online 模式 allclose（fp16 容差）；episode 开头 per-lag validity 正确；`DelaySlowImage` 帧选择与 `vlm_delay` 值正确。
8. **冒烟**：`debug_flowpi` CPU 10 步训练 loss 有限、ckpt 保存/恢复；训练循环零在线 SEA-RAFT 调用（断言）。
9. SEA-RAFT wrapper：随机权重 + 随机 480×640 → `[2,60,80]` 有限值。

---

## 8. 关键设计决策记录

| 决策 | 内容 | 理由 |
|---|---|---|
| πR² 而非"标准 FM+单步下发" | per-position τ + staircase 训练 + 单 NFE streaming + warm-start（§2） | 标量 time 下中间 Euler 结果不可下发；πR² 的实时性依赖 Diffusion Forcing 结构，训练/推理必须匹配 |
| Fresh State 快通道 | state token 每 NFE 重新编码进 suffix；`discrete_state_input=False`；与 flow 分路 | πR² 核心：fresh proprioception 每次调用可见；光流不能替代本体反馈 |
| Flow 每次 NFE 刷新 | `denoise_step` 内重算 `embed_flow(latest)` | FlowPi 的立身之本：高频视觉运动反馈 |
| 慢通道显式延迟 | 训练随机 d_vlm + E_delay；部署用实测 prefix_age | πR² 做法：不仅延迟，还要告诉模型延迟了多少 |
| 初始化 | Tokenizer/QKV 正常 init；仅 gate zero-init | 双重 zero-init 会令 flow 分支无梯度（∂L/∂γ ∝ CA=0 且 ∂L/∂CA ∝ tanh(γ)=0） |
| CA 参数位置 | 3-slot 堆叠放 scan 外层 + `flow_slot` + `lax.cond` | scanned Block 内建参数会堆 18 份（75M vs 12.6M）；非注入层不执行 matmul |
| per-lag flow mask | `[B,K]` validity → 广播到 80 token | 区分"静止"与"历史不存在" |
| 频率 | V1 固定 50Hz、Δ=3（60ms） | flow=motion×Δt，与物理时间耦合；不宣称任意解耦 |
| SEA-RAFT 缓存 | 训练前离线缓存 raw fp16（mmap）；推理在线 | 冻结⇒确定函数；raw 存储使归一化参数可后调 |
| 冻结/微调 | VT+SEA-RAFT 冻结；VLM Transformer/AE/Flow 全量微调 | 用户指定；πR² 允许从已有 flow policy 微调 |
| 注入层 | (7,12,16)，AE suffix 全部 token（state+actions）作为 Q | 0.4/0.65/0.9×18；state token 参与 CA 无害且信息一致 |
