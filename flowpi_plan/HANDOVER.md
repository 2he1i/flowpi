# FlowPi — Agent Handover Document

> **交接人**: DeepSeek-V4-Flash (openPI 实例)
> **接任人**: 下一个任意模型（无会话上下文）
> **项目名称**: FlowPi = π0.5 + πR² streaming + Fresh State + Optical Flow 快通道
> **仓库根目录**: `flowPi/`（openPI 分支，基于 `openpi` π0.5 JAX 实现）
> **架构方案**: `flowpi_plan/ARCHITECTURE.md`（完整技术方案，354 行）
> **执行指令**: `flowpi_plan/KIMI_PROMPT.md`（原始下发任务书，127 行）

---

## 0. 项目本质一句话

FlowPi 在 πR²（per-position noise + 单 NFE streaming）的实时闭环框架中增加 SEA-RAFT 光流快通道，解决 slow VLM 无法提供高频视觉运动反馈的问题。

```
                    Slow Channel (异步, 低频刷新)
I_{t-d_vlm}+Language ──→ PaliGemma ──→ cached prefix KV (含 delay embedding)
                                        │
                                        ▼  每个 NFE 共享 joint self-attn
                                Action Expert (suffix)   ←  18 层 AE
                                        ▲
                                        │
                         fresh proprioception s_t (每个 NFE 重新编码)
                                        │
 I_{t-Δ}, I_t → SEA-RAFT(frozen) → Flow Tokenizer → AE @ 7/12/16 层 CA
                                        │
                                        └── 3-slot gated Cross-Attn (scan 外层)
                                        ▼
                          πR² per-position Flow Matching (单 NFE)
                                        ▼
                      emit d 个动作 → buffer 左移 d → 尾部补 d 个 fresh noise
```

---

## 1. 仓库事实（已核实）

| 条目 | 值 |
|------|-----|
| 根目录 | `flowPi/` |
| 模型类 | `src/openpi/models/pi0.py::Pi0`，`pi05=True` 走 adaRMS |
| 双专家 | `src/openpi/models/gemma.py::Module`，专家0=PaliGemma, 专家1=Action Expert (depth=18) |
| AE 参数命名 | 专家1带`_1`后缀, 专家0无后缀 |
| scanned Block | `nn.scan(Block, variable_axes={"params": 0}, length=depth)` |
| remat | `nn.remat(Block, static_argnums=(10,), ...)` |
| time 约定 | t=1 是噪声, t=0 是干净动作; `x_t = t·ε + (1−t)·a`, Euler `x←x + dt·v`, `dt = −1/N` |
| pi05 state | pi05 默认离散化进 prompt（`discrete_state_input=True`）; flowpi 设 False |
| 推理 KV | 先跑 prefix 得 `kv_cache`, 再 `while_loop` 内仅跑 suffix |
| 数据 | LeRobot **v3.0**, aloha, **fps=50**, 1 个 episode, 424 帧, 14 维 state/action, 3 相机 480×640, 像素内嵌 parquet |
| 光流分辨率 | 480/640 恰被 8 整除 → flow 网格恒 `60×80` |
| SEA-RAFT | `SEA-RAFT/core/raft.py::RAFT`, M 配置 (dim=128, iters=4) |
| 环境 | 2×4090 (CUDA 13.2, Driver 595.58), `.venv` 已就绪, `uv sync --group dev` |
| 已安装包 | JAX (GPU), PyTorch (cu126), Flax, lerobot(git), 所有 openpi 依赖 |

---

## 2. 总体状态

| 里程碑 | 描述 | Git Commit | 测试状态 | 说明 |
|--------|------|-----------|---------|------|
| ✅ **M0** | 环境安装与基线验证 | 原有仓库已提供 | ✅ | uv sync, JAX GPU 可用 |
| ✅ **M1** | SEA-RAFT 封装 | `2c674df` | ✅ 3/3 通过 | `raft.py` 加 `return_low_res`; `sea_raft.py` 封装备 PyTorch numpy 接口 |
| ✅ **M2** | 数据管道 | `91c86c3` | ✅ 7/7 通过 | v3.0 parquet 读取器; 在线/cache 光流 transform; 延迟窗口; 预计算脚本 |
| ✅ **M3** | 模型核心 | `032290d` | ✅ 8/8 通过 | gemma RMSNorm per-position cond; Block 新签名+flow CA; flow_tokenizer; πR² loss; streaming API |
| ✅ **M4** | 训练配置与冒烟 | `ebbf97c` | ✅ (已提交) | debug_flowpi/flowpi_aloha 配置; FlowPiWeightLoader; 冻结策略 (仅 VT frozen) |
| ❌ **M5** | Streaming Runtime | 磁盘，未提交 | ❌ | flowpi_runtime.py (252行) + 测试 (100行) + infer 脚本 (114行) 已写但测试失败 |
| ❌ **M6** | 文档与收尾 | 未开始 | ❌ | README_USAGE.md, IMPLEMENTATION_NOTES.md, ruff 全绿, pytest 全部通过 |

---

## 3. 每个里程碑的详细说明

### M1 — SEA-RAFT 封装 (`src/openpi/training/sea_raft.py`)

**改动文件**:
- `SEA-RAFT/core/raft.py`: `forward()` 加 `return_low_res: bool = False`; True 时返回 dict 增加 `"flow_8x"` (最后一次 refinement 后 1/8 分辨率、未 upsample 的 flow)
- `src/openpi/training/sea_raft.py`: `SeaRaftFlowExtractor(ckpt_path, variant, device, iters)` → `compute(prev, curr)` numpy 接口
  - 输入 ` [B, n_cam, 3, H, W] uint8`
  - 输出 `[B, n_cam, 2, H//8, W//8] float32`
  - 用 `sys.path.insert(0, SEA-RAFT/core)` 处理其绝对导入
  - `ckpt_path=None` 时用 `torch.manual_seed(0)` 确保确定性的随机权重（多进程可比）
- `src/openpi/training/sea_raft_test.py` (3 tests, `@pytest.mark.slow`): 形状/有限性/默认键集

### M2 — 数据管道

**关键文件**:
- `src/openpi/training/lerobot_v3_dataset.py`: `LeRobotV3ParquetDataset` — 轻量读 v3.0 数据集 (pyarrow + PIL)，支持 `delta_timestamps` 堆叠 + `{key}_is_pad`
- `src/openpi/transforms.py`: 新增 3 个 transform + 2 个 helper
  - `compute_image_frame_offsets(K, Δ, vlm_delay_max)` → `(-max, ..., 0)` 升序
  - `normalize_flow(flow, scale, clamp)`
  - `LoadFlowCache`: 读离线缓存 (mmap) → `data["flow"]`, `data["flow_masks"]`
  - `ComputeFlow`: 在线 SEA-RAFT → 归一化 flow
  - `DelaySlowImage`: 采 `d_vlm ~ U{0..max}` → 选延迟帧 + `data["vlm_delay"]`
- `src/openpi/training/config.py`: `FlowDataConfig` + `DataConfig.flow` + LeRobotAlohaDataConfig 组装
- `src/openpi/training/data_loader.py`: 支持本地 v3 数据集; delta_timestamps 注入相机历史帧
- `src/openpi/policies/aloha_policy.py`: AlohaInputs 透传 flow/flow_masks/vlm_delay + 相机重映射
- `scripts/precompute_flow_cache.py`: 预计算脚本, 多 worker, `_process_episode`

**测试** `data_loader_flow_test.py` (4 tests, slow):
- online 模式: flow 形状、mask 开头 False、单帧图像
- cache roundtrip: 缓存 vs online 数值一致 (atol=0.1)
- DelaySlowImage 确定性
- `flow.enabled=False` 时基线结构不变

### M3 — 模型核心 (最重要的里程碑)

**改动文件**:

#### `src/openpi/models/gemma.py`
- `RMSNorm`: 自适应分支 cond 支持 `[b,D]` (旧) 和 `[b,s,D]` (per-position); ndim==2 vs ==3 分支
- `FlowGeom` dataclass: `num_heads, head_dim, injection_layers`
- `Block`: 加 `flow_enabled` 字段; 新 `__call__` 签名 (加 flow/flow_mask/flow_params/flow_slot); 层末以 `jax.lax.cond` 注入 flow Cross-Attn (仅注入层执行 CA matmul)
- `_flow_cross_attn`: `q = einsum("bsd,ndh->bsnh")`, `k/v = einsum("bfd,ndh->bfnh")`, `logits = einsum("bsnh,bfnh->bnsf")`, mask with `big_neg`, softmax, out proj
- `_flow_rmsnorm`: 无参数的 RMSNorm (用预计算 `flow_pre_norm_scale` 乘)
- `Module`: 加 `flow_geom` 字段; `setup` 内创建 3-slot 堆叠参数 (在 scan 外层, 不在 Block 内); `__call__` 支持 `flow`, `flow_mask` kwonly; `_make_flow_slot()` 从 injection_layers 构建 int32 数组

#### `src/openpi/models/flow_tokenizer.py`
- `FlowTokenizer(nnx.Module)`: 3 层 Conv(2→32, stride2) + GN? → 实际用 nnx.Conv + silu + norm + MLP
- 每相机 `[B, K, 2, 60, 80]` → 网格; 添加固定 2D sincos pos_emb + learned lag_emb + learned cam_emb
- 输出 `[B, 480, width]` tokens + `[B, 480]` per-lag token mask

#### `src/openpi/models/model.py`
- `Observation`: 加 `flow`, `flow_masks`, `vlm_delay` 可选字段 (默认 None, 全向后兼容)

#### `src/openpi/models/pi0.py`
- `__init__`: 条件创建 `flow_tokenizer`, `flow_state_proj`, `flow_vlm_delay` (zeros init); 传递 `flow_geom` 给 gemma Module
- `embed_prefix`: 图像 tokens 加 `+ flow_vlm_delay(vlm_delay)` (延迟嵌入)
- `embed_suffix`: `timestep` 支持 `[B]` (标准) 或 `[B,H]` (per-position πR²); per-position 时 `adarms_cond = [B, H+1, 1024]`; flowpi 启用时 +state token (discrete_state_input=False)
- `embed_flow`: 调用 `flow_tokenizer` → `(flow_tokens, flow_token_mask)` 或 None
- `compute_loss`: πR² 混合训练: 80% staircase + 20% 标准 FM; staircase = `[0,0,...,t,...,1,1]` 三段; inpaint 段不计 loss; per-position 掩码
- `sample_actions`: 保持标准完整去噪, 带 flow 支持
- `StreamingState` dataclass: `action_buffer`, `tau`, `kv_cache`, `prefix_mask`, `prefix_age`
- `warm_start`: 完整去噪 → staircase re-noise → 初始 buffer/tau
- `denoise_step`: 每 tick 单 NFE: fresh state + flow → Euler dt=d/(H-2d) → shift d → append fresh noise
- `refresh_prefix`: 慢通道刷新, age 清零
- `make_staircase_tau(H, d)`: `[0]*d, [(p-d)/(H-2d) for p in d..H-d], [1]*d`

#### `src/openpi/models/pi0_config.py`
- `FlowConfig` dataclass: 所有 flow/πR²/delay 参数
- `Pi0Config` 加 `flow: FlowConfig | None = None`; `__post_init__` 断言 `d_max < H/2`
- `inputs_spec`: flow 启用时生成 flow/flow_masks/vlm_delay spec
- `get_freeze_filter`: flow 启用且无 lora 时冻结 `PaliGemma/img.*`

#### 测试 `flowpi_test.py` (8 tests, slow):
1. ✅ `test_zero_gate_equivalence`: 门控全零时 flowPi 输出 == 基线输出 (atol=1e-4)
2. ✅ `test_zero_gate_equivalence_no_flow_input`: flow=None 观测时同样等价
3. ✅ `test_init_gradient_invariants`: step0 flow_gate grad≠0, tokenizer/QKV grad=0; 一步后全部 grad≠0
4. ✅ `test_flow_param_budget`: 参数仅 2-slot 堆叠, 不在 scan layers 内
5. ✅ `test_staircase_construction`: H=12,d=2 阶梯正确
6. ✅ `test_staircase_self_similarity`: d=1 时自相似 (减 dt 左移 = 原剖面)
7. ✅ `test_per_position_rmsnorm_shapes`: [b,s,d] cond 通路 shape 正确, 常数 cond 等价 [b,d]
8. ✅ `test_streaming_runtime`: warm_start+H-2 步 denoise: tau 循环复原, 每步 emit, tail fresh noise, prefix_age 递增/刷新正确

**关键设计决策**:
- **仅 `flow_gate` zero-init**: Tokenizer/QKV 全部正常初始化 (双重 zero-init 让 flow 分支无梯度)
- **3-slot 堆叠**放 scan 外层: 不在 Block 内建参数 (会沿 depth 堆 18 份)
- **`lax.cond`**保证非注入层不执行 CA matmul
- **Fresh State 与 Flow 分路**: state→suffix token, flow→Cross-Attn

### M4 — 训练配置与冒烟

**改动文件**:
- `src/openpi/training/config.py`: 新增 `debug_flowpi`, `flowpi_aloha` 配置
- `src/openpi/training/weight_loaders.py`: `FlowPiWeightLoader(missing_regex=".*(lora|flow).*")`
- 提交消息: `module-level init constants fix for graphdef determinism`

**配置详情**:
- `debug_flowpi`: dummy variant + FlowConfig(d_max=2) + FakeData + 10 steps + batch 2 + 关 wandb
- `flowpi_aloha`: `pi05=True, discrete_state_input=False, flow=FlowConfig()` + LeRobotAlohaDataConfig(flow=...) + FlowPiWeightLoader + freeze filter

---

## 4. M5 — Streaming Runtime (未完成)

### 文件 (磁盘上, 未提交)

#### `src/openpi/policies/flowpi_runtime.py`
- `_FrameRingBuffer`: CHW uint8 环形缓冲; `push(cam, frame)`, `advance()`, `get(offset)`
- `FlowPiRuntime`: `warm_start(obs)`, `tick(obs)`, `refresh_prefix(obs)`, `emit()`
- `_ingest_frame(obs)`: 图像 CHW uint8 → ring buffer; `_compute_flow()`: SEA-RAFT per-lag
- `_compute_flow`: 处理 invalid lags (episode 开头) → zero flow
- 后台 `_prefix_age` (每 tick +1 → delay embedding); `refresh_prefix` 通过 atomic 引用替换 KV cache

#### `scripts/flowpi_infer.py`
- argparse CLI: `--config-name`, `--checkpoint`, `--dataset`, `--slow-every-n`, `--max-frames`
- 加载模型 → 构建 runtime → 遍历数据集帧 → 计时 (raft/prefill/NFE) → 保存 `.npz`

#### `src/openpi/policies/flowpi_runtime_test.py`
- 创建 20 帧 dummy 观测 (480×640, 随机图像+state+token)
- 调用 `warm_start` + `refresh_prefix` + 19 步 `tick`
- 验证: emit 形状, finite, 每步 prefix_age 递增, refresh 后清零

### 当前错误

```
ValueError: axis 3 is out of bounds for array of dimension 3
```

`_prefix_forward` 内部 `preprocess_observation` 调用链中的 transposed — 根因是 `FlowPiRuntime` 的 `_ingest_frame` 和 `_prefix_forward` 中使用 numpy/JAX 混合操作时, 图像数组在 numpy/JAX 转换中出现形状错位导致的一个 `transpose` 轴越界。我尝试的多项修复（`_prefix_forward` 加预处理、`_compute_flow` 加 invalid lag 处理、shape unpacking 修正）均未完全解决——很可能是运行时创建观测(使用 jax 数组) 与 `preprocess_observation` (期望特定张量约定) 之间还有一两处数据布局不匹配。

### 建议的 M5 修复路径

1. 先在 `_prefix_forward` 内调用 `preprocess_observation` (已完成)
2. `tick` 中的 `_ingest_frame` 确保数据始终为 numpy HWC uint8 (当前代码已做)
3. 问题可能在于 `preprocess_observation` 接受的图像是 `[B,H,W,C]` jax array(来自测试构造的 Observation.images), 但 `_ingest_frame` 使用 `np.asarray(obs.images[cam])[0]` 取 `[0]` 后被 JAX 自动追踪为 JAX 操作 → 触发 `transpose` 错误。需要将图像先 `.block_until_ready()` 或 `.copy()` 成 numpy array
4. 最简单的修复: 在 `_ingest_frame` 中将 jax array 转 numpy: `np.asarray(jax.device_get(obs.images[cam]))[0]`

---

## 5. M6 — 文档与收尾 (未开始)

需要完成:
- `flowpi_plan/README_USAGE.md`: 环境、norm stats、precompute_flow_cache、训练、推理完整命令
- `flowpi_plan/IMPLEMENTATION_NOTES.md`: 各 Milestone 改动文件清单、与 ARCHITECTURE 偏差、已知限制
- `uv run ruff check src scripts` 干净
- `uv run ruff format --check` 干净
- `uv run pytest src scripts -q -m "not slow"` 全绿
- `-m slow` 单独跑并记录耗时
- 仓库基线测试保持通过

---

## 6. 已知问题与偏差

1. **M5 runtime 测试未通过** — 见 §4
2. **lerobot v3.0 数据集** — 安装的 lerobot(0.1.0, codebase v2.1) 不支持 v3.0，所以写了 `lerobot_v3_dataset.py` shim。正式训练时用户需要升级 lerobot 或继续使用该 shim
3. **`normalize_flow` 位置** — 实现在 `transforms.py`，但运行时也导入它，属无副作用复用
4. **虚拟 SigLIP 不可用** — `paligemma_variant="dummy"` 只影响 LLM，vision tower 始终是 "So400m/14" (真模型)。所以 dummy 配置仍需 GPU 去跑 vision transformer。CPU 冒烟通过但慢
5. **配置名拼写**: flowpi 相关文件（runtime 等）暂未经过 ruff 格式化

---

## 7. 关键命令速查

```bash
# 运行 M3 模型测试 (8个, ~6-7 min on GPU)
CUDA_VISIBLE_DEVICES=0 uv run python -m pytest src/openpi/models/flowpi_test.py -q

# 运行数据管道测试 (4个, ~1.5 min)
CUDA_VISIBLE_DEVICES=0 uv run python -m pytest src/openpi/training/data_loader_flow_test.py -q

# 运行 SEA-RAFT 测试 (3个, ~25s)
CUDA_VISIBLE_DEVICES=0 uv run python -m pytest src/openpi/training/sea_raft_test.py -q

# 基线回归测试 (8个, ~1-2 min per test on GPU)
CUDA_VISIBLE_DEVICES=0 uv run python -m pytest src/openpi/models/pi0_test.py -q

# 训练冒烟
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --config-name debug_flowpi --exp_name smoke

# 预计算光流缓存 (前20帧冒烟)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/precompute_flow_cache.py --config-name flowpi_aloha --data.flow.sea-raft-ckpt "" --max-frames 20

# Ruff 检查
uv run ruff check src scripts

# 全部测试 (非慢速)
uv run python -m pytest src scripts -q -m "not slow"
```

---

## 8. 核心文件索引

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/openpi/models/gemma.py` | ~540 | 双专家 Transformer + flow Cross-Attn 注入 |
| `src/openpi/models/pi0.py` | ~620 | π0.5 模型 + πR² loss + streaming API |
| `src/openpi/models/flow_tokenizer.py` | ~130 | CNN + embedding → flow tokens |
| `src/openpi/models/pi0_config.py` | ~140 | FlowConfig + inputs_spec + freeze_filter |
| `src/openpi/models/model.py` | ~350 | Observation(flow/vlm_delay) 基类 |
| `src/openpi/transforms.py` | ~480 | 数据转换 (flow cache/online/delay) |
| `src/openpi/training/sea_raft.py` | ~110 | SEA-RAFT 封装 |
| `src/openpi/training/lerobot_v3_dataset.py` | ~180 | v3.0 数据集读取器 |
| `src/openpi/training/data_loader.py` | ~580 | DataLoader + v3/v2 兼容 |
| `src/openpi/training/config.py` | ~1000 | 所有训练配置 (含 flowpi) |
| `src/openpi/training/weight_loaders.py` | ~104 | FlowPiWeightLoader |
| `scripts/precompute_flow_cache.py` | ~160 | 光流缓存预计算 |
| `src/openpi/policies/flowpi_runtime.py` | ~268 | **(uncommitted)** Streaming runtime |
| `scripts/flowpi_infer.py` | ~115 | **(uncommitted)** 离线回放 |
| `flowpi_plan/ARCHITECTURE.md` | 354 | 架构方案 |
| `flowpi_plan/KIMI_PROMPT.md` | 127 | 原始任务书 |

---

## 9. Git 历史

```
ebbf97c flowpi(m4): training configs + FlowPiWeightLoader + graphdef fix
032290d flowpi(m3): πR² per-position FM + flow cross-attention + streaming API + 8 tests
91c86c3 flowpi(m2): data pipeline fixes + deterministic random SEA-RAFT
6379c32 flowpi(m2): track SEA-RAFT as untracked vendor dir
fa76e45 flowpi(m2): flow data pipeline (v3 reader, transforms, precompute script)
2c674df flowpi(m1): SEA-RAFT wrapper with 1/8-res flow extraction
```

---

**交接结束。祝你顺利！**