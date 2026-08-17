# 下发指令：flowPi 实现（执行者：Kimi-2.7 Code）— v3 最终版

你（Kimi）负责**完整实现** flowPi。本文档是你的唯一任务书；架构与全部设计决策见同目录 `ARCHITECTURE.md`（**先读它**，本文不重复其内容，只规定"做什么、怎么验收"）。总设计出自 `../GLM-5.3.md` + 用户最终修改意见，其给定的整体架构不可更改；若你发现实现层面确实不可行，先在提交说明中记录理由再调整该细节。

## 0. 背景速览

- 仓库根目录 = openpi（π0.5 JAX）。**FlowPi = π0.5 + πR² streaming + Fresh State + 光流快通道**：
  - **πR²**：per-position noise `tau:[B,H]`（不是 scalar time）+ per-position AdaRMS + staircase 训练（80% πR² 样本 / 20% 标准 FM 样本）+ 单 NFE streaming runtime + episode 级 warm-start。**禁止**实现"标准 FM + 每 tick 单步去噪 + time 归零整条重置"——那在数学上不成立。
  - **快通道 = fresh state + flow**：每次 NFE 重新编码当前 state（suffix 中的 state token，`discrete_state_input=False`）与最新光流（SEA-RAFT→Tokenizer→AE 第 7/12/16 层末尾 gated Cross-Attn，**仅 3 套参数，scan 外层**）。
  - **慢通道 = 异步 VLM**：cached prefix + 显式 `d_vlm` delay embedding（训练随机 lag，部署用实测 prefix age）。
- 训练策略：SEA-RAFT 光流**训练前离线缓存**；冻结仅 SigLIP Vision Tower + SEA-RAFT；VLM Transformer / AE / Flow 分支全量微调。
- 初始化（重要修正）：**只有 `flow_gate` zero-init**；Tokenizer 与 Cross-Attn Q/K/V/O 全部正常初始化（双重 zero-init 会让 flow 分支无梯度）。
- 数据：LeRobot v3 aloha（50Hz、14 维、3×480×640 相机、像素内嵌 parquet）。样例 `test_data/adjust_bottle_ep0`。数据中无光流。
- 频率：V1 固定 50Hz 控制（`d=1`/tick，latency-adaptive 可到 `d_max=5`）、`flow_stride Δ=3`（60ms）。不宣称与数据 fps 任意解耦。
- 本机无 GPU、uv 未装：先装环境，一切测试 CPU 可跑。

## 1. 全局约束（每个 Milestone 都适用）

1. **不破坏基线**：`flow=None` 时模型图、数据管道、checkpoint 行为与当前 openpi 完全一致；新参数一律带默认值；仓库现有测试持续通过。
2. **命名规范**：所有新增参数路径必须含小写子串 `flow`（`flow_tokenizer`、`flow_q/flow_kv/flow_out/flow_gate/flow_pre_norm_scale`、`flow_state_proj`、`flow_vlm_delay`）。
3. **代码风格**：`uv run ruff check src scripts` 与 `uv run ruff format --check` 干净；不引入新依赖。
4. **不训练大模型**：CPU 冒烟用 dummy 变体（depth=4，注入层 (1,2)，`d_max=2`）与随机权重 SEA-RAFT。
5. 提交粒度：每个 Milestone 一个或多个自洽 commit，消息 `flowpi(mN): <内容>`。**不要 push，不要改 git config**。
6. 参考：`/tmp/opencode/baseline_{gemma,model,pi0,train}.py`、`pi0_config_parent.py` 为改动前基线快照，可 diff 自查。

## 2. Milestone 0：环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # 若已有 uv 则跳过
uv sync --group dev
uv run python -c "import jax, torch, lerobot; print(jax.devices(), torch.__version__)"
uv run pytest src/openpi/models/pi0_test.py -q
```
验收：全部成功（CPU 机器 jax 报 1 个 cpu device 即正常）。

## 3. Milestone 1：SEA-RAFT 封装

- **改** `SEA-RAFT/core/raft.py`：`forward(..., return_low_res: bool = False)`；True 时返回 dict 增加 `"flow_8x"`（最后一次 refinement 后 1/8 分辨率、未 unpad/upsample）。默认行为零变化。
- **新增** `src/openpi/training/sea_raft.py::SeaRaftFlowExtractor`（ARCHITECTURE §3.1）：M 配置、`ckpt_path=None` 随机权重、`compute(prev,curr)` numpy 接口。注意 raft.py 内 `from update import ...` 绝对导入需 `sys.path.insert(0, <repo>/SEA-RAFT/core)`。
- **测试** `sea_raft_test.py`：随机权重 + `(1,3,3,480,640)` uint8 → `(1,3,2,60,80)` 有限值（`@pytest.mark.slow`）；`return_low_res=False` 时 key 集合与原版一致。

## 4. Milestone 2：数据管道（光流缓存 + per-lag validity + d_vlm 延迟窗口）

细节见 ARCHITECTURE §4.1–4.2。

1. `ComputeFlow`（online transform）：堆叠帧 → 逐 lag RAFT → 归一化 flow + **per-lag validity**（不足历史的 lag → 零 flow + mask False）。
2. `scripts/precompute_flow_cache.py`：写 `{flow_cache_dir}/episode-{ep:06d}/{cam_key}.npy`（`[T,K,2,60,80]` raw fp16）+ `valid.npy`（`[T,K]` bool）+ `meta.json`（K/Δ/分辨率/权重摘要，加载校验）；支持 `--num-workers`。
3. `LoadFlowCache`（cache transform，data_transforms 最前）：按 `episode_index/frame_index` mmap 读 → fp32 → `clamp(f/flow_scale, ±flow_clamp)` → `data["flow"]/data["flow_masks"]`。
4. `DelaySlowImage`：`vlm_delay_max>0` 时图像 keys `delta_timestamps=[(-i)/fps for i in vlm_delay_max..0]`，采 `d_vlm~U{0..max}` 取 `frames[-1-d_vlm]` 为 prefix 图像，`data["vlm_delay"]=d_vlm`；flow/action 仍锚定当前帧。
5. `config.py`：`DataConfig.flow: FlowDataConfig | None`，字段 `enabled=True, mode: Literal["cache","online"]="cache", flow_cache_dir=None, sea_raft_ckpt=None, sea_raft_device="cpu", vlm_delay_max`（读 model.flow 单一来源，不重复存）。`LeRobotAlohaDataConfig` 增 `flow` 字段并按 mode 组装 transforms；`--data.flow.enabled false` 同时关闭缓存与延迟窗口（norm stats 用）。
6. `AlohaInputs/AlohaOutputs`：透传 `flow/flow_masks/vlm_delay/episode_index/frame_index`，相机映射同图像。

**测试** `data_loader_flow_test.py`（slow，CPU 随机权重）：
- online 模式（K=2、Δ=3）取 `test_data` 单样本：flow 三键 `(2,2,60,80)`、per-lag mask 在 episode 开头为 False、图像为单帧、actions horizon=50。
- cache 模式：预计算 1 episode → 读缓存样本与 online 数值 allclose（fp16 容差）；roundtrip 正确。
- `DelaySlowImage`：固定 seed 下帧选择与 `vlm_delay` 一致；`vlm_delay_max=0` 退化为当前帧。
- `flow.enabled=False` 时样本结构与改动前一致。

## 5. Milestone 3：模型核心（πR² + Flow 注入）

按 ARCHITECTURE §2–§3.6：

1. **`model.py`**：`Observation` 加 `flow/flow_masks/vlm_delay` 可选字段（`from_dict/to_dict/inputs_spec` 支持；spec 使 FakeData 自动生成）。
2. **`gemma.py`**：
   - `RMSNorm` 自适应分支支持 cond `[b,D]`（旧，行为不变）与 `[b,s,D]`（per-position）。
   - **flow 参数移到外层 `Module.setup`**：`flow_q [3,8,1024,128]`、`flow_kv [3,2,8,1024,128]`、`flow_out [3,8,128,1024]`（lecun_normal）、`flow_gate [3,1024]`（**zeros，唯一 zero-init**）、`flow_pre_norm_scale [3,1024]`（zeros）。
   - `Block.__call__(self, xs, kv_cache, flow, flow_mask, flow_params, flow_slot, positions, attn_mask, adarms_cond, deterministic=True)`；层末对专家 1 用 `jax.lax.cond(flow_slot >= 0, inject, identity, xs[1])` 注入（RMSNorm→CA→`tanh(flow_gate[slot])⊙` 相加；CA 8 heads×128、float32 logits、`big_neg=-2.3819763e38` 掩码、无 RoPE；**非注入层不执行 CA matmul**）。
   - scan/remat 精确值：`in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, 0, nn.broadcast, nn.broadcast, nn.broadcast)`（序：kv_cache, flow, flow_mask, flow_params, flow_slot, positions, attn_mask, adarms_cond）；`nn.remat(..., static_argnums=(10,))`。`flow_slot: [depth]` int（层 7→0、12→1、16→2，其余 −1；dummy 用 (1,2)）。
3. **`flow_tokenizer.py`**（nnx）：ARCHITECTURE §3.2；**全部正常初始化**（含末层 Linear）；per-lag mask 广播到 80 token。
4. **`pi0.py`**：
   - `embed_suffix(obs, noisy_actions, tau)`：tau 接受 `[B]`（广播，标准路径）或 `[B,H]`（per-position）；time emb/`time_mlp_*` 逐位置 → `adarms_cond=[B,H+1,1024]`（state token 用 `t=0` emb）；flow 启用时前置 `flow_state_proj(state)` state token（ar_mask `[True]+[True]+[False]×(H−1)`）。
   - `embed_prefix`：`+ flow_vlm_delay(vlm_delay)`（zeros-init `Embed(d_max+1, 2048)`）加到 prefix tokens。
   - `compute_loss`：ARCHITECTURE §2.3（逐行混合 80% staircase / 20% 标准；inpaint 段不计 loss；τ jitter 只作用于中间段）。
   - `sample_actions`：保持标准完整去噪（warm-start/对照/回归用）。
   - **新增** `warm_start / denoise_step / refresh_prefix` + `StreamingState`（ARCHITECTURE §3.6；`denoise_step` 内每次重算 `embed_flow(latest)` 与 fresh state token；NFE 步长 `dt=d/(H−2d)`；shift d + append fresh noise；返回 shift 后 `[0,d)`）。
5. **`pi0_config.py`**：`FlowConfig`（字段见 ARCHITECTURE §3.6，含 `d_max/p_standard/tau_jitter/vlm_delay_max`）+ `get_freeze_filter`：flow 启用且无 lora 时返回 `PathRegex(r"PaliGemma/img.*")`。

**测试** `flowpi_test.py`（dummy，CPU 快）：
- 零门控等价（注意对照模型的 `discrete_state_input=False` 一致性）。
- 初始化梯度不变式：step 0 `flow_gate` grad≠0（Tokenizer/QKV 允许 0）；一步更新后 gate≠0 且 Tokenizer/QKV grad≠0。
- 参数预算：scanned 参数树中断言**无** flow 参数；flow CA 仅 `[3,...]` 堆叠；flow 分支总量 ≈13.7M。
- staircase：构造/自相似（减 dt 后左移 d == 原 staircase）/loss mask/[0,d) 无梯度；标准路径 gate=0 时退化基线 loss。
- RMSNorm `[b,s,D]` 通路 shape；`[b,D]` 回归不变。
- `warm_start` + N tick `denoise_step(d=1)`：tau 剖面循环复原、每 tick emit 1 动作、buffer 左移、尾部 fresh noise 统计 ~N(0,1)、`refresh_prefix`/`prefix_age`/delay 索引正确。
- 仓库原 `pi0_test.py`/`model_test.py` 全绿。

## 6. Milestone 4：训练配置与冒烟

- **`config.py`** 新增：
  - `debug_flowpi`：dummy 变体 + `FlowConfig(d_max=2)` + `FakeDataConfig` + `get_freeze_filter()` + 10 steps/batch 2/关 wandb/overwrite。
  - `flowpi_aloha`：`Pi0Config(pi05=True, discrete_state_input=False, flow=FlowConfig())` + `LeRobotAlohaDataConfig(repo_id="<占位>/adjust_bottle", flow=FlowDataConfig(mode="cache", flow_cache_dir="<占位>", sea_raft_ckpt="<占位>"), repack 同 pi05_aloha_pen_uncap, assets=pi05_base/trossen)` + `FlowPiWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")` + freeze filter + 20k/batch 32。
- **`weight_loaders.py`**：`FlowPiWeightLoader`（`missing_regex=r".*(lora|flow).*"`）。
- 可选 `--optimizer-flow-lr`（optax `multi_transform` 按 `.*flow.*` 分组；默认 None）。
- **验收**（CPU）：
```bash
uv run python scripts/compute_norm_stats.py --config-name flowpi_aloha --data.flow.enabled false   # 无数据集则 mock 测试替代并注明
uv run python scripts/precompute_flow_cache.py --config-name flowpi_aloha --data.flow.sea-raft-ckpt ""   # 随机权重 + test_data 单 episode 冒烟
uv run scripts/train.py --config-name debug_flowpi --exp_name smoke
uv run scripts/train.py --config-name debug_flowpi --exp_name smoke --resume true
```
loss 有限；**训练循环零在线 SEA-RAFT 调用**（数据加载器断言）。

## 7. Milestone 5：streaming runtime 与离线回放

1. `src/openpi/policies/flowpi_runtime.py::FlowPiRuntime`（ARCHITECTURE §4.4）：帧 ring buffer、`StreamingState`、后台线程 `refresh_prefix`（atomic 替换，`prefix_age` 每 tick +1，delay 索引 `min(age, vlm_delay_max)`）、每 tick 在线 flow + `denoise_step(d)`；episode 开始 `warm_start`，运行中**永不整条重置**。固定 50Hz（构造时断言 `fast_hz == dataset_fps`，`d` 默认 1）。
2. `scripts/flowpi_infer.py`：`--config-name flowpi_aloha --checkpoint <dir> --dataset test_data/adjust_bottle_ep0 --slow-every-n 10`，输出动作 `.npz` + 计时（RAFT/prefill/NFE）。
3. 测试 `flowpi_runtime_test.py`（dummy + 随机图像序列，20 tick）：动作形状、tau 复原、emit 计数、shift 正确性、prefill 触发计数与 age/delay 索引。

## 8. Milestone 6：文档与收尾

- `flowpi_plan/README_USAGE.md`：环境、norm stats、`precompute_flow_cache`（正式训练前必跑）、train、infer 完整命令（用户替换路径：数据集、SEA-RAFT 权重、缓存目录、pi05_base）；`discrete_state_input=False` 与 pi05_base 的差异说明；K/Δ 与 d_vlm/d_max/slow-every-n 调参表；换 K/Δ 需重算缓存的提醒；PyTorch 路径不支持 flow；V1 固定 50Hz 的理由。
- `uv run ruff check src scripts` + `uv run pytest src scripts -q -m "not slow"` 全绿；`-m slow` 单独跑并记录耗时。
- `flowpi_plan/IMPLEMENTATION_NOTES.md`：各 Milestone 改动文件清单、与 ARCHITECTURE.md 的偏差及原因、已知限制。

## 9. 禁止事项

- 不得实现"标准 scalar-time FM + 每 tick 单步去噪 + time 归零重置"的伪 streaming；必须按 §2 的 per-position πR² 实现。
- 不得在 scanned `Block` 内创建 flow 参数（会堆 18 份）；不得让非注入层执行 CA matmul。
- 不得对 flow 分支使用双重 zero-init（仅 `flow_gate` 为零初始化）。
- 不得把 state 只留在 cached prefix；`denoise_step` 每次必须重新编码 fresh state。
- 不得在 streaming 期间只算一次 flow tokens 闭包捕获；每次 NFE 重算。
- 不得改动 SEA-RAFT 训练代码/模型定义（仅 `return_low_res` 一处增量）；不得解冻 Vision Tower/SEA-RAFT；不得把 SEA-RAFT 纳入 JAX 参数树或 checkpoint；训练循环不得在线调用 SEA-RAFT。
- 不得宣称/实现与数据 fps 任意解耦的控制频率（V1 固定 50Hz）。

## 10. 优先级与顺序

严格 M0→M6（后者依赖前者）。若 M2 的真实数据加载在 CPU 不可行，允许用 parquet 前 N 帧子集跑通测试，其余照常。
