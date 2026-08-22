# FlowPI 离线训练命令

先停止之前卡在 GCS 下载阶段的训练进程，然后执行：

```bash
cd /inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/flowpi && \
OPENPI_DATA_HOME=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/pi05/offline_assets/openpi_cache \
FLOWPI_WEIGHT_LOADER_PATH=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/pi05/offline_assets/openpi_cache/openpi-assets/checkpoints/pi05_base/params \
JAX_PLATFORMS=cuda \
FLOWPI_XLA_MEM_FRACTION=0.98 \
FLOWPI_SAVE_INTERVAL=2000 \
FLOWPI_SEA_RAFT_CKPT=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/SEA-RAFT-FT/SEA-RAFT/checkpoints/24000_robot-ft-M-4gpu-shadow-15k-to-25k_robot-ft-M-4gpu-shadow-15k-to-25k-20260822-053345.pth \
FLOWPI_TRAIN_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
FLOWPI_TRAIN_EXPECTED_GPUS=8 \
FLOWPI_GLOBAL_BATCH=128 \
FLOWPI_WANDB_ENABLED=0 \
bash scripts/run_flowpi_cache_and_train.sh train
```

该命令使用共享盘中的离线 `pi05_base` 权重和 tokenizer，不会重新生成 Flow cache，也不需要 H 实例联网。

注意：每个续行符 `\` 必须是该行最后一个字符，后面不能有空格；路径中没有空格。

## 断线续训

训练默认启用 `resume`。同一个实验名 `flowpi_8xh200` 重新执行上面的命令时，会自动从 checkpoint 目录中最新的完整 checkpoint 继续，并恢复模型参数、AdamW 状态、EMA、学习率调度器和训练步数。

当前配置每 2000 steps 保存一次；Step 2000 之前如果任务被杀掉，只能从 base checkpoint 重新开始。若服务器稳定性较差，可以改成每 1000 steps 保存一次：

```bash
FLOWPI_SAVE_INTERVAL=1000 \
```

如果需要从指定 checkpoint 恢复，例如 Step 5000，在命令中增加：

```bash
FLOWPI_RESUME_STEP=5000 \
```

不要设置 `FLOWPI_OVERWRITE=1`，否则会清理现有 checkpoint 目录。

## tmux 训练监控

在 H 平台执行下面命令，会创建或重新连接到 `flowpi-monitor` tmux 会话。监控程序只读共享日志，不会影响训练；会显示 Step、Loss、grad norm、Flow delay、实际速度、吞吐、ETA、GPU 利用率/显存和 checkpoint 状态。

```bash
cd /inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/flowpi && \
tmux new-session -A -s flowpi-monitor \
  ".venv/bin/python scripts/monitor_flowpi_training.py --run-root logs/flowpi_cache_train/flowpi_8xh200 --refresh 30"
```

退出 tmux 但保持监控运行：`Ctrl-b`，然后按 `d`。重新进入：

```bash
tmux attach -t flowpi-monitor
```

## Flow-required 对比训练（保留原 2k checkpoint）

原实验 `flowpi_8xh200` 和 `checkpoints/flowpi_aloha/flowpi_8xh200/2000` 不会被覆盖。下面的
launcher 使用独立的实验名、日志目录和 checkpoint 目录：

```bash
cd /inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/flowpi && \
OPENPI_DATA_HOME=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/pi05/offline_assets/openpi_cache \
FLOWPI_WEIGHT_LOADER_PATH=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/pi05/offline_assets/openpi_cache/openpi-assets/checkpoints/pi05_base/params \
JAX_PLATFORMS=cuda \
FLOWPI_XLA_MEM_FRACTION=0.98 \
FLOWPI_SAVE_INTERVAL=2000 \
FLOWPI_EXP_NAME=flowpi_8xh200_flow_required \
FLOWPI_LOG_ROOT=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/flowpi/logs/flowpi_cache_train \
FLOWPI_FLOW_REQUIRED_PROB=0.5 \
FLOWPI_FLOW_REQUIRED_VLM_DELAY_MIN=5 \
FLOWPI_FLOW_GATE_INIT=0.01 \
FLOWPI_RESUME=0 \
FLOWPI_SEA_RAFT_CKPT=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/SEA-RAFT-FT/SEA-RAFT/checkpoints/24000_robot-ft-M-4gpu-shadow-15k-to-25k_robot-ft-M-4gpu-shadow-15k-to-25k-20260822-053345.pth \
FLOWPI_TRAIN_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
FLOWPI_TRAIN_EXPECTED_GPUS=8 \
FLOWPI_GLOBAL_BATCH=128 \
FLOWPI_WANDB_ENABLED=0 \
bash scripts/train_flowpi_flow_required_8xh200.sh
```

这次仍直接读取已有 SEA-RAFT cache，不重新 caching。新的日志位于
`logs/flowpi_cache_train/flowpi_8xh200_flow_required/`，新的 checkpoint 位于
`checkpoints/flowpi_aloha/flowpi_8xh200_flow_required/`。比较时重点看每个 run 的
`mean_vlm_delay`、`flow_gate_tanh_abs_layer*`、`flow_ca_residual_ratio_layer*`、loss，以及
`frac_flow_required`、`loss_flow_required`、`loss_normal` 和同一步的 checkpoint。
这里的 `FLOWPI_FLOW_REQUIRED_PROB=0.5` 表示 eligible 样本中的 forced-stale augmentation 比例，
不是整个 batch 中所有 stale VLM 样本的总比例。

监控新 run：

```bash
cd /inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/flowpi && \
.venv/bin/python scripts/monitor_flowpi_training.py \
  --run-root logs/flowpi_cache_train/flowpi_8xh200_flow_required --refresh 30
```

## 当前直接运行命令（每 2000 steps 保存 checkpoint）

```bash
cd /inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/flowpi && \
OPENPI_DATA_HOME=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/pi05/offline_assets/openpi_cache \
FLOWPI_WEIGHT_LOADER_PATH=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/pi05/offline_assets/openpi_cache/openpi-assets/checkpoints/pi05_base/params \
JAX_PLATFORMS=cuda \
FLOWPI_XLA_MEM_FRACTION=0.98 \
FLOWPI_SAVE_INTERVAL=2000 \
FLOWPI_EXP_NAME=flowpi_8xh200_flow_required \
FLOWPI_LOG_ROOT=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/flowpi/logs/flowpi_cache_train \
FLOWPI_FLOW_REQUIRED_PROB=0.5 \
FLOWPI_FLOW_REQUIRED_VLM_DELAY_MIN=5 \
FLOWPI_FLOW_GATE_INIT=0.01 \
FLOWPI_RESUME=0 \
FLOWPI_SEA_RAFT_CKPT=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/SEA-RAFT-FT/SEA-RAFT/checkpoints/24000_robot-ft-M-4gpu-shadow-15k-to-25k_robot-ft-M-4gpu-shadow-15k-to-25k-20260822-053345.pth \
FLOWPI_TRAIN_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
FLOWPI_TRAIN_EXPECTED_GPUS=8 \
FLOWPI_GLOBAL_BATCH=128 \
FLOWPI_WANDB_ENABLED=0 \
bash scripts/train_flowpi_flow_required_8xh200.sh
```
