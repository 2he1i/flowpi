# FlowPI 8x RTX 4090 LoRA 训练

这套配置与 H100/H200 训练完全隔离：

- VLM language backbone：LoRA
- Action Expert：全参数训练
- Flow tokenizer / cross-attention / gate / delay embedding：全参数训练
- SigLIP：冻结
- SEA-RAFT：只读取已有 cache，不参与 policy training
- `global batch=32`，每卡 batch=4
- bf16，EMA 关闭以节省 24 GB 显存中的完整模型副本
- checkpoint 每 2000 steps 保存

直接运行：

```bash
cd /inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/flowpi && \
OPENPI_DATA_HOME=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/pi05/offline_assets/openpi_cache \
FLOWPI_WEIGHT_LOADER_PATH=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/DOMINO/policy/pi05/offline_assets/openpi_cache/openpi-assets/checkpoints/pi05_base/params \
JAX_PLATFORMS=cuda \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
FLOWPI_EXPECTED_GPUS=8 \
FLOWPI_GLOBAL_BATCH=32 \
FLOWPI_SAVE_INTERVAL=2000 \
FLOWPI_KEEP_PERIOD=2000 \
FLOWPI_XLA_MEM_FRACTION=0.92 \
FLOWPI_EMA_DECAY=None \
FLOWPI_SEA_RAFT_CKPT=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/SEA-RAFT-FT/SEA-RAFT/checkpoints/24000_robot-ft-M-4gpu-shadow-15k-to-25k_robot-ft-M-4gpu-shadow-15k-to-25k-20260822-053345.pth \
FLOWPI_WANDB_ENABLED=0 \
bash scripts/train_flowpi_8x4090_lora.sh
```

日志和 checkpoint 不会写入 H 平台实验目录：

```text
logs/flowpi_8x4090_lora/
checkpoints_8x4090_lora/flowpi_aloha_8x4090_lora/flowpi_8x4090_lora/
```

如果该实验目录已有 checkpoint，默认自动续训；需要从头开始时设置：

```bash
FLOWPI_RESUME=0 FLOWPI_OVERWRITE=1 bash scripts/train_flowpi_8x4090_lora.sh
```
