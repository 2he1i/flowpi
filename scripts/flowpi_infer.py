"""Offline replay of a FlowPi model on a dataset episode.

The replay feeds the runtime the *fresh* observation pipeline (current frames, state, prompt) —
exactly what online deployment produces. The runtime itself computes the online SEA-RAFT flow
and the slow-channel delay; the training-only flow/delay transforms (ComputeFlow, LoadFlowCache,
DelaySlowImage) and the camera history loading are therefore NOT applied here.

Usage:
    uv run python scripts/flowpi_infer.py --config-name flowpi_aloha \
        --checkpoint /path/to/checkpoint --dataset test_data/adjust_bottle_ep0 \
        --slow-every-n 10
"""

import argparse
import dataclasses
import pathlib
import time

import numpy as np
import torch

import openpi.models.model as _model
import openpi.policies.flowpi_runtime as flowpi_runtime
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True, help="Training config name (e.g. flowpi_aloha)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--dataset", required=True, help="Local dataset root (overrides the config repo_id)")
    parser.add_argument("--slow-every-n", type=int, default=10, help="Prefix refresh interval (ticks)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit the number of frames")
    parser.add_argument("--skip-normalization", action="store_true", help="Skip state normalization (debug only)")
    args = parser.parse_args()

    # Load the config and create the model.
    train_config = _config.get_config(args.config_name)
    model = train_config.model.load(args.checkpoint)
    flow_cfg = train_config.model.flow
    if flow_cfg is None or not flow_cfg.enabled:
        raise ValueError("The model checkpoint must have flow enabled.")

    # Full (flow-enabled) config: carries the SEA-RAFT checkpoint/device used at training time.
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    # Rebuild the data config with the flow pipeline disabled: the replay must feed the runtime
    # fresh current frames + state + prompt, and the runtime computes the flow and the slow delay
    # itself. Disabling flow also drops the camera history from the dataset (single-frame images).
    runtime_data_config = dataclasses.replace(train_config.data, repo_id=args.dataset, flow=None).create(
        train_config.assets_dirs, train_config.model
    )

    # Standard pipeline: repack + inputs (no flow/delay) + normalize + model transforms, one
    # sample per frame, collated into a batch of 1 for the runtime.
    dataset = _data_loader.transform_dataset(
        _data_loader.create_torch_dataset(runtime_data_config, action_horizon=1, model_config=train_config.model),
        runtime_data_config,
        skip_norm_stats=args.skip_normalization,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=_data_loader._collate_fn,  # noqa: SLF001
        num_workers=0,
    )
    frame_count = len(dataset)
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)

    # Build the runtime (fails fast when no SEA-RAFT checkpoint is configured).
    runtime = flowpi_runtime.FlowPiRuntime(
        model,
        flow_config=flow_cfg,
        sea_raft_ckpt=data_config.flow.sea_raft_ckpt if data_config.flow else None,
        sea_raft_device=data_config.flow.sea_raft_device if data_config.flow else "cpu",
        d=1,
    )

    timing = {"raft_ms": [], "nfe_ms": [], "prefill_ms": []}
    all_actions = []

    for frame_idx, batch in enumerate(loader):
        if frame_idx >= frame_count:
            break
        obs = _model.Observation.from_dict(batch)

        if frame_idx == 0:
            # First frame: warm start + initial prefix refresh.
            t0 = time.perf_counter()
            runtime.warm_start(obs)
            t1 = time.perf_counter()
            runtime.refresh_prefix(obs)
            t2 = time.perf_counter()
            timing["prefill_ms"].append((t2 - t1) * 1000)
            # Emit the initial in-flight actions.
            all_actions.append(runtime.emit())
            continue

        # Fast tick.
        t0 = time.perf_counter()
        acts = runtime.tick(obs)
        dt = (time.perf_counter() - t0) * 1000
        timing["nfe_ms"].append(dt)
        all_actions.append(acts)

        # Slow-channel refresh.
        if frame_idx % args.slow_every_n == 0:
            t0 = time.perf_counter()
            runtime.refresh_prefix(obs)
            timing["prefill_ms"].append((time.perf_counter() - t0) * 1000)

    # Save actions.
    actions_np = np.concatenate(all_actions, axis=0)
    out_path = pathlib.Path(args.checkpoint) / "replay_actions.npz"
    np.savez(out_path, actions=actions_np)
    print(f"Saved {len(all_actions)} actions to {out_path}")

    # Print timing stats.
    for name, vals in timing.items():
        if vals:
            arr = np.array(vals)
            print(f"{name}: mean={arr.mean():.1f}ms, min={arr.min():.1f}ms, max={arr.max():.1f}ms, n={len(arr)}")


if __name__ == "__main__":
    main()
