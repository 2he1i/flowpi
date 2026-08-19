"""Offline replay of a FlowPi model on a dataset episode.

The replay feeds the runtime the *fresh* observation pipeline (current frames, state, prompt) —
exactly what online deployment produces. The runtime itself computes the online SEA-RAFT flow
and the slow-channel delay; the training-only flow/delay transforms (ComputeFlow, LoadFlowCache,
DelaySlowImage) and the camera history loading are therefore NOT applied here.

The runtime expects FULL-RESOLUTION camera frames (480x640): it computes the SEA-RAFT flow on
them and lets the model preprocess the same observation for the VLM. `ResizeImages` is therefore
dropped from the runtime data config's model transforms — the model preprocessor resizes to
224x224 internally, identically to the training pipeline.

Usage:
    uv run python scripts/flowpi_infer.py --config-name flowpi_aloha \
        --checkpoint /path/to/checkpoint --dataset data/adjust_bottle_ep0 \
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
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


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
    model = _checkpoints.load_model_from_checkpoint(train_config.model, args.checkpoint)
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
    # The runtime computes the SEA-RAFT flow on full-resolution frames; drop `ResizeImages` so
    # the batch carries 480x640 images (the model preprocessor resizes to 224x224 internally).
    runtime_data_config = dataclasses.replace(
        runtime_data_config,
        model_transforms=_transforms.Group(
            inputs=tuple(
                t for t in runtime_data_config.model_transforms.inputs if not isinstance(t, _transforms.ResizeImages)
            )
        ),
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

    all_actions = []

    for frame_idx, batch in enumerate(loader):
        if frame_idx >= frame_count:
            break
        obs = _model.Observation.from_dict(batch)

        if frame_idx == 0:
            # First frame: warm start + synchronous initial prefix refresh (must be active
            # before the first fast tick).
            runtime.warm_start(obs)
            runtime.refresh_prefix(obs, wait=True)
            # Emit the initial in-flight actions.
            all_actions.append(runtime.emit())
            continue

        # Fast tick.
        t0 = time.perf_counter()
        acts = runtime.tick(obs)
        runtime.stats["tick_total_ms"].append((time.perf_counter() - t0) * 1000)
        all_actions.append(acts)

        # Slow-channel refresh (async: the prefill runs in the background and is installed at
        # the next fast tick, so the replay loop never blocks on the VLM).
        if frame_idx % args.slow_every_n == 0:
            runtime.refresh_prefix(obs)

    # Drain the slow worker and propagate any prefill exception to the main thread.
    runtime.close()

    # Save actions.
    actions_np = np.concatenate(all_actions, axis=0)
    out_path = pathlib.Path(args.checkpoint) / "replay_actions.npz"
    np.savez(out_path, actions=actions_np)
    print(f"Saved {len(all_actions)} actions to {out_path}")

    # Print timing stats (f_fast / f_slow / d_VLM distributions).
    for name, vals in runtime.stats.items():
        if vals:
            arr = np.array(vals)
            print(f"{name}: mean={arr.mean():.1f}ms, min={arr.min():.1f}ms, max={arr.max():.1f}ms, n={len(arr)}")
    if runtime.num_generation_drops:
        print(f"dropped prefix generations: {runtime.num_generation_drops}")


if __name__ == "__main__":
    main()
