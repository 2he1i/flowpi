"""Offline replay of a FlowPi model on a dataset episode.

The replay feeds the runtime the *fresh* observation pipeline (current frames, state, prompt) —
exactly what online deployment produces. The runtime itself computes the online SEA-RAFT flow
and the slow-channel delay; the training-only flow/delay transforms (ComputeFlow, LoadFlowCache,
DelaySlowImage) and the camera history loading are therefore NOT applied here.

The runtime expects FULL-RESOLUTION camera frames (480x640): it computes the SEA-RAFT flow on
them and lets the model preprocess the same observation for the VLM. `ResizeImages` is therefore
dropped from the runtime data config's model transforms — the model preprocessor resizes to
224x224 internally, identically to the training pipeline.

Two replay modes:

- *functional* (default): no wall-clock pacing; a tick is as fast as the hardware allows.
- *realtime* (``--realtime``): ticks are paced to a fixed control period (``--control-hz``), so
  one tick is a physical 20 ms at 50 Hz and the measured delays are in real milliseconds. This
  is the mode to use for deployment-equivalent timing / freshness measurements.

Actions are saved in *robot-executable* space (unnormalized + the data config's output
transforms, e.g. AbsoluteActions + AlohaOutputs), matching the policy server's output pipeline.

Usage:
    uv run python scripts/flowpi_infer.py --config-name flowpi_aloha \
        --checkpoint /path/to/checkpoint --dataset data/adjust_bottle_ep0 \
        --slow-every-n 10 [--realtime --control-hz 50]
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
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace ticks to a fixed control period (wall clock) instead of running as fast as possible.",
    )
    parser.add_argument("--control-hz", type=float, default=50.0, help="Control frequency in realtime mode (Hz)")
    parser.add_argument("--jax-device", type=str, default=None, help="JAX model device (e.g. cuda:0 / gpu:1)")
    parser.add_argument("--sea-raft-device", type=str, default=None, help="SEA-RAFT torch device (e.g. cuda:0)")
    parser.add_argument(
        "--telemetry-json",
        type=pathlib.Path,
        default=None,
        help="Dump per-tick freshness telemetry + timing stats to a JSON file (for "
        "scripts/fit_vlm_delay.py). Use --realtime for deployment-equivalent delays.",
    )
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

    # Robot-executable output pipeline, mirroring the policy server: unnormalize the model-space
    # actions, then invert the data output transforms (e.g. AbsoluteActions + AlohaOutputs).
    # Only meaningful when normalization was applied; with --skip-normalization the raw
    # model-space actions are saved (debug only).
    output_transform = None
    if not args.skip_normalization and data_config.norm_stats is not None:
        output_transform = _transforms.compose(
            [
                _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ]
        )

    # Build the runtime (fails fast when no SEA-RAFT checkpoint is configured).
    sea_raft_device = args.sea_raft_device or (data_config.flow.sea_raft_device if data_config.flow else "cpu")
    runtime = flowpi_runtime.FlowPiRuntime(
        model,
        flow_config=flow_cfg,
        sea_raft_ckpt=data_config.flow.sea_raft_ckpt if data_config.flow else None,
        sea_raft_device=sea_raft_device,
        jax_device=args.jax_device,
        d=1,
    )

    all_actions = []
    # (actions, normalized state) pairs; the state is needed by the output transforms to
    # invert the delta/absolute repack.
    all_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    period = 1.0 / args.control_hz

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
            emit = runtime.emit()
            all_actions.append(emit)
            all_outputs.append((emit, np.asarray(obs.state[0])))
            continue

        # Fast tick (paced to the control period in realtime mode).
        loop_t0 = time.perf_counter()
        t0 = time.perf_counter()
        acts = runtime.tick(obs)
        runtime.stats["tick_total_ms"].append((time.perf_counter() - t0) * 1000)
        all_actions.append(acts)
        all_outputs.append((acts, np.asarray(obs.state[0])))

        if args.realtime:
            elapsed = time.perf_counter() - loop_t0
            if elapsed < period:
                time.sleep(period - elapsed)
        runtime.stats["tick_wall_ms"].append((time.perf_counter() - loop_t0) * 1000)

        # Slow-channel refresh (async: the prefill runs in the background and is installed at
        # the next fast tick, so the replay loop never blocks on the VLM).
        if frame_idx % args.slow_every_n == 0:
            runtime.refresh_prefix(obs)

    # Drain the slow worker and propagate any prefill exception to the main thread.
    runtime.close()

    # Post-process the actions into robot-executable space (unnormalized + output transforms).
    actions_np = np.concatenate(all_actions, axis=0)
    if output_transform is not None:
        processed = [
            output_transform({"actions": np.asarray(acts), "state": state})["actions"] for acts, state in all_outputs
        ]
        actions_np = np.concatenate([np.asarray(out) for out in processed], axis=0)
    else:
        print("WARNING: --skip-normalization: saving raw model-space actions (debug only)")

    out_path = pathlib.Path(args.checkpoint) / "replay_actions.npz"
    np.savez(out_path, actions=actions_np)
    print(f"Saved {len(all_actions)} actions to {out_path}")

    # Timing stats (f_fast / f_slow / d_VLM distributions).
    for name, vals in runtime.stats.items():
        if vals:
            arr = np.array(vals)
            print(f"{name}: mean={arr.mean():.1f}ms, min={arr.min():.1f}ms, max={arr.max():.1f}ms, n={len(arr)}")
    if runtime.num_generation_drops:
        print(f"dropped prefix generations: {runtime.num_generation_drops}")

    # Freshness telemetry: reconstruct Age_VLM (ticks and ms) from the per-tick source ticks.
    if runtime.telemetry:
        ticks = np.array([t["tick"] for t in runtime.telemetry])
        prefix_src = np.array([t["prefix_source_tick"] for t in runtime.telemetry])
        delays = np.array([t["delay_ticks"] for t in runtime.telemetry])
        print(
            f"freshness: ticks={len(ticks)}, "
            f"prefix_source_tick last={prefix_src[-1]} (of tick {ticks[-1]}), "
            f"age_ticks mean={delays.mean():.2f} max={delays.max()} (d_max={flow_cfg.vlm_delay_max})"
        )
    if runtime.stats.get("prefix_age_ms_at_install"):
        arr = np.array(runtime.stats["prefix_age_ms_at_install"])
        print(f"prefix_age_ms_at_install: mean={arr.mean():.1f}ms, max={arr.max():.1f}ms, n={len(arr)}")

    if args.telemetry_json is not None:
        payload = {
            "vlm_delay_max": flow_cfg.vlm_delay_max,
            "telemetry": runtime.telemetry,
            "stats": {name: vals for name, vals in runtime.stats.items() if vals},
        }
        with open(args.telemetry_json, "w") as f:
            import json

            json.dump(payload, f, indent=2)
        print(f"Wrote telemetry to {args.telemetry_json}")


if __name__ == "__main__":
    main()
