"""Offline replay of a FlowPi model on one or more dataset episodes.

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
import json
import os
import pathlib
import time

# Keep JAX from reserving all visible GPUs before the explicitly selected model/SEA-RAFT devices
# are constructed. The online RoboTwin launcher sets this too, but the standalone replay script
# must be safe when invoked directly.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flowpi_checkpoint as _checkpoint_config
import jax
import numpy as np
import torch

import openpi.models.model as _model
import openpi.policies.flowpi_runtime as flowpi_runtime
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sea_raft as _sea_raft
import openpi.transforms as _transforms


def _preserve_episode_index(group: _transforms.Group) -> _transforms.Group:
    """Keep the raw episode id through the replay-only repack transform."""
    inputs = list(group.inputs)
    for index, transform in enumerate(inputs):
        if isinstance(transform, _transforms.RepackTransform):
            structure = dict(transform.structure)
            structure.setdefault("episode_index", "episode_index")
            inputs[index] = dataclasses.replace(transform, structure=structure)
            return dataclasses.replace(group, inputs=tuple(inputs))
    raise ValueError("Replay requires a RepackTransform so it can detect episode boundaries.")


def _default_single_jax_device() -> str:
    """Choose one device for standalone replay instead of Orbax's all-device replication."""
    try:
        if jax.devices("gpu"):
            return "gpu:0"
    except RuntimeError:
        pass
    return "cpu"


def _create_inference_data_factory(
    train_config: _config.TrainConfig,
    checkpoint: str,
    sea_raft_ckpt: pathlib.Path,
):
    """Keep runtime settings while removing transforms that only belong to training.

    A FlowPi training config normally uses LoadFlowCache and DelaySlowImage.
    Neither should be constructed for replay: the runtime computes flow online and
    tracks the slow-channel delay itself. Point the factory at the checkpoint assets
    first so normalization uses the exact statistics saved with the trained weights.
    """
    data_factory = train_config.data
    checkpoint_assets_dir = _checkpoints.resolve_checkpoint_assets_dir(checkpoint)
    if checkpoint_assets_dir is not None:
        data_factory = dataclasses.replace(
            data_factory,
            assets=dataclasses.replace(data_factory.assets, assets_dir=str(checkpoint_assets_dir)),
        )

    flow_factory = getattr(data_factory, "flow", None)
    if flow_factory is not None:
        data_factory = dataclasses.replace(
            data_factory,
            flow=dataclasses.replace(
                flow_factory,
                # Do not inherit the training-only resume checkpoint (or its placeholder) from
                # the config. Replay always uses the raw inference checkpoint explicitly passed
                # to this function.
                sea_raft_ckpt=str(sea_raft_ckpt),
                load_flow_cache=False,
                sample_vlm_delay=False,
            ),
        )
    return data_factory


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
        "--sea-raft-precision",
        choices=("fp32", "fp16", "bf16", "auto"),
        default="fp16",
        help="SEA-RAFT inference precision; fp16 uses Tensor Cores on RTX 4090 (default: fp16).",
    )
    parser.add_argument(
        "--sea-raft-ckpt",
        type=pathlib.Path,
        default=_sea_raft.default_inference_checkpoint(),
        help=(
            "Raw SEA-RAFT model checkpoint for inference. The default is the repository-local "
            "SEA-RAFT/ckpt/shadow-24k.pth; this is intentionally separate from the training "
            "resume checkpoint."
        ),
    )
    parser.add_argument(
        "--telemetry-json",
        type=pathlib.Path,
        default=None,
        help="Dump per-tick freshness telemetry + timing stats to a JSON file (for "
        "scripts/fit_vlm_delay.py). Use --realtime for deployment-equivalent delays.",
    )
    args = parser.parse_args()
    if args.slow_every_n <= 0:
        raise ValueError("--slow-every-n must be positive")
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    jax_device = args.jax_device or _default_single_jax_device()

    sea_raft_ckpt = args.sea_raft_ckpt.expanduser()
    if not sea_raft_ckpt.is_file():
        raise FileNotFoundError(
            f"Inference SEA-RAFT checkpoint not found: {sea_raft_ckpt}. "
            "Place the raw model checkpoint under SEA-RAFT/ckpt/shadow-24k.pth or pass "
            "--sea-raft-ckpt explicitly."
        )

    # Load the config and create the model.
    train_config = _checkpoint_config.align_train_config_with_checkpoint(
        _config.get_config(args.config_name), args.checkpoint
    )
    model = _checkpoints.load_model_from_checkpoint(
        train_config.model,
        args.checkpoint,
        jax_device=jax_device,
    )
    flow_cfg = train_config.model.flow
    if flow_cfg is None or not flow_cfg.enabled:
        raise ValueError("The model checkpoint must have flow enabled.")

    # Keep the flow settings (SEA-RAFT checkpoint/device and model-derived geometry) for the
    # runtime, but never construct training-only cache/delay transforms. The factory also points
    # at checkpoint-local assets when available, making replay independent of the training assets
    # location and ensuring normalization matches the restored weights.
    inference_data_factory = _create_inference_data_factory(train_config, args.checkpoint, sea_raft_ckpt)
    data_config = inference_data_factory.create(train_config.assets_dirs, train_config.model)

    # Rebuild the data config with the flow pipeline disabled: the replay must feed the runtime
    # fresh current frames + state + prompt, and the runtime computes the flow and the slow delay
    # itself. Disabling flow also drops the camera history from the dataset (single-frame images).
    runtime_data_config = dataclasses.replace(inference_data_factory, repo_id=args.dataset, flow=None).create(
        train_config.assets_dirs, train_config.model
    )
    # The runtime computes the SEA-RAFT flow on full-resolution frames; drop `ResizeImages` so
    # the batch carries 480x640 images (the model preprocessor resizes to 224x224 internally).
    runtime_data_config = dataclasses.replace(
        runtime_data_config,
        repack_transforms=_preserve_episode_index(runtime_data_config.repack_transforms),
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

    # Build the runtime with the inference-only raw SEA-RAFT checkpoint. This must not use the
    # training checkpoint: the latter may contain optimizer/scheduler state for resuming SEA-RAFT.
    sea_raft_device = args.sea_raft_device or (data_config.flow.sea_raft_device if data_config.flow else "cpu")
    runtime = flowpi_runtime.FlowPiRuntime(
        model,
        flow_config=flow_cfg,
        sea_raft_ckpt=sea_raft_ckpt,
        sea_raft_variant=data_config.flow.sea_raft_variant if data_config.flow else "M",
        sea_raft_iters=data_config.flow.sea_raft_iters if data_config.flow else None,
        sea_raft_device=sea_raft_device,
        sea_raft_precision=args.sea_raft_precision,
        jax_device=jax_device,
        d=1,
    )

    all_actions = []
    # (actions, normalized state) pairs; the state is needed by the output transforms to
    # invert the delta/absolute repack.
    all_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    period = 1.0 / args.control_hz
    active_episode_index: int | None = None

    for frame_idx, batch in enumerate(loader):
        if frame_idx >= frame_count:
            break
        obs = _model.Observation.from_dict(batch)

        episode_values = np.asarray(batch.get("episode_index", [])).reshape(-1)
        if episode_values.size != 1:
            raise ValueError(
                "Replay could not recover exactly one episode_index per frame. "
                "Use a LeRobot v3 dataset with episode_index in its metadata."
            )
        episode_index = int(episode_values[0])
        if active_episode_index != episode_index:
            # First frame of each episode: warm_start resets all temporal state and already
            # installs the initial prefix before the first fast tick.
            runtime.warm_start(obs)
            emit = runtime.emit()
            all_actions.append(emit)
            all_outputs.append((emit, np.asarray(obs.state[0])))
            active_episode_index = episode_index
            continue

        # Fast tick (paced to the control period in realtime mode).
        loop_t0 = time.perf_counter()
        acts = runtime.tick(obs)
        all_actions.append(acts)
        all_outputs.append((acts, np.asarray(obs.state[0])))

        # Launch the next slow refresh before pacing the tick. Its worker can now overlap the
        # remaining control-period sleep instead of starting one full tick late.
        if frame_idx % args.slow_every_n == 0:
            runtime.refresh_prefix()

        if args.realtime:
            elapsed = time.perf_counter() - loop_t0
            if elapsed < period:
                time.sleep(period - elapsed)
        runtime.stats["tick_wall_ms"].append((time.perf_counter() - loop_t0) * 1000)

    # Drain the slow worker and propagate any prefill exception to the main thread.
    runtime.close()
    runtime_metrics = runtime.metrics_summary()

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

    print(
        flowpi_runtime.format_metrics_table(
            runtime_metrics,
            title="FlowPi Offline Replay 推理指标 / Inference Metrics",
        ),
        flush=True,
    )

    if args.telemetry_json is not None:
        payload = {
            "vlm_delay_max": flow_cfg.vlm_delay_max,
            # Keep the legacy top-level fields for fit_vlm_delay.py and add the complete
            # latency/frequency/counter summary for deployment analysis.
            "telemetry": runtime_metrics["telemetry"],
            "stats": runtime_metrics["stats"],
            "metrics": runtime_metrics,
        }
        with open(args.telemetry_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote telemetry to {args.telemetry_json}")


if __name__ == "__main__":
    main()
