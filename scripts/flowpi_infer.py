"""Run the minimal three-GPU FlowPi inference loop on a LeRobot dataset.

The loop intentionally has no simulator adapter or rate controller. Every dataset frame is
processed in order: the slow JAX replica computes a fresh VLM prefix, the fast JAX replica
performs one streaming NFE, and SEA-RAFT computes optical-flow features on its Torch GPU.

All paths are command-line arguments so a checkout contains no machine-specific dataset or
checkpoint links. The script writes its action output only to the explicit ``--output`` path.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib

# Do this before importing JAX so the two explicitly selected replicas do not reserve all GPUs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flowpi_checkpoint as _checkpoint_config
import numpy as np
import torch

import openpi.models.model as _model
import openpi.policies.flowpi_runtime as _flowpi_runtime
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


def _preserve_episode_index(group: _transforms.Group) -> _transforms.Group:
    """Keep episode boundaries in the inference-only repack transform."""
    inputs = list(group.inputs)
    for index, transform in enumerate(inputs):
        if isinstance(transform, _transforms.RepackTransform):
            structure = dict(transform.structure)
            structure.setdefault("episode_index", "episode_index")
            inputs[index] = dataclasses.replace(transform, structure=structure)
            return dataclasses.replace(group, inputs=tuple(inputs))
    raise ValueError("FlowPi inference requires a RepackTransform with episode_index metadata.")


def _checkpoint_asset_id(assets_dir: pathlib.Path, configured: str | None) -> str | None:
    """Use an explicit asset id, or infer the only asset directory in a local checkpoint."""
    if configured:
        return configured
    if not assets_dir.is_dir():
        return None
    candidates = tuple(child.name for child in assets_dir.iterdir() if child.is_dir())
    return candidates[0] if len(candidates) == 1 else None


def _inference_data_factory(
    train_config: _config.TrainConfig,
    checkpoint: str,
    *,
    assets_dir: pathlib.Path | None,
    asset_id: str | None,
) -> _config.DataConfigFactory:
    """Remove training-only flow transforms and point normalization at external assets."""
    factory = train_config.data
    checkpoint_assets = _checkpoints.resolve_checkpoint_assets_dir(checkpoint)
    selected_assets = assets_dir or checkpoint_assets
    selected_asset_id = asset_id
    if selected_assets is not None:
        selected_asset_id = _checkpoint_asset_id(selected_assets, selected_asset_id)
        factory = dataclasses.replace(
            factory,
            assets=dataclasses.replace(
                factory.assets,
                assets_dir=str(selected_assets),
                asset_id=selected_asset_id,
            ),
        )

    flow_factory = getattr(factory, "flow", None)
    if flow_factory is not None:
        # Inference computes flow online and never reads the training cache or samples a
        # training-time slow-image delay.
        factory = dataclasses.replace(
            factory,
            flow=dataclasses.replace(
                flow_factory,
                sea_raft_ckpt=None,
                load_flow_cache=False,
                sample_vlm_delay=False,
            ),
        )
    return factory


def _runtime_data_config(
    factory: _config.DataConfigFactory,
    train_config: _config.TrainConfig,
    dataset: pathlib.Path,
) -> _config.DataConfig:
    """Build a fresh-frame data pipeline with no training-only flow transforms."""
    factory = dataclasses.replace(factory, repo_id=str(dataset), flow=None)
    data_config = factory.create(train_config.assets_dirs, train_config.model)
    return dataclasses.replace(
        data_config,
        repack_transforms=_preserve_episode_index(data_config.repack_transforms),
        model_transforms=_transforms.Group(
            inputs=tuple(
                transform
                for transform in data_config.model_transforms.inputs
                if not isinstance(transform, _transforms.ResizeImages)
            )
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="Training config used by the checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path, help="FlowPi checkpoint directory.")
    parser.add_argument("--dataset", required=True, type=pathlib.Path, help="External LeRobot v3 dataset root.")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Output .npz path for actions.")
    parser.add_argument("--assets-dir", type=pathlib.Path, help="External normalization-assets directory.")
    parser.add_argument("--asset-id", help="Normalization asset id inside --assets-dir.")
    parser.add_argument("--max-frames", type=int, help="Optional limit for a short data smoke run.")
    parser.add_argument("--skip-normalization", action="store_true", help="Save raw model-space actions.")
    parser.add_argument("--slow-jax-device", required=True, help="JAX device for the slow VLM replica, e.g. gpu:0.")
    parser.add_argument("--fast-jax-device", required=True, help="JAX device for the fast NFE replica, e.g. gpu:1.")
    parser.add_argument("--sea-raft-device", required=True, help="Torch device for SEA-RAFT, e.g. cuda:2.")
    parser.add_argument("--sea-raft-ckpt", required=True, type=pathlib.Path, help="Raw SEA-RAFT checkpoint path.")
    parser.add_argument("--sea-raft-variant", choices=("S", "M", "L"), help="SEA-RAFT model variant.")
    parser.add_argument("--sea-raft-iters", type=int, help="SEA-RAFT refinement iterations.")
    parser.add_argument(
        "--sea-raft-precision",
        choices=("fp32", "fp16", "bf16", "auto"),
        default="fp16",
        help="SEA-RAFT inference precision.",
    )
    parser.add_argument("--telemetry-json", type=pathlib.Path, help="Optional runtime telemetry output path.")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset = args.dataset.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.sea_raft_ckpt = args.sea_raft_ckpt.expanduser().resolve()
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset}")
    if not args.sea_raft_ckpt.is_file():
        raise FileNotFoundError(f"SEA-RAFT checkpoint does not exist: {args.sea_raft_ckpt}")
    return args


def main() -> None:
    args = _parse_args()
    train_config = _checkpoint_config.align_train_config_with_checkpoint(
        _config.get_config(args.config_name), args.checkpoint
    )
    model_flow = train_config.model.flow
    if model_flow is None or not model_flow.enabled:
        raise ValueError("The selected config/checkpoint must have model.flow enabled.")

    factory = _inference_data_factory(
        train_config,
        str(args.checkpoint),
        assets_dir=args.assets_dir.expanduser().resolve() if args.assets_dir else None,
        asset_id=args.asset_id,
    )
    data_config = factory.create(train_config.assets_dirs, train_config.model)
    runtime_data_config = _runtime_data_config(factory, train_config, args.dataset)

    # Two independent parameter copies make the three-device layout explicit. Both JAX calls are
    # deliberately synchronized; one action is emitted per dataset frame.
    slow_model = _checkpoints.load_model_from_checkpoint(
        train_config.model,
        str(args.checkpoint),
        jax_device=args.slow_jax_device,
    )
    fast_model = _checkpoints.load_model_from_checkpoint(
        train_config.model,
        str(args.checkpoint),
        jax_device=args.fast_jax_device,
    )

    dataset = _data_loader.transform_dataset(
        _data_loader.create_torch_dataset(
            runtime_data_config,
            action_horizon=1,
            model_config=train_config.model,
        ),
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
    frame_count = len(dataset) if args.max_frames is None else min(len(dataset), args.max_frames)

    output_transform = None
    if not args.skip_normalization and data_config.norm_stats is not None:
        output_transform = _transforms.compose(
            [
                _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ]
        )

    flow_data = data_config.flow
    runtime = _flowpi_runtime.FlowPiRuntime(
        fast_model,
        slow_model=slow_model,
        flow_config=model_flow,
        sea_raft_ckpt=str(args.sea_raft_ckpt),
        sea_raft_variant=args.sea_raft_variant or (flow_data.sea_raft_variant if flow_data else "M"),
        sea_raft_iters=args.sea_raft_iters or (flow_data.sea_raft_iters if flow_data else None),
        sea_raft_device=args.sea_raft_device,
        sea_raft_precision=args.sea_raft_precision,
        jax_device=args.fast_jax_device,
        slow_jax_device=args.slow_jax_device,
    )

    actions: list[np.ndarray] = []
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    active_episode: int | None = None
    try:
        for frame_index, batch in enumerate(loader):
            if frame_index >= frame_count:
                break
            observation = _model.Observation.from_dict(batch)
            episode_values = np.asarray(batch.get("episode_index", [])).reshape(-1)
            if episode_values.size != 1:
                raise ValueError("The dataset must provide exactly one episode_index per frame.")
            episode = int(episode_values[0])
            if active_episode != episode:
                runtime.warm_start(observation)
                emitted = runtime.emit()
                active_episode = episode
            else:
                emitted = runtime.tick(observation)

            actions.append(np.asarray(emitted))
            outputs.append((np.asarray(emitted), np.asarray(observation.state[0])))

    finally:
        runtime.close()

    if not actions:
        raise ValueError("The dataset produced no frames.")
    if output_transform is not None:
        actions_np = np.concatenate(
            [np.asarray(output_transform({"actions": value, "state": state})["actions"]) for value, state in outputs],
            axis=0,
        )
    else:
        actions_np = np.concatenate(actions, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, actions=actions_np)
    print(f"Saved {len(actions_np)} actions to {args.output}")
    if args.telemetry_json:
        args.telemetry_json.parent.mkdir(parents=True, exist_ok=True)
        args.telemetry_json.write_text(json.dumps(runtime.metrics_summary(), indent=2))
        print(f"Wrote telemetry to {args.telemetry_json}")


if __name__ == "__main__":
    main()
