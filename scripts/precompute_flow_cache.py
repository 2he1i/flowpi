"""Precompute the SEA-RAFT optical flow cache for a flowpi training config.

For every episode, camera, frame t, and lag k = 1..K, computes `flow_8x = RAFT(I_{t-k*Δ}, I_t)`
at 1/8 resolution and stores the *raw* flow as float16 (normalization is deferred to load time so
that flow_scale / flow_clamp can be changed without recomputing the cache).

Output layout:
  {flow_cache_dir}/meta.json
  {flow_cache_dir}/episode-{ep:06d}/{cam_key}.npy   # [T, K, 2, H//8, W//8] float16 (raw)
  {flow_cache_dir}/episode-{ep:06d}/valid.npy       # [T, K] bool

Episode -> parquet-row indexing is *not* reimplemented here: `LeRobotV3ParquetDataset` is the
single source of truth (see `episode_frames`), so episodes sharing a parquet file are sliced to
their own row range and multi-file datasets stay aligned with the training loader.

Usage:
  uv run python scripts/precompute_flow_cache.py --config-name flowpi_aloha \
      [--data.flow.sea-raft-ckpt /path/to/sea_raft.pth] [--max-frames 20] [--num-workers 4]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import pathlib

import numpy as np
from tqdm import tqdm
import tyro


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    known, remaining = parser.parse_known_args()
    import openpi.training.config as _config

    configs = {k: (k, v) for k, v in _config._CONFIGS_DICT.items()}  # noqa: SLF001
    train_config = tyro.extras.overridable_config_cli(configs, args=remaining, prog="precompute_flow_cache")
    return known, train_config


def _episode_entries(root: pathlib.Path) -> list[dict]:
    """Episode metadata (index + length). Indexing is delegated to
    `LeRobotV3ParquetDataset.episode_frames` at decode time; only the episode list is read here."""
    from openpi.training.lerobot_v3_dataset import LeRobotV3ParquetDataset

    dataset = LeRobotV3ParquetDataset(root)
    return [{"episode_index": ep.episode_index, "length": ep.length} for ep in dataset.episodes]


def _cache_name(cam_key: str) -> str:
    """Cache file name for a camera: the dataset key with the 'observation.images.' prefix stripped,
    matching the post-repack camera names used by LoadFlowCache."""
    return cam_key.removeprefix("observation.images.")


def _process_episode(task: dict) -> dict:
    """Worker: computes and saves the flow cache for one episode."""
    from openpi.training.lerobot_v3_dataset import LeRobotV3ParquetDataset
    from openpi.training.sea_raft import SeaRaftFlowExtractor

    extractor = SeaRaftFlowExtractor(
        ckpt_path=task["sea_raft_ckpt"] or None,
        variant=task.get("sea_raft_variant", "M"),
        iters=task.get("sea_raft_iters"),
        device=task["sea_raft_device"],
        allow_random_init=task.get("sea_raft_allow_random_init", False),
    )
    root = pathlib.Path(task["root"])
    entry = task["entry"]
    k_num, stride, batch_size = task["num_flow_steps"], task["flow_stride_frames"], task["batch_size"]

    # Single source of truth for episode -> parquet rows: the same dataset reader used by the
    # training loader decodes exactly the episode's local row range (never a whole file, so
    # multi-episode / multi-file datasets stay aligned with training).
    dataset = LeRobotV3ParquetDataset(root)
    frames = dataset.episode_frames(entry["episode_index"])
    t_len = next(iter(frames.values())).shape[0]
    # With --max-frames the entry length was truncated in main(); the decoded rows must cover
    # at least that many frames (and exactly the metadata length when untruncated).
    if t_len < entry["length"] or (task.get("max_frames") is None and t_len != entry["length"]):
        raise ValueError(
            f"Episode {entry['episode_index']}: metadata length {entry['length']} but the parquet "
            f"rows decoded to {t_len} frames. Episode metadata and data files are inconsistent."
        )
    # --max-frames truncates the cache (smoke runs) per episode, never crossing into another
    # episode's rows (the metadata length was already truncated in main()).
    if task.get("max_frames") is not None:
        frames = {cam: arr[: entry["length"]] for cam, arr in frames.items()}
    t_len = next(iter(frames.values())).shape[0]
    _, _, height, width = next(iter(frames.values())).shape

    # Fail fast instead of writing a cache whose flow grid silently disagrees with the model:
    # SEA-RAFT downsamples by 8, so the cached flow is (H//8, W//8) of *this* frame size. The
    # meta.json records the actual input size, and LoadFlowCache._validate_meta() checks it
    # against flow_image_size at load time. Do NOT resize here: silently resampling the dataset
    # would hide a resolution mismatch between the dataset and the training config. Checked
    # before creating the episode dir so a mismatch leaves no partial cache behind.
    expected = tuple(task["flow_image_size"])
    if (height, width) != expected:
        raise ValueError(
            f"Episode {entry['episode_index']}: dataset frames are {height}x{width} but "
            f"flow_image_size is {expected[0]}x{expected[1]}. The cached flow grid would not "
            "match the model's flow tokenizer. Configure flow_image_size to the dataset "
            "resolution or resize the dataset."
        )

    ep_dir = pathlib.Path(task["flow_cache_dir"]) / f"episode-{entry['episode_index']:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    valid = np.zeros((t_len, k_num), dtype=bool)
    pairs = []  # (t, k, cam)
    for t in range(t_len):
        for k in range(1, k_num + 1):
            if t >= k * stride:
                valid[t, k - 1] = True
                pairs.extend((t, k, cam) for cam in task["cam_keys"])

    flows = {cam: np.zeros((t_len, k_num, 2, height // 8, width // 8), dtype=np.float16) for cam in task["cam_keys"]}

    for start in tqdm(
        range(0, len(pairs), batch_size), desc=f"ep{entry['episode_index']:06d}", disable=not task["verbose"]
    ):
        chunk = pairs[start : start + batch_size]
        prev = np.stack([frames[cam][t - k * stride] for t, k, cam in chunk], axis=0)[None]
        curr = np.stack([frames[cam][t] for t, k, cam in chunk], axis=0)[None]
        out = extractor.compute(prev, curr)[0]  # [B, 2, h, w]
        for (t, k, cam), flow in zip(chunk, out, strict=True):
            flows[cam][t, k - 1] = flow.astype(np.float16)

    for cam in task["cam_keys"]:
        np.save(ep_dir / f"{_cache_name(cam)}.npy", flows[cam])
    np.save(ep_dir / "valid.npy", valid)
    return {
        "episode": entry["episode_index"],
        "num_pairs": len(pairs),
        "image_size": [height, width],
    }


def main():
    extra, train_config = _parse_args()
    model_flow = getattr(train_config.model, "flow", None)
    if model_flow is None or not model_flow.enabled:
        raise ValueError("Precomputing the flow cache requires a config with model.flow enabled.")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    flow_cfg = data_config.flow
    if flow_cfg is None or not flow_cfg.enabled:
        raise ValueError("--data.flow.enabled must be true to precompute the flow cache.")
    if flow_cfg.flow_cache_dir is None:
        raise ValueError("--data.flow.flow-cache-dir must be set.")

    repo_id = data_config.repo_id
    local_root = pathlib.Path(repo_id)
    if not (local_root / "meta" / "info.json").exists():
        raise ValueError(
            f"flow cache precomputation currently requires a local LeRobot v3.0 dataset path, got {repo_id}"
        )

    with open(local_root / "meta" / "info.json") as f:
        info = json.load(f)
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError(f"Expected a LeRobot v3.0 dataset, got codebase_version={info.get('codebase_version')}")

    cam_keys = [key for key, ft in info["features"].items() if ft.get("dtype") in ("image", "video")]
    cache_dir = pathlib.Path(flow_cfg.flow_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not extra.overwrite and (cache_dir / "meta.json").exists():
        raise FileExistsError(f"Flow cache already exists at {cache_dir}. Use --overwrite to recompute.")

    entries = _episode_entries(local_root)
    if extra.max_frames is not None:
        if extra.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        remaining = extra.max_frames
        selected_entries = []
        for entry in entries:
            if remaining <= 0:
                break
            take = min(entry["length"], remaining)
            if take <= 0:
                continue
            selected_entries.append({**entry, "length": take})
            remaining -= take
        entries = selected_entries

    if not entries:
        raise ValueError("No episode frames selected for flow-cache precomputation.")

    tasks = [
        {
            "root": str(local_root),
            "entry": entry,
            "cam_keys": cam_keys,
            "num_flow_steps": model_flow.num_flow_steps,
            "flow_stride_frames": model_flow.flow_stride_frames,
            "flow_image_size": list(model_flow.flow_image_size),
            "max_frames": extra.max_frames,
            "flow_cache_dir": str(cache_dir),
            "sea_raft_ckpt": flow_cfg.sea_raft_ckpt,
            "sea_raft_variant": flow_cfg.sea_raft_variant,
            "sea_raft_iters": flow_cfg.sea_raft_iters,
            "sea_raft_device": flow_cfg.sea_raft_device,
            "sea_raft_allow_random_init": flow_cfg.sea_raft_allow_random_init,
            "batch_size": 16,
            "verbose": extra.num_workers <= 1,
        }
        for entry in entries
    ]

    if extra.num_workers <= 1:
        results = [_process_episode(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=extra.num_workers) as pool:
            results = list(pool.map(_process_episode, tasks))

    from openpi.training.sea_raft import checkpoint_sha256

    meta = {
        "num_flow_steps": model_flow.num_flow_steps,
        "flow_stride_frames": model_flow.flow_stride_frames,
        # The actual frame size fed to SEA-RAFT (validated == flow_image_size in the workers).
        "image_size": results[0]["image_size"],
        "fps": info["fps"],
        "sea_raft_ckpt": flow_cfg.sea_raft_ckpt,
        "sea_raft_checkpoint_sha256": checkpoint_sha256(flow_cfg.sea_raft_ckpt),
        "sea_raft_variant": flow_cfg.sea_raft_variant,
        "sea_raft_iters": flow_cfg.sea_raft_iters,
        "camera_keys": sorted(_cache_name(cam) for cam in cam_keys),
        "episodes": {str(r["episode"]): r["num_pairs"] for r in results},
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote flow cache to {cache_dir}: {sum(r['num_pairs'] for r in results)} flow pairs.")


if __name__ == "__main__":
    main()
