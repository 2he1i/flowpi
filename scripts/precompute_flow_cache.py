"""Precompute the SEA-RAFT optical flow cache for a flowpi training config.

For every episode, camera, frame t, and lag k = 1..K, computes `flow_8x = RAFT(I_{t-k*Δ}, I_t)`
at 1/8 resolution and stores the *raw* flow as float16 (normalization is deferred to load time so
that flow_scale / flow_clamp can be changed without recomputing the cache).

Output layout:
  {flow_cache_dir}/meta.json
  {flow_cache_dir}/episode-{ep:06d}/{cam_key}.npy   # [T, K, 2, H//8, W//8] float16 (raw)
  {flow_cache_dir}/episode-{ep:06d}/valid.npy       # [T, K] bool

Usage:
  uv run python scripts/precompute_flow_cache.py --config-name flowpi_aloha \
      [--data.flow.sea-raft-ckpt /path/to/sea_raft.pth] [--max-frames 20] [--num-workers 4]
"""

import argparse
import dataclasses
import io
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import tyro
from PIL import Image
from tqdm import tqdm


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    known, remaining = parser.parse_known_args()
    import openpi.training.config as _config  # noqa: PLC0415

    configs = {k: (k, v) for k, v in _config._CONFIGS_DICT.items()}
    train_config = tyro.extras.overridable_config_cli(configs, args=remaining, prog="precompute_flow_cache")
    return known, train_config


def _episode_entries(root: pathlib.Path) -> list[dict]:
    entries = []
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        columns = table.column_names
        chunk_col = "data/chunk_index" if "data/chunk_index" in columns else "chunk_index"
        file_col = "data/file_index" if "data/file_index" in columns else "file_index"
        for row in range(table.num_rows):
            entries.append(
                {
                    "episode_index": int(table.column("episode_index")[row].as_py()),
                    "length": int(table.column("length")[row].as_py()),
                    "chunk_index": int(table.column(chunk_col)[row].as_py()),
                    "file_index": int(table.column(file_col)[row].as_py()),
                }
            )
    entries.sort(key=lambda e: e["episode_index"])
    return entries


def _decode_episode_frames(root: pathlib.Path, entry: dict, cam_keys: list[str]) -> dict[str, np.ndarray]:
    """Decodes all frames of one episode for every camera -> {cam: [T, 3, H, W] uint8}."""
    path = root / "data" / f"chunk-{entry['chunk_index']:03d}" / f"file-{entry['file_index']:03d}.parquet"
    table = pq.read_table(path, columns=cam_keys)
    out = {}
    for cam in cam_keys:
        frames = []
        for row in range(table.num_rows):
            item = table.column(cam)[row].as_py()
            img = Image.open(io.BytesIO(item["bytes"]))
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            frames.append(np.transpose(arr, (2, 0, 1)))
        out[cam] = np.stack(frames, axis=0)
    return out


def _process_episode(task: dict) -> dict:
    """Worker: computes and saves the flow cache for one episode."""
    from openpi.training.sea_raft import SeaRaftFlowExtractor  # noqa: PLC0415

    extractor = SeaRaftFlowExtractor(
        ckpt_path=task["sea_raft_ckpt"] or None, variant="M", device=task["sea_raft_device"]
    )
    root = pathlib.Path(task["root"])
    entry = task["entry"]
    k_num, stride, batch_size = task["num_flow_steps"], task["flow_stride_frames"], task["batch_size"]

    ep_dir = pathlib.Path(task["flow_cache_dir"]) / f"episode-{entry['episode_index']:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    frames = _decode_episode_frames(root, entry, task["cam_keys"])
    t_len = next(iter(frames.values())).shape[0]
    _, _, height, width = next(iter(frames.values())).shape

    valid = np.zeros((t_len, k_num), dtype=bool)
    pairs = []  # (t, k, cam)
    for t in range(t_len):
        for k in range(1, k_num + 1):
            if t >= k * stride:
                valid[t, k - 1] = True
                for cam in task["cam_keys"]:
                    pairs.append((t, k, cam))

    flows = {cam: np.zeros((t_len, k_num, 2, height // 8, width // 8), dtype=np.float16) for cam in task["cam_keys"]}

    for start in tqdm(range(0, len(pairs), batch_size), desc=f"ep{entry['episode_index']:06d}", disable=not task["verbose"]):
        chunk = pairs[start : start + batch_size]
        prev = np.stack([frames[cam][t - k * stride] for t, k, cam in chunk], axis=0)[None]
        curr = np.stack([frames[cam][t] for t, k, cam in chunk], axis=0)[None]
        out = extractor.compute(prev, curr)[0]  # [B, 2, h, w]
        for (t, k, cam), flow in zip(chunk, out, strict=True):
            flows[cam][t, k - 1] = flow.astype(np.float16)

    for cam in task["cam_keys"]:
        np.save(ep_dir / f"{cam}.npy", flows[cam])
    np.save(ep_dir / "valid.npy", valid)
    return {"episode": entry["episode_index"], "num_pairs": len(pairs)}


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
        remaining = extra.max_frames
        for entry in entries:
            take = min(entry["length"], remaining)
            entry["length"] = take
            remaining -= take
            if remaining <= 0:
                entries = [e for e in entries if e["length"] > 0]
                break

    tasks = [
        {
            "root": str(local_root),
            "entry": entry,
            "cam_keys": cam_keys,
            "num_flow_steps": model_flow.num_flow_steps,
            "flow_stride_frames": model_flow.flow_stride_frames,
            "flow_cache_dir": str(cache_dir),
            "sea_raft_ckpt": flow_cfg.sea_raft_ckpt,
            "sea_raft_device": flow_cfg.sea_raft_device,
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

    meta = {
        "num_flow_steps": model_flow.num_flow_steps,
        "flow_stride_frames": model_flow.flow_stride_frames,
        "image_size": list(model_flow.flow_image_size),
        "fps": info["fps"],
        "sea_raft_ckpt": flow_cfg.sea_raft_ckpt,
        "sea_raft_variant": "M",
        "episodes": {str(r["episode"]): r["num_pairs"] for r in results},
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote flow cache to {cache_dir}: {sum(r['num_pairs'] for r in results)} flow pairs.")


if __name__ == "__main__":
    main()
