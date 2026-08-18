"""Minimal read-only LeRobot v3.0 dataset loader.

The `lerobot` package pinned by openpi (0.1.0, codebase v2.1) cannot read v3.0 datasets
(e.g. `test_data/adjust_bottle_ep0`: parquet-embedded PNG images, `meta/tasks.parquet`,
`meta/episodes/chunk-*/file-*.parquet`). This module implements the small subset of the
`LeRobotDataset` interface that openpi's data pipeline needs:

- random access via `__getitem__` / `__len__`
- `delta_timestamps` per key (stacked along a leading axis, matching upstream semantics,
  including clamping to episode bounds and `{key}_is_pad` flags)
- `.meta.tasks`, `.fps`, `.camera_keys`

Images are returned as `uint8` numpy arrays in `[channel, height, width]` layout (stacked to
`[T, channel, height, width]` when `delta_timestamps` are requested), matching the layout that
the upstream LeRobot dataset returns and that openpi's transforms expect.
"""

from collections.abc import Sequence
import dataclasses
import io
import json
import pathlib

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


@dataclasses.dataclass(frozen=True)
class _EpisodeEntry:
    episode_index: int
    length: int
    chunk_index: int
    file_index: int
    dataset_from_index: int


class _V3Meta:
    def __init__(self, root: pathlib.Path):
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Not a LeRobot v3.0 dataset: missing {info_path}")
        with open(info_path) as f:
            self.info = json.load(f)
        version = self.info.get("codebase_version", "")
        if not version.startswith("v3"):
            raise ValueError(f"LeRobotV3ParquetDataset only supports v3.0 datasets, got codebase_version={version}")
        self.fps = self.info["fps"]
        self.features = self.info["features"]
        self.tasks = self._load_tasks(root)

    @staticmethod
    def _load_tasks(root: pathlib.Path) -> dict[int, str]:
        tasks_path = root / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            return {}
        table = pq.read_table(tasks_path)
        columns = table.column_names
        task_str_col = "__index_level_0__" if "__index_level_0__" in columns else "task"
        idx_col = "task_index" if "task_index" in columns else None
        tasks = {}
        strings = table.column(task_str_col).to_pylist()
        indices = table.column(idx_col).to_pylist() if idx_col is not None else range(len(strings))
        for i, s in zip(indices, strings, strict=True):
            tasks[int(i)] = s
        return tasks

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft.get("dtype") in ("image", "video")]


class LeRobotV3ParquetDataset:
    """Read-only LeRobot v3.0 dataset with `delta_timestamps` support."""

    def __init__(self, root: str | pathlib.Path, delta_timestamps: dict[str, Sequence[float]] | None = None):
        self.root = pathlib.Path(root)
        self.meta = _V3Meta(self.root)
        self.delta_timestamps = {k: list(v) for k, v in (delta_timestamps or {}).items()}
        self._delta_indices = {k: [round(dt * self.meta.fps) for dt in v] for k, v in self.delta_timestamps.items()}

        self._episodes: list[_EpisodeEntry] = self._load_episodes()
        # Global (from, to) index range per episode.
        self._ep_bounds = []
        start = 0
        for ep in self._episodes:
            self._ep_bounds.append((start, start + ep.length))
            start += ep.length
        self._num_frames = start

        self._parquet_files: dict[tuple[int, int], pq.ParquetFile] = {}
        self._tables: dict[tuple[int, int], pa.Table] = {}
        self._image_cache: dict[tuple[str, int], np.ndarray] = {}
        self._image_cache_limit = 4096

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def camera_keys(self) -> list[str]:
        return self.meta.camera_keys

    def _load_episodes(self) -> list[_EpisodeEntry]:
        episodes = []
        episodes_dir = self.root / "meta" / "episodes"
        for path in sorted(episodes_dir.rglob("*.parquet")):
            table = pq.read_table(path)
            columns = table.column_names
            chunk_col = "data/chunk_index" if "data/chunk_index" in columns else "chunk_index"
            file_col = "data/file_index" if "data/file_index" in columns else "file_index"
            from_col = "dataset_from_index" if "dataset_from_index" in columns else None
            for row in range(table.num_rows):
                ep_idx = int(table.column("episode_index")[row].as_py())
                length = int(table.column("length")[row].as_py())
                chunk_index = int(table.column(chunk_col)[row].as_py())
                file_index = int(table.column(file_col)[row].as_py())
                dataset_from = int(table.column(from_col)[row].as_py()) if from_col else 0
                episodes.append(
                    _EpisodeEntry(
                        episode_index=ep_idx,
                        length=length,
                        chunk_index=chunk_index,
                        file_index=file_index,
                        dataset_from_index=dataset_from,
                    )
                )
        if not episodes:
            raise ValueError(f"No episode metadata parquet files found under {episodes_dir}")
        episodes.sort(key=lambda e: e.dataset_from_index)
        return episodes

    def __len__(self) -> int:
        return self._num_frames

    def _locate(self, idx: int) -> tuple[_EpisodeEntry, int, int]:
        """Map a global dataset index to (episode, local_frame_index, row_index_in_parquet)."""
        for ep, (start, end) in zip(self._episodes, self._ep_bounds, strict=True):
            if start <= idx < end:
                local = idx - start
                return ep, local, ep.dataset_from_index + local
        raise IndexError(f"Index {idx} out of range for dataset with {self._num_frames} frames.")

    def _parquet_table(self, chunk_index: int, file_index: int) -> pa.Table:
        key = (chunk_index, file_index)
        if key not in self._tables:
            path = self.root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
            pf = pq.ParquetFile(path)
            self._parquet_files[key] = pf
            self._tables[key] = pf.read()
        return self._tables[key]

    def _read_column_rows(self, column: str, rows: Sequence[int]):
        # Identify the (chunk, file) for a row index via episode bounds. `rows` are row indices in
        # the "datasetFrom" numbering, which matches the parquet row order across files.
        table = None
        for row in rows:
            ep, _, _ = self._locate_row(row)
            t = self._parquet_table(ep.chunk_index, ep.file_index)
            if table is None:
                table = t
            elif t is not table:
                raise ValueError("Cross-file queries are not supported for this simple dataset reader.")
        return [table.column(column)[row].as_py() for row in rows]

    def _locate_row(self, row_index: int) -> tuple[_EpisodeEntry, int, int]:
        for ep, (start, end) in zip(self._episodes, self._ep_bounds, strict=True):
            if start <= row_index < end:
                return ep, row_index - start, row_index
        raise IndexError(f"Row index {row_index} out of range.")

    def _global_columns(self) -> list[str]:
        return [k for k, ft in self.meta.features.items() if ft.get("dtype") not in ("image", "video")]

    def _decode_image(self, key: str, row: int) -> np.ndarray:
        cache_key = (key, row)
        cached = self._image_cache.get(cache_key)
        if cached is not None:
            return cached
        ep, _, _ = self._locate_row(row)
        table = self._parquet_table(ep.chunk_index, ep.file_index)
        item = table.column(key)[row].as_py()
        img = Image.open(io.BytesIO(item["bytes"]))
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = np.transpose(arr, (2, 0, 1)).copy()  # HWC -> CHW
        if len(self._image_cache) >= self._image_cache_limit:
            self._image_cache.clear()
        self._image_cache[cache_key] = arr
        return arr

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        for ep, (ep_start, ep_end) in zip(self._episodes, self._ep_bounds, strict=True):  # noqa: B007
            if ep_start <= idx < ep_end:
                break
        else:
            raise IndexError(f"Index {idx} out of range for dataset with {self._num_frames} frames.")
        row = ep.dataset_from_index + (idx - ep_start)

        item: dict = {}
        for key in self._global_columns():
            if key in self.delta_timestamps:
                rows = []
                pad = []
                for delta in self._delta_indices[key]:
                    q = idx + delta
                    clamped = max(ep_start, min(ep_end - 1, q))
                    rows.append(self._locate(clamped)[2])
                    pad.append(q < ep_start or q >= ep_end)
                vals = self._read_column_rows(key, rows)
                item[key] = np.stack([np.asarray(v, dtype=np.float32) for v in vals], axis=0)
                item[f"{key}_is_pad"] = np.asarray(pad, dtype=bool)
            else:
                val = self._read_column_rows(key, [row])[0]
                ft = self.meta.features[key]
                if ft.get("dtype") in ("float32", "float64", "float"):
                    val = np.asarray(val, dtype=np.float32)
                elif ft.get("dtype") in ("int64", "int32", "int"):
                    val = int(val)
                item[key] = val

        for key in self.camera_keys:
            if key in self.delta_timestamps:
                frames = []
                pad = []
                for delta in self._delta_indices[key]:
                    q = idx + delta
                    clamped = max(ep_start, min(ep_end - 1, q))
                    frames.append(self._decode_image(key, self._locate(clamped)[2]))
                    pad.append(q < ep_start or q >= ep_end)
                item[key] = np.stack(frames, axis=0)
                item[f"{key}_is_pad"] = np.asarray(pad, dtype=bool)
            else:
                item[key] = self._decode_image(key, row)

        item["task"] = self.meta.tasks.get(int(item.get("task_index", -1)), "")
        return item
