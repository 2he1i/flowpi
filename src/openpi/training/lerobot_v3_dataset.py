"""Minimal read-only LeRobot v3.0 dataset loader.

The `lerobot` package pinned by openpi (0.1.0, codebase v2.1) cannot read v3.0 datasets
(e.g. `data/adjust_bottle_ep0`: parquet-embedded PNG images, `meta/tasks.parquet`,
`meta/episodes/chunk-*/file-*.parquet`). This module implements the small subset of the
`LeRobotDataset` interface that openpi's data pipeline needs:

- random access via `__getitem__` / `__len__`
- `delta_timestamps` per key (stacked along a leading axis, matching upstream semantics,
  including clamping to episode bounds and `{key}_is_pad` flags)
- `.meta.tasks`, `.fps`, `.camera_keys`
- `episode_frames(episode_index)` for the flow cache precomputation (single indexing
  source of truth shared with `__getitem__`)

Images are returned as `uint8` numpy arrays in `[channel, height, width]` layout (stacked to
`[T, channel, height, width]` when `delta_timestamps` are requested), matching the layout that
the upstream LeRobot dataset returns and that openpi's transforms expect.

Indexing model (v3.0 layout): every data parquet file (`data/chunk-*/file-*.parquet`) holds a
*contiguous* slice of the global dataset index space, and its `index` column stores the global
index of each row. Episodes reference their first global index via `dataset_from_index`
(metadata only — it is NOT a row offset inside a parquet file when a file spans several
episodes or starts past global index 0). All row lookups therefore go through a per-file
`dataset_start` map (derived from the `index` column, validated contiguous) and index the local
table as `global_row - dataset_start`. This module is the single source of truth for that
mapping; `scripts/precompute_flow_cache.py` reuses `episode_frames()` instead of re-deriving
the episode-to-file-row relationship.
"""

from collections.abc import Sequence
import dataclasses
import io
import itertools
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
    dataset_to_index: int


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
        # Global (from, to) index range per episode, in the same order as `_episodes`.
        self._ep_bounds = []
        start = 0
        for ep in self._episodes:
            self._ep_bounds.append((start, start + ep.length))
            start += ep.length
        self._num_frames = start

        self._parquet_files: dict[tuple[int, int], pq.ParquetFile] = {}
        self._tables: dict[tuple[int, int], pa.Table] = {}
        # Dataset global-index range [start, end) covered by each data parquet file.
        # `end` may be None until the file is opened (derived from the `index` column).
        self._file_bounds: dict[tuple[int, int], tuple[int, int | None]] = {}
        self._image_cache: dict[tuple[str, int], np.ndarray] = {}
        self._image_cache_limit = 4096

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def camera_keys(self) -> list[str]:
        return self.meta.camera_keys

    @property
    def episodes(self) -> list[_EpisodeEntry]:
        """Episode metadata, sorted by `dataset_from_index` (global index order)."""
        return list(self._episodes)

    def _load_episodes(self) -> list[_EpisodeEntry]:
        episodes = []
        episodes_dir = self.root / "meta" / "episodes"
        for path in sorted(episodes_dir.rglob("*.parquet")):
            table = pq.read_table(path)
            columns = table.column_names
            chunk_col = "data/chunk_index" if "data/chunk_index" in columns else "chunk_index"
            file_col = "data/file_index" if "data/file_index" in columns else "file_index"
            from_col = "dataset_from_index" if "dataset_from_index" in columns else None
            to_col = "dataset_to_index" if "dataset_to_index" in columns else None
            for row in range(table.num_rows):
                ep_idx = int(table.column("episode_index")[row].as_py())
                length = int(table.column("length")[row].as_py())
                chunk_index = int(table.column(chunk_col)[row].as_py())
                file_index = int(table.column(file_col)[row].as_py())
                dataset_from = int(table.column(from_col)[row].as_py()) if from_col else 0
                dataset_to = int(table.column(to_col)[row].as_py()) if to_col else dataset_from + length
                if dataset_to != dataset_from + length:
                    raise ValueError(
                        f"Episode {ep_idx}: dataset_to_index {dataset_to} != dataset_from_index {dataset_from} "
                        f"+ length {length}"
                    )
                episodes.append(
                    _EpisodeEntry(
                        episode_index=ep_idx,
                        length=length,
                        chunk_index=chunk_index,
                        file_index=file_index,
                        dataset_from_index=dataset_from,
                        dataset_to_index=dataset_to,
                    )
                )
        if not episodes:
            raise ValueError(f"No episode metadata parquet files found under {episodes_dir}")
        episodes.sort(key=lambda e: e.dataset_from_index)
        return episodes

    def __len__(self) -> int:
        return self._num_frames

    def _locate(self, idx: int) -> tuple[_EpisodeEntry, int, int]:
        """Map a global dataset index to (episode, local_frame_index, global_index).

        The returned global index is what episode metadata calls `dataset_from_index + local`;
        it is a *global* index, NOT a row offset into the episode's parquet file.
        """
        for ep, (start, end) in zip(self._episodes, self._ep_bounds, strict=True):
            if start <= idx < end:
                return ep, idx - start, ep.dataset_from_index + (idx - start)
        raise IndexError(f"Index {idx} out of range for dataset with {self._num_frames} frames.")

    def _parquet_table(self, chunk_index: int, file_index: int) -> pa.Table:
        key = (chunk_index, file_index)
        if key not in self._tables:
            path = self.root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
            pf = pq.ParquetFile(path)
            self._parquet_files[key] = pf
            table = pf.read()
            self._tables[key] = table
            self._file_bounds[key] = self._derive_file_bounds(key, table)
        return self._tables[key]

    def _derive_file_bounds(self, key: tuple[int, int], table: pa.Table) -> tuple[int, int]:
        """Dataset global-index range [start, end) covered by a data parquet file.

        Ground truth is the `index` column of the data file (the global dataset index of each
        row); it must be contiguous and row-aligned. When the column is absent, falls back to
        the episode metadata: the file's rows are ordered by global index, so the range is
        `[min dataset_from_index, max dataset_to_index)` over the episodes stored in the file,
        and the episodes must tile the range without gaps.
        """
        if "index" in table.column_names:
            index = np.asarray(table.column("index").to_numpy())
            start = int(index[0])
            if index.shape != (table.num_rows,) or not np.array_equal(index, np.arange(start, start + table.num_rows)):
                raise ValueError(
                    f"Data file chunk-{key[0]:03d}/file-{key[1]:03d}.parquet has a non-contiguous 'index' "
                    "column: rows are not aligned with the global dataset index."
                )
            return (start, start + table.num_rows)

        in_file = [e for e in self._episodes if (e.chunk_index, e.file_index) == key]
        if not in_file:
            raise ValueError(f"No episode metadata references chunk-{key[0]:03d}/file-{key[1]:03d}.parquet")
        in_file.sort(key=lambda e: e.dataset_from_index)
        start = in_file[0].dataset_from_index
        for prev, cur in itertools.pairwise(in_file):
            if prev.dataset_to_index != cur.dataset_from_index:
                raise ValueError(
                    f"Episodes sharing chunk-{key[0]:03d}/file-{key[1]:03d}.parquet do not tile the global "
                    f"index range contiguously: episode {prev.episode_index} ends at "
                    f"{prev.dataset_to_index} but episode {cur.episode_index} starts at "
                    f"{cur.dataset_from_index}."
                )
        end = in_file[-1].dataset_to_index
        return (start, end)

    def _file_bounds_for(self, chunk_index: int, file_index: int) -> tuple[int, int]:
        """Dataset global-index range of a data parquet file (opening it if necessary)."""
        key = (chunk_index, file_index)
        if key not in self._file_bounds:
            self._parquet_table(chunk_index, file_index)
        bounds = self._file_bounds[key]
        assert bounds[1] is not None
        return bounds

    def _table_and_local_row(self, global_row: int) -> tuple[pa.Table, int]:
        """Map a global dataset index to (parquet table, row index inside that table)."""
        ep, _, _ = self._locate(global_row)
        table = self._parquet_table(ep.chunk_index, ep.file_index)
        start, _ = self._file_bounds_for(ep.chunk_index, ep.file_index)
        local = global_row - start
        if not 0 <= local < table.num_rows:
            raise IndexError(
                f"Global index {global_row} maps outside "
                f"chunk-{ep.chunk_index:03d}/file-{ep.file_index:03d}.parquet "
                f"(dataset range [{start}, {start + table.num_rows})). Episode metadata and data files "
                "are inconsistent."
            )
        return table, local

    def _read_column_rows(self, column: str, rows: Sequence[int]):
        """Read `rows` (global dataset indices) from their parquet tables."""
        table, _ = self._table_and_local_row(rows[0])
        for row in rows[1:]:
            other, _ = self._table_and_local_row(row)
            if other is not table:
                raise ValueError("Cross-file queries are not supported for this simple dataset reader.")
        return [table.column(column)[self._table_and_local_row(row)[1]].as_py() for row in rows]

    def _global_columns(self) -> list[str]:
        return [k for k, ft in self.meta.features.items() if ft.get("dtype") not in ("image", "video")]

    def _decode_image(self, key: str, row: int) -> np.ndarray:
        """Decode one image at global dataset index `row`."""
        cache_key = (key, row)
        cached = self._image_cache.get(cache_key)
        if cached is not None:
            return cached
        table, local = self._table_and_local_row(row)
        item = table.column(key)[local].as_py()
        arr = self._decode_image_item(item)
        if len(self._image_cache) >= self._image_cache_limit:
            self._image_cache.clear()
        self._image_cache[cache_key] = arr
        return arr

    @staticmethod
    def _decode_image_item(item) -> np.ndarray:
        """Decode a parquet image cell (dict with 'bytes') to a [C, H, W] uint8 array."""
        img = Image.open(io.BytesIO(item["bytes"]))
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return np.transpose(arr, (2, 0, 1)).copy()  # HWC -> CHW

    def episode_frames(self, episode_index: int) -> dict[str, np.ndarray]:
        """Decode every camera frame of one episode -> {cam: [T, 3, H, W] uint8}.

        Shares the episode -> parquet-row mapping with `__getitem__` (this class is the single
        indexing source of truth): only the episode's local rows are decoded, never the whole
        file, so episodes sharing a parquet file or crossing file boundaries stay correct.
        """
        for ep in self._episodes:
            if ep.episode_index == episode_index:
                break
        else:
            raise IndexError(f"No episode with index {episode_index} in this dataset.")
        table = self._parquet_table(ep.chunk_index, ep.file_index)
        start, _ = self._file_bounds_for(ep.chunk_index, ep.file_index)
        local_start = ep.dataset_from_index - start
        local_end = ep.dataset_to_index - start
        out = {}
        for cam in self.camera_keys:
            frames = [
                self._decode_image_item(table.column(cam)[local].as_py()) for local in range(local_start, local_end)
            ]
            out[cam] = np.stack(frames, axis=0)
        return out

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        for ep, (ep_start, ep_end) in zip(self._episodes, self._ep_bounds, strict=True):  # noqa: B007
            if ep_start <= idx < ep_end:
                break
        else:
            raise IndexError(f"Index {idx} out of range for dataset with {self._num_frames} frames.")
        row = ep.dataset_from_index + (idx - ep_start)  # global dataset index

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
