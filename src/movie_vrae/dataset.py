from __future__ import annotations

import bisect
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class MovieWindowDataset(Dataset[dict[str, Any]]):
  def __init__(
    self,
    processed_dir: str | Path,
    split: str,
    window_size: int | None = None,
    cache_size: int = 8,
  ) -> None:
    self.processed_dir = Path(processed_dir)
    manifest_path = self.processed_dir / 'manifest.json'
    with manifest_path.open('r', encoding='utf-8') as handle:
      self.manifest = json.load(handle)

    manifest_window = int(self.manifest['window_size'])
    self.window_size = manifest_window if window_size is None else int(window_size)
    if self.window_size != manifest_window:
      raise ValueError(
        f'Processed data uses window_size={manifest_window}, but '
        f'window_size={self.window_size} was requested.'
      )

    self.records = [
      record for record in self.manifest['movies'] if record['split'] == split
    ]
    self.split = split
    self._sample_counts = [
      max(0, int(record['n_scenes']) - self.window_size)
      for record in self.records
    ]
    self._cumulative_counts = np.cumsum(self._sample_counts, dtype=np.int64).tolist()

    self._load_scenes = lru_cache(maxsize=cache_size)(self._load_scenes_uncached)

  def __len__(self) -> int:
    if not self._cumulative_counts:
      return 0
    return int(self._cumulative_counts[-1])

  def _load_scenes_uncached(self, relative_path: str) -> np.ndarray:
    path = self.processed_dir / relative_path
    with np.load(path, allow_pickle=False) as archive:
      return archive['scene_vectors'].copy()

  def _resolve_index(self, index: int) -> tuple[int, int]:
    if index < 0:
      index += len(self)
    if index < 0 or index >= len(self):
      raise IndexError(index)

    movie_index = bisect.bisect_right(self._cumulative_counts, index)
    prior_count = 0 if movie_index == 0 else self._cumulative_counts[movie_index - 1]
    start_scene = index - prior_count
    return movie_index, int(start_scene)

  def __getitem__(self, index: int) -> dict[str, Any]:
    movie_index, start_scene = self._resolve_index(index)
    record = self.records[movie_index]
    scenes = self._load_scenes(record['data_path'])
    target_scene = start_scene + self.window_size

    inputs = torch.from_numpy(
      scenes[start_scene:target_scene].astype(np.float32, copy=False)
    )
    target = torch.from_numpy(
      scenes[target_scene].astype(np.float32, copy=False)
    )

    return {
      'inputs': inputs,
      'target': target,
      'movie_id': record['movie_id'],
      'title': record['title'],
      'target_scene_index': target_scene,
    }

  def target_counts(self) -> tuple[int, int]:
    positives = 0
    total = 0
    for record in self.records:
      scenes = self._load_scenes(record['data_path'])
      targets = scenes[self.window_size:]
      positives += int(targets.sum(dtype=np.int64))
      total += int(targets.size)
    return positives, total
