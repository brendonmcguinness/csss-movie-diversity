from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_data import (  # noqa: E402
  MAX_CHARACTERS,
  PACKED_SIZE,
  MovieValidationError,
  load_movie_json,
  pack_upper_triangle,
  unpack_upper_triangle,
)
from model import VRAENextScene  # noqa: E402


def test_pack_round_trip() -> None:
  matrix = np.zeros((MAX_CHARACTERS, MAX_CHARACTERS), dtype=np.uint8)
  matrix[np.ix_([0, 2, 4], [0, 2, 4])] = 1
  packed = pack_upper_triangle(matrix)
  restored = unpack_upper_triangle(packed)
  assert packed.shape == (PACKED_SIZE,)
  np.testing.assert_array_equal(restored, matrix)


def test_scene_clique_and_singleton_diagonal(tmp_path: Path) -> None:
  movie_path = tmp_path / 'movie.json'
  movie_path.write_text(json.dumps({
    'title': 'Synthetic',
    'character_ids': [
      {'name': 'A', 'id': 0},
      {'name': 'B', 'id': 1},
      {'name': 'C', 'id': 2},
    ],
    'scenes': [
      {'heading': 'SCENE 1', 'character_ids': [0, 1, 2]},
      {'heading': 'SCENE 2', 'character_ids': [2]},
    ],
  }), encoding='utf-8')
  movie = load_movie_json(movie_path)
  first = unpack_upper_triangle(movie.scene_vectors[0])
  second = unpack_upper_triangle(movie.scene_vectors[1])
  assert int(first[:3, :3].sum()) == 9
  assert second[2, 2] == 1
  assert int(second.sum()) == 1


def test_more_than_256_characters_is_rejected(tmp_path: Path) -> None:
  movie_path = tmp_path / 'too_many.json'
  movie_path.write_text(json.dumps({
    'title': 'Too Many',
    'character_ids': [
      {'name': f'C{i}', 'id': i}
      for i in range(MAX_CHARACTERS + 1)
    ],
    'scenes': [],
  }), encoding='utf-8')
  with pytest.raises(MovieValidationError, match='exceeding the limit'):
    load_movie_json(movie_path)


def test_model_forward_shape() -> None:
  model = VRAENextScene(
    input_size=PACKED_SIZE,
    embedding_size=16,
    hidden_size=16,
    latent_size=4,
    dropout=0.0,
  )
  inputs = torch.zeros((2, 5, PACKED_SIZE), dtype=torch.float32)
  logits, mu, logvar = model(inputs)
  assert logits.shape == (2, PACKED_SIZE)
  assert mu.shape == (2, 4)
  assert logvar.shape == (2, 4)
