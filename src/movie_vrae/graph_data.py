from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAX_CHARACTERS = 2 ** 8
UPPER_ROWS, UPPER_COLS = np.triu_indices(MAX_CHARACTERS)
PACKED_SIZE = len(UPPER_ROWS)
DIAGONAL_MASK = UPPER_ROWS == UPPER_COLS
OFF_DIAGONAL_MASK = ~DIAGONAL_MASK


class MovieValidationError(ValueError):
  pass


@dataclass(frozen=True)
class MovieData:
  title: str
  character_names: list[str]
  scene_headings: list[str]
  scene_vectors: np.ndarray
  source_path: str

  @property
  def n_characters(self) -> int:
    return len(self.character_names)

  @property
  def n_scenes(self) -> int:
    return int(self.scene_vectors.shape[0])


def pack_upper_triangle(matrix: np.ndarray) -> np.ndarray:
  matrix = np.asarray(matrix)
  expected_shape = (MAX_CHARACTERS, MAX_CHARACTERS)
  if matrix.shape != expected_shape:
    raise ValueError(f'Expected matrix shape {expected_shape}, got {matrix.shape}.')
  if not np.array_equal(matrix, matrix.T):
    raise ValueError('The adjacency matrix must be symmetric.')
  return matrix[UPPER_ROWS, UPPER_COLS]


def unpack_upper_triangle(vector: np.ndarray) -> np.ndarray:
  vector = np.asarray(vector)
  if vector.shape != (PACKED_SIZE,):
    raise ValueError(f'Expected packed vector shape {(PACKED_SIZE,)}, got {vector.shape}.')
  matrix = np.zeros((MAX_CHARACTERS, MAX_CHARACTERS), dtype=vector.dtype)
  matrix[UPPER_ROWS, UPPER_COLS] = vector
  matrix[UPPER_COLS, UPPER_ROWS] = vector
  return matrix


def _require_list(value: Any, field: str, source: Path) -> list[Any]:
  if not isinstance(value, list):
    raise MovieValidationError(f'{source}: {field} must be a list.')
  return value


def load_movie_json(
  path: str | Path,
  max_characters: int = MAX_CHARACTERS,
) -> MovieData:
  source = Path(path)
  try:
    with source.open('r', encoding='utf-8') as handle:
      raw = json.load(handle)
  except (OSError, json.JSONDecodeError) as exc:
    raise MovieValidationError(f'{source}: could not read valid JSON: {exc}') from exc

  if not isinstance(raw, dict):
    raise MovieValidationError(f'{source}: the top-level JSON value must be an object.')

  title = raw.get('title')
  if not isinstance(title, str) or not title.strip():
    raise MovieValidationError(f'{source}: title must be a nonempty string.')

  raw_characters = _require_list(raw.get('character_ids'), 'character_ids', source)
  if len(raw_characters) > max_characters:
    raise MovieValidationError(
      f'{source}: {title!r} has {len(raw_characters)} characters, exceeding '
      f'the limit of {max_characters}.'
    )

  parsed_characters: list[tuple[int, str]] = []
  for position, entry in enumerate(raw_characters):
    if not isinstance(entry, dict):
      raise MovieValidationError(
        f'{source}: character_ids[{position}] must be an object.'
      )
    character_id = entry.get('id')
    name = entry.get('name')
    if not isinstance(character_id, int):
      raise MovieValidationError(
        f'{source}: character_ids[{position}].id must be an integer.'
      )
    if not isinstance(name, str) or not name.strip():
      raise MovieValidationError(
        f'{source}: character_ids[{position}].name must be a nonempty string.'
      )
    parsed_characters.append((character_id, name.strip()))

  parsed_characters.sort(key=lambda item: item[0])
  observed_ids = [character_id for character_id, _ in parsed_characters]
  expected_ids = list(range(len(parsed_characters)))
  if observed_ids != expected_ids:
    raise MovieValidationError(
      f'{source}: character IDs must be exactly 0, 1, ..., n-1. '
      f'Observed IDs begin {observed_ids[:10]}.'
    )
  character_names = [name for _, name in parsed_characters]

  raw_scenes = _require_list(raw.get('scenes'), 'scenes', source)
  scene_vectors = np.zeros((len(raw_scenes), PACKED_SIZE), dtype=np.uint8)
  scene_headings: list[str] = []

  for scene_index, scene in enumerate(raw_scenes):
    if not isinstance(scene, dict):
      raise MovieValidationError(f'{source}: scenes[{scene_index}] must be an object.')
    heading = scene.get('heading')
    if not isinstance(heading, str):
      raise MovieValidationError(
        f'{source}: scenes[{scene_index}].heading must be a string.'
      )
    scene_headings.append(heading.strip())

    raw_ids = _require_list(
      scene.get('character_ids'),
      f'scenes[{scene_index}].character_ids',
      source,
    )
    if not all(isinstance(character_id, int) for character_id in raw_ids):
      raise MovieValidationError(
        f'{source}: scenes[{scene_index}].character_ids must contain integers only.'
      )
    character_ids = sorted(set(raw_ids))
    invalid_ids = [
      character_id
      for character_id in character_ids
      if character_id < 0 or character_id >= len(character_names)
    ]
    if invalid_ids:
      raise MovieValidationError(
        f'{source}: scenes[{scene_index}] contains invalid character IDs '
        f'{invalid_ids[:10]}.'
      )

    if character_ids:
      matrix = np.zeros((MAX_CHARACTERS, MAX_CHARACTERS), dtype=np.uint8)
      matrix[np.ix_(character_ids, character_ids)] = 1
      scene_vectors[scene_index] = pack_upper_triangle(matrix)

  return MovieData(
    title=title.strip(),
    character_names=character_names,
    scene_headings=scene_headings,
    scene_vectors=scene_vectors,
    source_path=str(source.resolve()),
  )
