from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from graph_data import MAX_CHARACTERS, MovieData, MovieValidationError, load_movie_json


def load_config(path: str | Path) -> dict[str, Any]:
  with Path(path).open('r', encoding='utf-8') as handle:
    return json.load(handle)


def safe_movie_id(index: int, movie: MovieData) -> str:
  stem = Path(movie.source_path).stem.lower()
  stem = re.sub(r'[^a-z0-9]+', '-', stem).strip('-') or 'movie'
  digest = hashlib.sha1(movie.source_path.encode('utf-8')).hexdigest()[:8]
  return f'{index:04d}-{stem}-{digest}'


def split_movies(
  count: int,
  seed: int,
  test_fraction: float,
  validation_fraction_of_development: float,
) -> list[str]:
  if count < 3:
    raise ValueError('At least three valid movies are required for train/validation/test splits.')

  indices = list(range(count))
  random.Random(seed).shuffle(indices)

  test_count = int(round(count * test_fraction))
  test_count = min(max(1, test_count), count - 2)
  development_count = count - test_count
  validation_count = int(round(development_count * validation_fraction_of_development))
  validation_count = min(max(1, validation_count), development_count - 1)

  split_by_index = ['train'] * count
  for index in indices[:test_count]:
    split_by_index[index] = 'test'
  for index in indices[test_count:test_count + validation_count]:
    split_by_index[index] = 'validation'
  return split_by_index


def main() -> None:
  parser = argparse.ArgumentParser(
    description='Validate movie JSON files and create scene-local adjacency datasets.'
  )
  parser.add_argument('--input-dir', required=True)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--config', default='config.json')
  parser.add_argument('--recursive', action='store_true')
  args = parser.parse_args()

  config = load_config(args.config)
  data_config = config['data']
  input_dir = Path(args.input_dir)
  output_dir = Path(args.output_dir)
  pattern = '**/*.json' if args.recursive else '*.json'
  source_paths = sorted(input_dir.glob(pattern))
  if not source_paths:
    raise SystemExit(f'No JSON files were found in {input_dir}.')

  movies: list[MovieData] = []
  errors: list[str] = []
  over_limit: list[str] = []
  for source_path in source_paths:
    try:
      movie = load_movie_json(
        source_path,
        max_characters=int(data_config['max_characters']),
      )
      movies.append(movie)
    except MovieValidationError as exc:
      message = str(exc)
      errors.append(message)
      if 'exceeding the limit' in message:
        over_limit.append(message)

  if errors:
    print('Dataset validation failed. No processed dataset was written.', file=sys.stderr)
    if over_limit:
      print('\nMovies exceeding the 256-character limit:', file=sys.stderr)
      for message in over_limit:
        print(f'  - {message}', file=sys.stderr)
    other_errors = [message for message in errors if message not in over_limit]
    if other_errors:
      print('\nOther validation errors:', file=sys.stderr)
      for message in other_errors:
        print(f'  - {message}', file=sys.stderr)
    raise SystemExit(2)

  splits = split_movies(
    count=len(movies),
    seed=int(config['seed']),
    test_fraction=float(data_config['test_fraction']),
    validation_fraction_of_development=float(
      data_config['validation_fraction_of_development']
    ),
  )

  if output_dir.exists():
    shutil.rmtree(output_dir)
  (output_dir / 'movies').mkdir(parents=True, exist_ok=True)

  window_size = int(data_config['window_size'])
  records: list[dict[str, Any]] = []
  split_positive_counts = {'train': 0, 'validation': 0, 'test': 0}
  split_total_counts = {'train': 0, 'validation': 0, 'test': 0}
  train_positive_by_coordinate = np.zeros(
    movies[0].scene_vectors.shape[1],
    dtype=np.int64,
  )
  train_target_count = 0

  for index, (movie, split) in enumerate(zip(movies, splits)):
    movie_id = safe_movie_id(index, movie)
    data_relative = Path('movies') / f'{movie_id}.npz'
    metadata_relative = Path('movies') / f'{movie_id}.metadata.json'

    np.savez_compressed(
      output_dir / data_relative,
      scene_vectors=movie.scene_vectors,
    )
    metadata = {
      'movie_id': movie_id,
      'title': movie.title,
      'source_path': movie.source_path,
      'character_names': movie.character_names,
      'scene_headings': movie.scene_headings,
    }
    with (output_dir / metadata_relative).open('w', encoding='utf-8') as handle:
      json.dump(metadata, handle, indent=2, ensure_ascii=False)

    targets = movie.scene_vectors[window_size:]
    positive_count = int(targets.sum(dtype=np.int64))
    total_count = int(targets.size)
    split_positive_counts[split] += positive_count
    split_total_counts[split] += total_count
    if split == 'train' and len(targets) > 0:
      train_positive_by_coordinate += targets.sum(axis=0, dtype=np.int64)
      train_target_count += len(targets)

    records.append({
      'movie_id': movie_id,
      'title': movie.title,
      'source_path': movie.source_path,
      'data_path': str(data_relative),
      'metadata_path': str(metadata_relative),
      'n_characters': movie.n_characters,
      'n_scenes': movie.n_scenes,
      'n_samples': max(0, movie.n_scenes - window_size),
      'split': split,
    })

  if train_target_count > 0:
    global_frequency = train_positive_by_coordinate / train_target_count
  else:
    global_frequency = np.zeros_like(train_positive_by_coordinate, dtype=np.float64)
  np.save(output_dir / 'global_frequency.npy', global_frequency.astype(np.float32))

  manifest = {
    'format_version': 1,
    'max_characters': MAX_CHARACTERS,
    'packed_size': int(movies[0].scene_vectors.shape[1]),
    'window_size': window_size,
    'seed': int(config['seed']),
    'split_positive_counts': split_positive_counts,
    'split_total_counts': split_total_counts,
    'movies': records,
  }
  with (output_dir / 'manifest.json').open('w', encoding='utf-8') as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)

  split_movie_counts = {
    split: sum(record['split'] == split for record in records)
    for split in ('train', 'validation', 'test')
  }
  split_sample_counts = {
    split: sum(
      record['n_samples'] for record in records if record['split'] == split
    )
    for split in ('train', 'validation', 'test')
  }
  print(f'Processed {len(records)} movies into {output_dir}.')
  print(f'Movie counts: {split_movie_counts}')
  print(f'Window counts: {split_sample_counts}')
  short_movies = [record for record in records if record['n_samples'] == 0]
  if short_movies:
    print(
      f'Warning: {len(short_movies)} movies contain at most {window_size} scenes '
      'and contribute no training windows.'
    )


if __name__ == '__main__':
  main()
