from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from graph_data import (
  DIAGONAL_MASK,
  OFF_DIAGONAL_MASK,
  UPPER_COLS,
  UPPER_ROWS,
  load_movie_json,
  unpack_upper_triangle,
)
from model import VRAENextScene


def build_model(checkpoint: dict[str, Any], device: torch.device) -> VRAENextScene:
  model_config = checkpoint['config']['model']
  model = VRAENextScene(
    input_size=int(checkpoint['input_size']),
    embedding_size=int(model_config['embedding_size']),
    hidden_size=int(model_config['hidden_size']),
    latent_size=int(model_config['latent_size']),
    num_layers=int(model_config['num_layers']),
    dropout=float(model_config['dropout']),
  ).to(device)
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  return model


def ranked_entries(
  packed_probabilities: np.ndarray,
  character_names: list[str],
  mask: np.ndarray,
  limit: int,
) -> list[dict[str, Any]]:
  valid = mask & (UPPER_ROWS < len(character_names)) & (UPPER_COLS < len(character_names))
  packed_indices = np.flatnonzero(valid)
  order = packed_indices[np.argsort(packed_probabilities[packed_indices])[::-1]]
  output: list[dict[str, Any]] = []
  for packed_index in order[:limit]:
    first = int(UPPER_ROWS[packed_index])
    second = int(UPPER_COLS[packed_index])
    output.append({
      'character_i': first,
      'character_j': second,
      'name_i': character_names[first],
      'name_j': character_names[second],
      'probability': float(packed_probabilities[packed_index]),
    })
  return output


def main() -> None:
  parser = argparse.ArgumentParser(description='Predict the next scene-local interaction matrix.')
  parser.add_argument('--movie-json', required=True)
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--output-prefix', default='next_scene_prediction')
  parser.add_argument(
    '--last-observed-scene',
    type=int,
    default=None,
    help='Zero-based final observed scene. Defaults to the final scene in the JSON file.',
  )
  parser.add_argument('--top', type=int, default=25)
  args = parser.parse_args()

  device = torch.device('cpu')
  checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
  config = checkpoint['config']
  torch.set_num_threads(int(config['training']['cpu_threads']))
  model = build_model(checkpoint, device)
  movie = load_movie_json(args.movie_json)
  window_size = int(config['data']['window_size'])

  last_observed = (
    movie.n_scenes - 1
    if args.last_observed_scene is None
    else args.last_observed_scene
  )
  if last_observed < window_size - 1 or last_observed >= movie.n_scenes:
    raise SystemExit(
      f'last-observed-scene must be between {window_size - 1} and '
      f'{movie.n_scenes - 1}.'
    )
  first_observed = last_observed - window_size + 1
  inputs = torch.from_numpy(
    movie.scene_vectors[first_observed:last_observed + 1]
      .astype(np.float32, copy=False)
  ).unsqueeze(0).to(device)

  with torch.no_grad():
    probabilities = model.predict_proba(inputs)[0].cpu().numpy()

  threshold = float(checkpoint['validation_thresholds']['all'])
  binary_packed = (probabilities >= threshold).astype(np.uint8)
  probability_matrix = unpack_upper_triangle(probabilities)
  binary_matrix = unpack_upper_triangle(binary_packed)

  output_prefix = Path(args.output_prefix)
  np.savez_compressed(
    output_prefix.with_suffix('.npz'),
    packed_probabilities=probabilities,
    probability_matrix=probability_matrix,
    packed_binary=binary_packed,
    binary_matrix=binary_matrix,
    threshold=np.array(threshold, dtype=np.float32),
  )
  summary = {
    'title': movie.title,
    'source_path': movie.source_path,
    'observed_scene_indices': list(range(first_observed, last_observed + 1)),
    'predicted_scene_index': last_observed + 1,
    'threshold': threshold,
    'top_interactions': ranked_entries(
      probabilities,
      movie.character_names,
      OFF_DIAGONAL_MASK,
      args.top,
    ),
    'top_character_appearances': ranked_entries(
      probabilities,
      movie.character_names,
      DIAGONAL_MASK,
      args.top,
    ),
  }
  with output_prefix.with_suffix('.json').open('w', encoding='utf-8') as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)
  print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
  main()
