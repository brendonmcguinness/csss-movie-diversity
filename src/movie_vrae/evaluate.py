from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MovieWindowDataset
from evaluation import (
  evaluate_model,
  evaluate_probability_function,
  make_baseline_probability_function,
  thresholds_from_metrics,
)
from model import VRAENextScene


def make_loader(
  dataset: MovieWindowDataset,
  batch_size: int,
  num_workers: int,
) -> DataLoader:
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,
  )


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
  return model


def main() -> None:
  parser = argparse.ArgumentParser(description='Evaluate the VRAE and baselines.')
  parser.add_argument('--processed-dir', required=True)
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--output', default='evaluation.json')
  args = parser.parse_args()

  device = torch.device('cpu')
  checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
  config = checkpoint['config']
  torch.set_num_threads(int(config['training']['cpu_threads']))

  validation_dataset = MovieWindowDataset(args.processed_dir, split='validation')
  test_dataset = MovieWindowDataset(args.processed_dir, split='test')
  if len(validation_dataset) == 0 or len(test_dataset) == 0:
    raise SystemExit('Validation and test splits must each contain at least one window.')

  batch_size = int(config['training']['batch_size'])
  num_workers = int(config['training']['num_workers'])
  validation_loader = make_loader(validation_dataset, batch_size, num_workers)
  test_loader = make_loader(test_dataset, batch_size, num_workers)
  bins = int(config['evaluation']['histogram_bins'])
  top_k = tuple(int(value) for value in config['evaluation']['top_k'])

  model = build_model(checkpoint, device)
  model_validation = evaluate_model(
    model,
    validation_loader,
    device,
    bins=bins,
    top_k=top_k,
  )
  model_thresholds = checkpoint.get(
    'validation_thresholds',
    thresholds_from_metrics(model_validation),
  )
  model_test = evaluate_model(
    model,
    test_loader,
    device,
    bins=bins,
    top_k=top_k,
    thresholds=model_thresholds,
  )

  global_frequency = np.load(
    Path(args.processed_dir) / 'global_frequency.npy',
    allow_pickle=False,
  )
  baseline_results: dict[str, Any] = {}
  for baseline in ('all_zero', 'repeat_last', 'recent_union', 'global_frequency'):
    probability_function = make_baseline_probability_function(
      baseline,
      global_frequency=global_frequency,
    )
    validation_metrics = evaluate_probability_function(
      loader=validation_loader,
      probability_function=probability_function,
      device=device,
      bins=bins,
      top_k=top_k,
    )
    if baseline == 'global_frequency':
      thresholds = thresholds_from_metrics(validation_metrics)
    else:
      thresholds = {'all': 0.5, 'diagonal': 0.5, 'edges': 0.5}
    test_metrics = evaluate_probability_function(
      loader=test_loader,
      probability_function=probability_function,
      device=device,
      bins=bins,
      top_k=top_k,
      thresholds=thresholds,
    )
    baseline_results[baseline] = {
      'validation': validation_metrics,
      'thresholds': thresholds,
      'test': test_metrics,
    }

  output = {
    'checkpoint': str(Path(args.checkpoint).resolve()),
    'model': {
      'validation': model_validation,
      'thresholds': model_thresholds,
      'test': model_test,
    },
    'baselines': baseline_results,
  }
  output_path = Path(args.output)
  with output_path.open('w', encoding='utf-8') as handle:
    json.dump(output, handle, indent=2)
  print(json.dumps(output, indent=2))


if __name__ == '__main__':
  main()
