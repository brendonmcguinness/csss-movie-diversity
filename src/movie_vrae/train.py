from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset import MovieWindowDataset
from evaluation import evaluate_model, thresholds_from_metrics
from losses import linear_kl_beta, vrae_forecasting_loss
from model import VRAENextScene


def load_config(path: str | Path) -> dict[str, Any]:
  with Path(path).open('r', encoding='utf-8') as handle:
    return json.load(handle)


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def main() -> None:
  parser = argparse.ArgumentParser(description='Train the CPU-only VRAE forecaster.')
  parser.add_argument('--processed-dir', required=True)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--config', default='config.json')
  parser.add_argument('--resume', default=None)
  args = parser.parse_args()

  config = load_config(args.config)
  seed = int(config['seed'])
  set_seed(seed)
  device = torch.device('cpu')
  torch.set_num_threads(int(config['training']['cpu_threads']))

  train_dataset = MovieWindowDataset(args.processed_dir, split='train')
  validation_dataset = MovieWindowDataset(args.processed_dir, split='validation')
  if len(train_dataset) == 0 or len(validation_dataset) == 0:
    raise SystemExit('Training and validation splits must each contain at least one window.')

  loader_options = {
    'batch_size': int(config['training']['batch_size']),
    'num_workers': int(config['training']['num_workers']),
    'pin_memory': False,
  }
  train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
  validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

  model_config = config['model']
  input_size = int(train_dataset.manifest['packed_size'])
  model = VRAENextScene(
    input_size=input_size,
    embedding_size=int(model_config['embedding_size']),
    hidden_size=int(model_config['hidden_size']),
    latent_size=int(model_config['latent_size']),
    num_layers=int(model_config['num_layers']),
    dropout=float(model_config['dropout']),
  ).to(device)

  training_config = config['training']
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(training_config['learning_rate']),
    weight_decay=float(training_config['weight_decay']),
  )

  positives, total = train_dataset.target_counts()
  negatives = total - positives
  if positives <= 0:
    raise SystemExit('The training targets contain no positive adjacency entries.')
  raw_positive_weight = negatives / positives
  capped_positive_weight = min(
    raw_positive_weight,
    float(training_config['max_positive_weight']),
  )
  positive_weight = torch.tensor(capped_positive_weight, device=device)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  log_path = output_dir / 'training_log.jsonl'
  start_epoch = 0
  best_edge_ap = -float('inf')
  epochs_without_improvement = 0

  if args.resume:
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = int(checkpoint['epoch']) + 1
    best_edge_ap = float(checkpoint.get('best_edge_ap', -float('inf')))

  epochs = int(training_config['epochs'])
  for epoch in range(start_epoch, epochs):
    model.train()
    beta = linear_kl_beta(
      epoch=epoch,
      maximum=float(training_config['kl_beta']),
      warmup_epochs=int(training_config['kl_warmup_epochs']),
    )
    sums = {'loss': 0.0, 'prediction_loss': 0.0, 'kl_loss': 0.0}
    batch_count = 0

    for batch in train_loader:
      inputs = batch['inputs'].to(device=device, dtype=torch.float32)
      targets = batch['target'].to(device=device, dtype=torch.float32)
      optimizer.zero_grad(set_to_none=True)
      logits, mu, logvar = model(inputs)
      loss, parts = vrae_forecasting_loss(
        logits=logits,
        targets=targets,
        mu=mu,
        logvar=logvar,
        positive_weight=positive_weight,
        beta=beta,
      )
      loss.backward()
      clip_grad_norm_(model.parameters(), float(training_config['gradient_clip']))
      optimizer.step()

      for key in sums:
        sums[key] += parts[key]
      batch_count += 1

    train_summary = {
      key: value / max(1, batch_count)
      for key, value in sums.items()
    }
    validation_metrics = evaluate_model(
      model=model,
      loader=validation_loader,
      device=device,
      bins=int(config['evaluation']['histogram_bins']),
      top_k=tuple(int(value) for value in config['evaluation']['top_k']),
    )
    thresholds = thresholds_from_metrics(validation_metrics)
    edge_ap = float(validation_metrics['edges']['average_precision_approx'])

    record = {
      'epoch': epoch,
      'beta': beta,
      'train': train_summary,
      'validation': validation_metrics,
      'positive_weight': capped_positive_weight,
    }
    with log_path.open('a', encoding='utf-8') as handle:
      handle.write(json.dumps(record) + '\n')
    print(json.dumps(record, indent=2))

    checkpoint = {
      'epoch': epoch,
      'model_state': model.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'config': config,
      'input_size': input_size,
      'positive_weight': capped_positive_weight,
      'validation_thresholds': thresholds,
      'best_edge_ap': max(best_edge_ap, edge_ap),
      'processed_dir': str(Path(args.processed_dir).resolve()),
    }
    torch.save(checkpoint, output_dir / 'last.pt')

    if edge_ap > best_edge_ap:
      best_edge_ap = edge_ap
      epochs_without_improvement = 0
      checkpoint['best_edge_ap'] = best_edge_ap
      torch.save(checkpoint, output_dir / 'best.pt')
    else:
      epochs_without_improvement += 1

    if epochs_without_improvement >= int(training_config['early_stopping_patience']):
      print('Early stopping triggered.')
      break


if __name__ == '__main__':
  main()
