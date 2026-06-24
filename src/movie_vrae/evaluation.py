from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from graph_data import DIAGONAL_MASK, OFF_DIAGONAL_MASK
from metrics import StreamingBinaryMetrics, TopKMetrics


ProbabilityFunction = Callable[[dict[str, Any]], torch.Tensor]


def _make_masks(device: torch.device) -> dict[str, torch.Tensor | None]:
  return {
    'all': None,
    'diagonal': torch.from_numpy(DIAGONAL_MASK).to(device=device),
    'edges': torch.from_numpy(OFF_DIAGONAL_MASK).to(device=device),
  }


@torch.no_grad()
def evaluate_probability_function(
  loader: DataLoader,
  probability_function: ProbabilityFunction,
  device: torch.device,
  bins: int = 2048,
  top_k: tuple[int, ...] = (10, 25, 50),
  thresholds: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
  masks = _make_masks(device)
  streaming = {
    name: StreamingBinaryMetrics(bins=bins)
    for name in masks
  }
  topk = {
    name: TopKMetrics(k_values=top_k)
    for name in masks
  }

  for batch in loader:
    batch['inputs'] = batch['inputs'].to(device=device, dtype=torch.float32)
    batch['target'] = batch['target'].to(device=device, dtype=torch.float32)
    probabilities = probability_function(batch).to(device=device)

    for name, mask in masks.items():
      streaming[name].update(probabilities, batch['target'], mask=mask)
      topk[name].update(probabilities, batch['target'], mask=mask)

  results: dict[str, dict[str, float]] = {}
  for name in masks:
    threshold = None if thresholds is None else thresholds.get(name)
    results[name] = streaming[name].compute(threshold=threshold)
    results[name].update(topk[name].compute())
  return results


@torch.no_grad()
def evaluate_model(
  model: torch.nn.Module,
  loader: DataLoader,
  device: torch.device,
  bins: int = 2048,
  top_k: tuple[int, ...] = (10, 25, 50),
  thresholds: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
  model.eval()

  def probability_function(batch: dict[str, Any]) -> torch.Tensor:
    return model.predict_proba(batch['inputs'])

  return evaluate_probability_function(
    loader=loader,
    probability_function=probability_function,
    device=device,
    bins=bins,
    top_k=top_k,
    thresholds=thresholds,
  )


def thresholds_from_metrics(
  metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
  return {
    name: float(values['threshold'])
    for name, values in metrics.items()
  }


def make_baseline_probability_function(
  baseline: str,
  global_frequency: np.ndarray | None = None,
) -> ProbabilityFunction:
  if baseline == 'all_zero':
    def all_zero(batch: dict[str, Any]) -> torch.Tensor:
      return torch.zeros_like(batch['target'])
    return all_zero

  if baseline == 'repeat_last':
    def repeat_last(batch: dict[str, Any]) -> torch.Tensor:
      return batch['inputs'][:, -1, :]
    return repeat_last

  if baseline == 'recent_union':
    def recent_union(batch: dict[str, Any]) -> torch.Tensor:
      return batch['inputs'].amax(dim=1)
    return recent_union

  if baseline == 'global_frequency':
    if global_frequency is None:
      raise ValueError('global_frequency is required for the global_frequency baseline.')
    frequency_tensor = torch.from_numpy(global_frequency.astype(np.float32))

    def frequency(batch: dict[str, Any]) -> torch.Tensor:
      return frequency_tensor.to(batch['target'].device).expand_as(batch['target'])
    return frequency

  raise ValueError(f'Unknown baseline: {baseline}')
