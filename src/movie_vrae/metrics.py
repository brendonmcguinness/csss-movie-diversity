from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class StreamingBinaryMetrics:
  bins: int = 2048
  positive_histogram: np.ndarray = field(init=False)
  negative_histogram: np.ndarray = field(init=False)

  def __post_init__(self) -> None:
    if self.bins < 16:
      raise ValueError('bins must be at least 16.')
    self.positive_histogram = np.zeros(self.bins, dtype=np.int64)
    self.negative_histogram = np.zeros(self.bins, dtype=np.int64)

  def update(
    self,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
  ) -> None:
    probabilities = probabilities.detach().cpu()
    targets = targets.detach().cpu().bool()
    if mask is not None:
      mask = mask.detach().cpu().bool()
      probabilities = probabilities[:, mask]
      targets = targets[:, mask]

    probabilities = probabilities.reshape(-1).clamp(0.0, 1.0)
    targets = targets.reshape(-1)
    bin_indices = torch.clamp(
      (probabilities * (self.bins - 1)).long(),
      min=0,
      max=self.bins - 1,
    )
    positive_counts = torch.bincount(
      bin_indices[targets],
      minlength=self.bins,
    ).numpy()
    negative_counts = torch.bincount(
      bin_indices[~targets],
      minlength=self.bins,
    ).numpy()
    self.positive_histogram += positive_counts
    self.negative_histogram += negative_counts

  def compute(self, threshold: float | None = None) -> dict[str, float]:
    positives_desc = self.positive_histogram[::-1]
    negatives_desc = self.negative_histogram[::-1]
    true_positives = np.cumsum(positives_desc, dtype=np.float64)
    false_positives = np.cumsum(negatives_desc, dtype=np.float64)
    total_positives = float(self.positive_histogram.sum())
    total_negatives = float(self.negative_histogram.sum())

    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    if total_positives > 0:
      recall = true_positives / total_positives
    else:
      recall = np.zeros_like(true_positives)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)

    recall_increments = np.diff(np.concatenate(([0.0], recall)))
    average_precision = float(np.sum(precision * recall_increments))
    thresholds_desc = np.arange(self.bins - 1, -1, -1) / (self.bins - 1)

    if threshold is None:
      selected_index = int(np.argmax(f1)) if len(f1) else 0
    else:
      selected_index = int(np.argmin(np.abs(thresholds_desc - threshold)))

    selected_tp = true_positives[selected_index] if len(true_positives) else 0.0
    selected_fp = false_positives[selected_index] if len(false_positives) else 0.0
    selected_fn = max(0.0, total_positives - selected_tp)
    selected_tn = max(0.0, total_negatives - selected_fp)
    accuracy = (selected_tp + selected_tn) / max(
      total_positives + total_negatives,
      1.0,
    )

    return {
      'average_precision_approx': average_precision,
      'threshold': float(thresholds_desc[selected_index]),
      'precision': float(precision[selected_index]),
      'recall': float(recall[selected_index]),
      'f1': float(f1[selected_index]),
      'accuracy': float(accuracy),
      'positive_count': int(total_positives),
      'negative_count': int(total_negatives),
    }


@dataclass
class TopKMetrics:
  k_values: tuple[int, ...] = (10, 25, 50)
  precision_sums: dict[int, float] = field(init=False)
  recall_sums: dict[int, float] = field(init=False)
  sample_counts: dict[int, int] = field(init=False)

  def __post_init__(self) -> None:
    self.precision_sums = {k: 0.0 for k in self.k_values}
    self.recall_sums = {k: 0.0 for k in self.k_values}
    self.sample_counts = {k: 0 for k in self.k_values}

  def update(
    self,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
  ) -> None:
    probabilities = probabilities.detach().cpu()
    targets = targets.detach().cpu().bool()
    if mask is not None:
      mask = mask.detach().cpu().bool()
      probabilities = probabilities[:, mask]
      targets = targets[:, mask]

    for row_probabilities, row_targets in zip(probabilities, targets):
      positive_count = int(row_targets.sum())
      if positive_count == 0:
        continue
      for k in self.k_values:
        effective_k = min(k, row_probabilities.numel())
        top_indices = torch.topk(row_probabilities, effective_k).indices
        hits = int(row_targets[top_indices].sum())
        self.precision_sums[k] += hits / effective_k
        self.recall_sums[k] += hits / positive_count
        self.sample_counts[k] += 1

  def compute(self) -> dict[str, float]:
    output: dict[str, float] = {}
    for k in self.k_values:
      count = max(1, self.sample_counts[k])
      output[f'precision_at_{k}'] = self.precision_sums[k] / count
      output[f'recall_at_{k}'] = self.recall_sums[k] / count
      output[f'samples_with_positives_at_{k}'] = self.sample_counts[k]
    return output
