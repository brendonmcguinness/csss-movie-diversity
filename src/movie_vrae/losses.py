from __future__ import annotations

import torch
import torch.nn.functional as functional


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
  return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())


def vrae_forecasting_loss(
  logits: torch.Tensor,
  targets: torch.Tensor,
  mu: torch.Tensor,
  logvar: torch.Tensor,
  positive_weight: torch.Tensor,
  beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
  prediction_loss = functional.binary_cross_entropy_with_logits(
    logits,
    targets,
    pos_weight=positive_weight,
    reduction='mean',
  )
  divergence = kl_divergence(mu, logvar)
  total = prediction_loss + beta * divergence
  return total, {
    'loss': float(total.detach().cpu()),
    'prediction_loss': float(prediction_loss.detach().cpu()),
    'kl_loss': float(divergence.detach().cpu()),
    'beta': float(beta),
  }


def linear_kl_beta(epoch: int, maximum: float, warmup_epochs: int) -> float:
  if warmup_epochs <= 0:
    return float(maximum)
  fraction = min(1.0, max(0.0, (epoch + 1) / warmup_epochs))
  return float(maximum * fraction)
