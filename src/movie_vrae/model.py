from __future__ import annotations

import torch
from torch import nn


class VRAENextScene(nn.Module):
  def __init__(
    self,
    input_size: int,
    embedding_size: int = 128,
    hidden_size: int = 128,
    latent_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    if num_layers < 1:
      raise ValueError('num_layers must be at least 1.')

    recurrent_dropout = dropout if num_layers > 1 else 0.0
    self.input_size = input_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.latent_size = latent_size
    self.num_layers = num_layers

    self.input_projection = nn.Sequential(
      nn.Linear(input_size, embedding_size),
      nn.LayerNorm(embedding_size),
      nn.GELU(),
      nn.Dropout(dropout),
    )
    self.encoder = nn.LSTM(
      input_size=embedding_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True,
      dropout=recurrent_dropout,
    )
    self.to_mu = nn.Linear(hidden_size, latent_size)
    self.to_logvar = nn.Linear(hidden_size, latent_size)

    self.latent_to_hidden = nn.Linear(latent_size, num_layers * hidden_size)
    self.latent_to_cell = nn.Linear(latent_size, num_layers * hidden_size)
    self.decoder_start = nn.Parameter(torch.zeros(1, 1, embedding_size))
    self.decoder = nn.LSTM(
      input_size=embedding_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True,
      dropout=recurrent_dropout,
    )
    self.output_projection = nn.Linear(hidden_size, input_size)

    nn.init.normal_(self.decoder_start, mean=0.0, std=0.02)

  def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    embedded = self.input_projection(inputs)
    _, (hidden, _) = self.encoder(embedded)
    final_hidden = hidden[-1]
    mu = self.to_mu(final_hidden)
    logvar = self.to_logvar(final_hidden).clamp(min=-12.0, max=12.0)
    return mu, logvar

  @staticmethod
  def reparameterize(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    sample: bool,
  ) -> torch.Tensor:
    if not sample:
      return mu
    standard_deviation = torch.exp(0.5 * logvar)
    noise = torch.randn_like(standard_deviation)
    return mu + noise * standard_deviation

  def decode(self, latent: torch.Tensor) -> torch.Tensor:
    batch_size = latent.shape[0]
    hidden = self.latent_to_hidden(latent).view(
      batch_size,
      self.num_layers,
      self.hidden_size,
    ).transpose(0, 1).contiguous()
    cell = self.latent_to_cell(latent).view(
      batch_size,
      self.num_layers,
      self.hidden_size,
    ).transpose(0, 1).contiguous()
    start = self.decoder_start.expand(batch_size, -1, -1)
    decoded, _ = self.decoder(start, (hidden, cell))
    return self.output_projection(decoded[:, 0, :])

  def forward(
    self,
    inputs: torch.Tensor,
    sample_latent: bool | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sample_latent is None:
      sample_latent = self.training
    mu, logvar = self.encode(inputs)
    latent = self.reparameterize(mu, logvar, sample=sample_latent)
    logits = self.decode(latent)
    return logits, mu, logvar

  @torch.no_grad()
  def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
    logits, _, _ = self.forward(inputs, sample_latent=False)
    return torch.sigmoid(logits)
