from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from recon.config import SAE_REPO_ID, SAE_TOPK, SAE_WIDTH, sae_id_for_layer


class TopKSAE(torch.nn.Module):
    """Top-K SAE matching Qwen-Scope residual checkpoints."""

    def __init__(
        self,
        w_enc: torch.Tensor,
        b_enc: torch.Tensor,
        w_dec: torch.Tensor,
        b_dec: torch.Tensor,
        k: int,
    ) -> None:
        super().__init__()
        self.W_enc = torch.nn.Parameter(w_enc, requires_grad=False)
        self.b_enc = torch.nn.Parameter(b_enc, requires_grad=False)
        self.W_dec = torch.nn.Parameter(w_dec, requires_grad=False)
        self.b_dec = torch.nn.Parameter(b_dec, requires_grad=False)
        self.k = int(k)
        self.d_sae = int(w_enc.shape[0])
        self.d_in = int(w_enc.shape[1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.linear(x, self.W_enc, self.b_enc)
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, vals)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return F.linear(feature_acts, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        return self.decode(acts), acts


def _orient_enc(w: torch.Tensor, d_in: int, d_sae: int) -> torch.Tensor:
    if w.shape == (d_sae, d_in):
        return w.contiguous()
    if w.shape == (d_in, d_sae):
        return w.T.contiguous()
    raise ValueError(f"W_enc shape {tuple(w.shape)} does not match {(d_sae, d_in)}")


def _orient_dec(w: torch.Tensor, d_in: int, d_sae: int) -> torch.Tensor:
    if w.shape == (d_in, d_sae):
        return w.contiguous()
    if w.shape == (d_sae, d_in):
        return w.T.contiguous()
    raise ValueError(f"W_dec shape {tuple(w.shape)} does not match {(d_in, d_sae)}")


def load_topk_from_hf(
    layer: int,
    device: torch.device | str,
    dtype: torch.dtype,
    d_in: int,
) -> TopKSAE:
    filename = sae_id_for_layer(layer)
    path = hf_hub_download(SAE_REPO_ID, filename)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{filename} is not a dict state")

    w_enc = _orient_enc(state["W_enc"], d_in, SAE_WIDTH)
    w_dec = _orient_dec(state["W_dec"], d_in, SAE_WIDTH)
    b_enc = state["b_enc"].reshape(-1)
    b_dec = state["b_dec"].reshape(-1)
    if int(w_enc.shape[0]) != SAE_WIDTH:
        raise ValueError(f"{filename} d_sae={w_enc.shape[0]}, expected {SAE_WIDTH}")
    sae = TopKSAE(
        w_enc.to(dtype=dtype),
        b_enc.to(dtype=dtype),
        w_dec.to(dtype=dtype),
        b_dec.to(dtype=dtype),
        k=SAE_TOPK,
    )
    return sae.to(device).eval()


def encode_decode(sae: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_in = x.to(dtype=next(sae.parameters()).dtype)
    if hasattr(sae, "encode") and hasattr(sae, "decode"):
        acts = sae.encode(x_in)
        if isinstance(acts, tuple):
            acts = acts[0]
        x_hat = sae.decode(acts)
        return x_hat, acts
    x_hat, acts = sae(x_in)
    return x_hat, acts


@dataclass
class LoadedSAE:
    layer: int
    sae: Any
    source: str


def load_sae(layer: int, device: torch.device | str, dtype: torch.dtype, d_in: int) -> LoadedSAE:
    sae = load_topk_from_hf(layer, device, dtype, d_in)
    return LoadedSAE(layer=layer, sae=sae, source="hf_topk")
