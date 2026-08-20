from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReconAccumulators:
    n: int = 0
    sum_err2: float = 0.0
    sum_x: float = 0.0
    sum_x2: float = 0.0
    sum_cosine: float = 0.0
    sum_rel_l2: float = 0.0
    sum_l0: float = 0.0
    n_tokens: int = 0

    def update(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        feature_acts: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> None:
        """Accumulate metrics over masked tokens. Tensors are [batch, seq, dim]."""
        mask = token_mask.bool()
        if mask.ndim == x.ndim:
            pass
        elif mask.ndim == x.ndim - 1:
            mask = mask.unsqueeze(-1)
        else:
            raise ValueError(f"mask ndim {mask.ndim} incompatible with x {tuple(x.shape)}")

        x32 = x.detach().float()
        hat32 = x_hat.detach().float()
        acts = feature_acts.detach()
        keep = mask.expand_as(x32)[..., 0]

        if keep.sum() == 0:
            return

        x_tok = x32[keep]
        hat_tok = hat32[keep]
        acts_tok = acts[keep]

        err = x_tok - hat_tok
        n = x_tok.numel()
        self.n += int(n)
        self.n_tokens += int(keep.sum().item())
        self.sum_err2 += float(err.square().sum().item())
        self.sum_x += float(x_tok.sum().item())
        self.sum_x2 += float(x_tok.square().sum().item())

        x_norm = x_tok.norm(dim=-1).clamp_min(1e-12)
        hat_norm = hat_tok.norm(dim=-1).clamp_min(1e-12)
        cosine = (x_tok * hat_tok).sum(dim=-1) / (x_norm * hat_norm)
        rel_l2 = (x_tok - hat_tok).norm(dim=-1) / x_norm
        self.sum_cosine += float(cosine.sum().item())
        self.sum_rel_l2 += float(rel_l2.sum().item())
        self.sum_l0 += float((acts_tok != 0).sum(dim=-1).float().sum().item())

    def finalize(self) -> dict[str, float | int]:
        if self.n == 0 or self.n_tokens == 0:
            raise RuntimeError("no tokens accumulated")
        mse = self.sum_err2 / self.n
        mean = self.sum_x / self.n
        var = self.sum_x2 / self.n - mean * mean
        fvu = mse / max(var, 1e-12)
        return {
            "n_elements": self.n,
            "n_tokens": self.n_tokens,
            "mse": mse,
            "fvu": fvu,
            "mean_activation": mean,
            "activation_var": var,
            "cosine": self.sum_cosine / self.n_tokens,
            "relative_l2": self.sum_rel_l2 / self.n_tokens,
            "l0": self.sum_l0 / self.n_tokens,
        }
