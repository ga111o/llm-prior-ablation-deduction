from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("RECON_RESULTS_DIR", ROOT / "results"))

MODEL_ID = os.environ.get("RECON_MODEL_ID", "Qwen/Qwen3.5-27B")
SAE_REPO_ID = os.environ.get("RECON_SAE_REPO", "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50")
SAE_WIDTH = int(os.environ.get("RECON_SAE_WIDTH", "81920"))
SAE_TOPK = int(os.environ.get("RECON_SAE_TOPK", "50"))

# Qwen3.5-27B has 64 layers. Depth uses round(n_layers * depth), same as Gemma.
LAYER_DEPTHS: dict[float, int] = {0.50: 32, 0.85: 54}

D_MODEL = 5120
N_LAYERS = 64
MAX_POSITION = 262_144
SEQ_LEN = int(os.environ.get("RECON_SEQ_LEN", "1024"))
N_SEQ = int(os.environ.get("RECON_N_SEQ", "2048"))
SMOKE_SEQ_LEN = 128
SMOKE_N_SEQ = 8
DATASET_NAME = os.environ.get("RECON_DATASET", "NeelNanda/pile-10k")
DATASET_TEXT_FIELD = os.environ.get("RECON_TEXT_FIELD", "text")

ATTN_IMPLEMENTATION = "sdpa"
COMPUTE_DTYPE = "bfloat16"


def sae_id_for_layer(layer: int) -> str:
    return f"layer{layer}.sae.pt"


def layers_for_sae_at(sae_at: str) -> list[int]:
    if sae_at == "50":
        return [LAYER_DEPTHS[0.50]]
    if sae_at == "85":
        return [LAYER_DEPTHS[0.85]]
    if sae_at == "both":
        return sorted(LAYER_DEPTHS.values())
    raise ValueError(f"unknown --sae-at {sae_at!r}; expected 50, 85, or both")


def result_stem(sae_at: str, smoke: bool) -> str:
    suffix = {"50": "_l50", "85": "_l85", "both": "_both"}[sae_at]
    stem = f"recon_error{suffix}"
    if smoke:
        stem += "_smoke"
    return stem
