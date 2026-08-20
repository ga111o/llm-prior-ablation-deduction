from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recon.config import (  # noqa: E402
    D_MODEL,
    LAYER_DEPTHS,
    MODEL_ID,
    N_LAYERS,
    N_SEQ,
    RESULTS_DIR,
    SAE_REPO_ID,
    SAE_TOPK,
    SAE_WIDTH,
    SEQ_LEN,
    SMOKE_N_SEQ,
    SMOKE_SEQ_LEN,
    layers_for_sae_at,
    result_stem,
    sae_id_for_layer,
)
from recon.data import iter_token_batches  # noqa: E402
from recon.metrics import ReconAccumulators  # noqa: E402
from recon.model import (  # noqa: E402
    check_cuda,
    forward_text,
    load_model,
    peak_vram_gb,
    probe_sdpa_backend,
    register_resid_hooks,
)
from recon.sae import encode_decode, load_sae  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure Qwen-Scope reconstruction error")
    p.add_argument("--smoke", action="store_true", help="8 sequences x 128 tokens")
    p.add_argument("--check-env", action="store_true", help="print CUDA/SDPA info and exit")
    p.add_argument("--n-seq", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument(
        "--sae-at",
        choices=("50", "85", "both"),
        default="both",
        help="SAE depth: 50%% (layer 32), 85%% (layer 54), or both",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = check_cuda()
    print(json.dumps(env, indent=2))
    if args.check_env:
        return

    smoke = args.smoke
    n_seq = args.n_seq or (SMOKE_N_SEQ if smoke else N_SEQ)
    seq_len = args.seq_len or (SMOKE_SEQ_LEN if smoke else SEQ_LEN)
    layer_indices = layers_for_sae_at(args.sae_at)
    depth_by_layer = {layer: depth for depth, layer in LAYER_DEPTHS.items()}

    torch.cuda.reset_peak_memory_stats()
    loaded = load_model(seq_len=seq_len)
    print(
        f"loaded {MODEL_ID} via {loaded.source}; "
        f"n_layers={len(loaded.layers)} device={loaded.device}"
    )
    if len(loaded.layers) != N_LAYERS:
        print(f"warning: expected {N_LAYERS} layers, found {len(loaded.layers)}")

    saes = {}
    sae_sources = {}
    for layer in layer_indices:
        packed = load_sae(layer, loaded.device, loaded.dtype, D_MODEL)
        saes[layer] = packed.sae
        sae_sources[layer] = packed.source
        print(f"SAE layer {layer}: {sae_id_for_layer(layer)} via {packed.source}")

    captured, handles = register_resid_hooks(loaded.layers, layer_indices)
    acc = {layer: ReconAccumulators() for layer in layer_indices}
    attn_backend = None
    n_done = 0

    try:
        for batch in iter_token_batches(loaded.tokenizer, n_seq=n_seq, seq_len=seq_len, device=loaded.device):
            captured.clear()
            if attn_backend is None:
                attn_backend = probe_sdpa_backend(
                    loaded.model,
                    batch["input_ids"],
                    batch["attention_mask"],
                )
                print(f"using SDPA backend: {attn_backend}")
                for layer, hidden in captured.items():
                    if hidden.shape[-1] != D_MODEL:
                        raise RuntimeError(
                            f"layer {layer} d_model={hidden.shape[-1]}, expected {D_MODEL}"
                        )
            else:
                forward_text(
                    loaded.model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    attn_backend=attn_backend,
                )

            token_mask = batch["token_mask"]
            for layer, hidden in captured.items():
                x_hat, acts = encode_decode(saes[layer], hidden)
                acc[layer].update(hidden, x_hat, acts, token_mask)
            n_done += 1
            if n_done % 50 == 0:
                tqdm.write(f"processed {n_done}/{n_seq} peak_vram={peak_vram_gb():.2f} GiB")
    finally:
        for h in handles:
            h.remove()

    layers_out = []
    for layer in layer_indices:
        stats = acc[layer].finalize()
        row = {
            "layer": layer,
            "depth": depth_by_layer[layer],
            "sae_id": sae_id_for_layer(layer),
            "sae_source": sae_sources[layer],
            **stats,
        }
        layers_out.append(row)
        print(
            f"layer {layer} ({depth_by_layer[layer]:.0%}): "
            f"FVU={stats['fvu']:.4f} MSE={stats['mse']:.4f} "
            f"cos={stats['cosine']:.4f} L0={stats['l0']:.2f} "
            f"n_tokens={stats['n_tokens']}"
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": smoke,
        "sae_at": args.sae_at,
        "model_id": MODEL_ID,
        "model_source": loaded.source,
        "sae_repo": SAE_REPO_ID,
        "sae_width": SAE_WIDTH,
        "sae_topk": SAE_TOPK,
        "n_seq": n_seq,
        "seq_len": seq_len,
        "attn_implementation": "sdpa",
        "attn_backend": attn_backend,
        "env": env,
        "peak_vram_gb": peak_vram_gb(),
        "note": (
            "TopK residual SAEs from Qwen-Scope on BF16 Qwen3.5-27B. "
            "L0 should be approximately the SAE Top-K (50)."
        ),
        "layers": layers_out,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = result_stem(args.sae_at, smoke)
    json_path = RESULTS_DIR / f"{stem}.json"
    csv_path = RESULTS_DIR / f"{stem}.csv"
    json_path.write_text(json.dumps(payload, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "depth",
                "sae_id",
                "fvu",
                "mse",
                "cosine",
                "relative_l2",
                "l0",
                "n_tokens",
                "attn_backend",
                "peak_vram_gb",
            ],
        )
        writer.writeheader()
        for row in layers_out:
            writer.writerow(
                {
                    "layer": row["layer"],
                    "depth": row["depth"],
                    "sae_id": row["sae_id"],
                    "fvu": row["fvu"],
                    "mse": row["mse"],
                    "cosine": row["cosine"],
                    "relative_l2": row["relative_l2"],
                    "l0": row["l0"],
                    "n_tokens": row["n_tokens"],
                    "attn_backend": attn_backend,
                    "peak_vram_gb": payload["peak_vram_gb"],
                }
            )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
