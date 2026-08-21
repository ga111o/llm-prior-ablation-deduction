# Qwen3.5-27B + Qwen-Scope reconstruction error

Measures **FVU / MSE / cosine / L0** of residual-stream TopK SAEs at **50% (layer 32)** and **85% (layer 54)** on BF16 `Qwen/Qwen3.5-27B`.

SAE checkpoint: `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` (`d_sae=81920`, Top-K=50). `L0_50` is the sparsity, not a layer index.

Notes: [Reconstruction_error.md](Reconstruction_error.md)

## Model

- `Qwen/Qwen3.5-27B` in **BF16** via `AutoModelForCausalLM` (text backbone only; no vision tower).
- 64 layers, `d_model=5120`, hybrid 3× Gated DeltaNet + 1× Gated Attention.
- Native context 262,144. No bitsandbytes / Unsloth / 4-bit path.

## Setup (DGX Spark, GB10 sm_121, ARM64)

Create a **new** venv on Spark. Do not copy an x86 `.venv`.

PyTorch must be an aarch64 CUDA wheel that covers Blackwell (`sm_120` binaries are compatible with `sm_121`):

```bash
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
# If transformers 5.2 lacks Qwen3.5 GB10 kernels:
# uv pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"
```

Optional: NGC `pytorch:25.10-py3` (or newer) instead of the cu128 wheel.

Do **not** install `flash-attn`, `causal_conv1d`, `fla`, Unsloth, or bitsandbytes. On GB10 the loader tries `set_use_kernels(True)` so Gated DeltaNet can use the Hub kernel (`Atlas-Inference/gdn`). If kernelize fails (transformers `Qwen3_5GatedDeltaNet.forward` vs the pinned Hub revision), it prints a warning and continues with the slower PyTorch GDN path. Force on with `RECON_USE_KERNELS=1` on other GPUs; force off with `RECON_USE_KERNELS=0`.

`Python.h` / Triton CPU warnings from `kernels-community/fla` JIT are unrelated to that GDN signature mismatch. Optional: install distro `python3-devel` if you want that compile to succeed.

Gated Hub models need a token in `~/.cache/huggingface/token` or `HF_TOKEN`.

x86 can run tokenizer-only checks (`culprit_infer/run_infer.py --count-only`). Full 27B jobs belong on Spark.

## Run

```bash
python reconstruction_error/scripts/run_recon.py --check-env

python reconstruction_error/scripts/run_recon.py --smoke --sae-at both

python reconstruction_error/scripts/run_recon.py --sae-at 50
python reconstruction_error/scripts/run_recon.py --sae-at 85
python reconstruction_error/scripts/run_recon.py --sae-at both
```

Outputs under `reconstruction_error/results/`:

| `--sae-at` | stem |
|---|---|
| `50` | `recon_error_l50` |
| `85` | `recon_error_l85` |
| `both` | `recon_error_both` |

Smoke adds `_smoke`. JSON and CSV are written for each stem.

Culprit infer (all EP1 chapters, thinking on; sampling T=0.6, seed 42 — not greedy):

```bash
python culprit_infer/run_infer.py --count-only
python culprit_infer/run_infer.py
```

After the thinking-loop fix, re-run on Spark with a short prompt-only pass first:

```bash
python culprit_infer/run_infer.py --mode prompt --max-new-tokens 2048
```

Confirm thinking closes with `</think>` and does not cycle the same n-gram. Same machine + same `--seed` is the reproducibility story; SDPA/Flash is not bitwise across kernels.

## Attention

Uses PyTorch SDPA (`attn_implementation="sdpa"`), preferring Flash. Does not install Dao `flash-attn`. Fallback order: flash → efficient → math.

## Config

| Variable | Default |
|---|---|
| `RECON_MODEL_ID` | `Qwen/Qwen3.5-27B` |
| `RECON_SAE_REPO` | `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` |
| `RECON_SAE_WIDTH` | `81920` |
| `RECON_SAE_TOPK` | `50` |
| `RECON_SEQ_LEN` | `1024` |
| `RECON_N_SEQ` | `2048` |
| `RECON_USE_KERNELS` | unset (try on if GB10; `0`/`false`/`off` force off, `1`/`true`/`on` force on) |

OOM: pass `--seq-len 512` (token count is written into the JSON).
