# umineko_llm_sae_bias

Reconstruction-error experiment (code, notes, results): [`reconstruction_error/`](reconstruction_error/).

Culprit inference (Qwen3.5-27B thinking on EP1 scripts): `.venv/bin/python culprit_infer/run_infer.py`

Run the 27B jobs on **DGX Spark** (GB10, ARM64). This x86 checkout is for editing; do not copy the local `.venv` to Spark.
