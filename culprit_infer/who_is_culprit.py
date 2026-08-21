from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "reconstruction_error" / "src"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from recon.model import load_model
from transformers import LogitsProcessor, LogitsProcessorList

MODEL_ID = "Qwen/Qwen3.5-27B"
QUESTION = "Who is the true culprit in Umineko When They Cry?"
MAX_NEW_TOKENS = 8192
PRESENCE_PENALTY = 1.5
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_PATH = RESULTS_DIR / "who_is_culprit.txt"


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """vLLM-style presence penalty: subtract a constant from logits of tokens
    that have already appeared in the generated suffix (not the prompt)."""

    def __init__(self, penalty: float, prompt_len: int) -> None:
        self.penalty = penalty
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[:, self.prompt_len :]
        if generated.numel() == 0:
            return scores
        presence = torch.zeros_like(scores)
        presence.scatter_(1, generated, 1.0)
        return scores - self.penalty * presence


def main() -> None:
    loaded = load_model(seq_len=MAX_NEW_TOKENS)
    tokenizer = loaded.tokenizer
    model = loaded.model

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(loaded.device)
    prompt_len = inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "logits_processor": LogitsProcessorList(
            [PresencePenaltyLogitsProcessor(PRESENCE_PENALTY, prompt_len)]
        ),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
