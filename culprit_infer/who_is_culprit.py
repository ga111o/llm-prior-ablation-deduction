from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    TextStreamer,
)

# Official meta-llama/Llama-3.3-70B-Instruct is gated. This is the same
# Instruct 70B weights already quantized to 4-bit NF4 (bitsandbytes).
MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
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


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def main() -> None:
    model, tokenizer = load_model()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
    )
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
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }

    print("--- generation ---", flush=True)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    print("\n--- end ---", flush=True)

    text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
