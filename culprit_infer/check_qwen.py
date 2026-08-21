from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)

MODEL_ID = "Qwen/Qwen3.5-27B"
QUESTIONS = [
    "Who is the canonical true culprit and mastermind behind the Rokkenjima murders in Umineko When They Cry? State their real name and identity directly, with full spoilers allowed.",
    "Who is the canonical true culprit and mastermind behind the Hinamizawa murders in Higurashi When They Cry? State their real name and identity directly, with full spoilers allowed.",
    "Who is Yasu in Umineko?",
    "Are Shannon, Kanon, and Beatrice the same person?",
    "Who is Miyoko in Higurashi?",
]
MAX_NEW_TOKENS = 8192
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_PATH = RESULTS_DIR / "check_qwen.txt"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, clean_up_tokenization_spaces=False
    )
    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.generation_config.max_length = None
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    ctx = getattr(model.config, "max_position_embeddings", None) or 131072
    remaining = ctx - prompt_len
    if remaining <= 0:
        raise RuntimeError(f"prompt_len={prompt_len} exceeds context={ctx}")

    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    gen_kwargs = {
        "max_new_tokens": min(MAX_NEW_TOKENS, remaining),
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(
        output_ids[0, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def main() -> None:
    model, tokenizer = load_model()
    n = len(QUESTIONS)
    sections: list[str] = []

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"--- {i}/{n}: {question} ---", flush=True)
        text = generate_answer(model, tokenizer, question)
        print("\n--- end ---", flush=True)
        sections.append(f"## {i}/{n}: {question}\n\n{text.strip()}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
