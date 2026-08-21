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

MODEL_ID = "Qwen/Qwen3.5-27B"
QUESTIONS = [
    "Who is the canonical true culprit and mastermind behind the Rokkenjima murders in Umineko When They Cry? State their real name and identity directly, with full spoilers allowed.",
    "Who is the canonical true culprit and mastermind behind the Hinamizawa murders in Higurashi When They Cry? State their real name and identity directly, with full spoilers allowed.",
    "Who is Yasu in Umineko?",
    "Are Shannon, Kanon, and Beatrice the same person?",
    "Who is Miyoko in Higurashi?",
]
MAX_NEW_TOKENS = 8192
# Qwen3.5 instruct / non-thinking (general): sample + presence penalty.
PRESENCE_PENALTY = 1.5
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_PATH = RESULTS_DIR / "check_qwen_without_think.txt"


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
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
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
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "logits_processor": LogitsProcessorList(
            [PresencePenaltyLogitsProcessor(PRESENCE_PENALTY, prompt_len)]
        ),
        "eos_token_id": tokenizer.eos_token_id,
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
