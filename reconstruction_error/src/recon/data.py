from __future__ import annotations

from collections.abc import Iterator

import torch
from datasets import load_dataset
from tqdm import tqdm

from recon.config import DATASET_NAME, DATASET_TEXT_FIELD


def apply_user_chat_template(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": False,
    }
    try:
        return inner.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return inner.apply_chat_template(messages, **kwargs)
        except Exception:
            pass
    except Exception:
        pass
    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        return inner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except Exception:
        return text


def special_token_mask(input_ids: torch.Tensor, tokenizer, attention_mask: torch.Tensor) -> torch.Tensor:
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    mask = attention_mask.bool()
    special_ids = getattr(inner, "all_special_ids", None) or []
    if special_ids:
        special = torch.tensor(list(special_ids), device=input_ids.device)
        mask = mask & ~torch.isin(input_ids, special)
    return mask


def iter_token_batches(
    tokenizer,
    n_seq: int,
    seq_len: int,
    dataset_name: str = DATASET_NAME,
    device: torch.device | str = "cuda",
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield exact-length sequences with no padding so SDPA Flash can run."""
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    if getattr(inner, "pad_token", None) is None:
        inner.pad_token = inner.eos_token

    ds = load_dataset(dataset_name, split="train", streaming=True)
    produced = 0
    leftover: list[int] = []
    pbar = tqdm(total=n_seq, desc="sequences", leave=False)
    for example in ds:
        if produced >= n_seq:
            break
        text = example.get(DATASET_TEXT_FIELD) or example.get("content") or ""
        if not isinstance(text, str) or len(text.strip()) < 80:
            continue
        text = text[: seq_len * 16]
        prompt = apply_user_chat_template(tokenizer, text)
        ids = inner(prompt, add_special_tokens=False)["input_ids"]
        leftover.extend(ids)
        while len(leftover) >= seq_len and produced < n_seq:
            chunk = leftover[:seq_len]
            leftover = leftover[seq_len:]
            input_ids = torch.tensor([chunk], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            token_mask = special_token_mask(input_ids, tokenizer, attention_mask)
            if int(token_mask.sum().item()) < max(16, seq_len // 8):
                continue
            produced += 1
            pbar.update(1)
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_mask": token_mask,
            }
    pbar.close()
    if produced < n_seq:
        raise RuntimeError(f"only produced {produced}/{n_seq} sequences from {dataset_name}")
