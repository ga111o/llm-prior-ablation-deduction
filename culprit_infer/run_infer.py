from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "reconstruction_error" / "src"))

from recon.config import MAX_POSITION, MODEL_ID  # noqa: E402

INSTRUCTION = (
    "Using this as a reference, review the entire script and infer the final culprit. need to find the 'real culprit,' not just the surface-level part. Even if some mysterious event occurs, explain it entirely as 'humanity and trickery.'"
)
PROMPT_ONLY = "Who is the true culprit in Umineko When They Cry?"
SCRIPTS_DIR = REPO_ROOT / "umineko-scripts"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHAPTER_FILES = ["00_opening.txt", *[f"{i:02d}.txt" for i in range(1, 17)]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Infer the EP1 culprit with Qwen3.5-27B thinking"
    )
    p.add_argument("--count-only", action="store_true", help="print token count and exit")
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument(
        "--mode",
        choices=("oneshot", "prompt"),
        default="oneshot",
        help="oneshot: full EP1 script; prompt: question only",
    )
    return p.parse_args()


def load_chapters() -> list[tuple[str, str]]:
    chapters: list[tuple[str, str]] = []
    for name in CHAPTER_FILES:
        path = SCRIPTS_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"missing chapter file: {path}")
        chapters.append((name, path.read_text(encoding="utf-8")))
    return chapters


def concatenated_script(chapters: list[tuple[str, str]]) -> str:
    return "\n\n".join(text for _name, text in chapters)


def oneshot_user_text(script: str) -> str:
    return f"{script}\n\n{INSTRUCTION}"


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def inner_tokenizer(processor) -> Any:
    return getattr(processor, "tokenizer", processor)


def user_messages(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


def _chat_kwargs() -> dict[str, Any]:
    return {
        "add_generation_prompt": True,
        "enable_thinking": True,
    }


THINKING_REQUIRED = "tokenizer does not support enable_thinking; thinking is required"


def _is_thinking_unsupported(exc: BaseException) -> bool:
    return "enable_thinking" in str(exc)


def _apply_chat_template(processor, text: str, **extra: Any) -> Any:
    inner = inner_tokenizer(processor)
    kwargs = {**extra, **_chat_kwargs()}
    message_variants = [
        user_messages(text),
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
    ]
    targets = (processor,) if processor is inner else (processor, inner)
    last_exc: BaseException | None = None
    thinking_unsupported = False
    for messages in message_variants:
        for target in targets:
            try:
                return target.apply_chat_template(messages, **kwargs)
            except TypeError as exc:
                last_exc = exc
                if _is_thinking_unsupported(exc):
                    thinking_unsupported = True
            except Exception as exc:
                last_exc = exc
    if thinking_unsupported:
        raise RuntimeError(THINKING_REQUIRED) from last_exc
    raise RuntimeError("failed to apply chat template with thinking enabled") from last_exc


def apply_chat_text(processor, text: str) -> str:
    return _apply_chat_template(processor, text, tokenize=False)


def _token_len(ids: Any) -> int:
    if hasattr(ids, "shape"):
        return int(ids.shape[-1])
    if ids and isinstance(ids[0], (list, tuple)):
        return len(ids[0])
    return len(ids)


def count_prompt_tokens(processor, text: str) -> int:
    inner = inner_tokenizer(processor)
    if getattr(inner, "pad_token", None) is None:
        inner.pad_token = inner.eos_token
    try:
        encoded = _apply_chat_template(processor, text, tokenize=True, return_dict=True)
        return _token_len(encoded["input_ids"])
    except RuntimeError as exc:
        if str(exc) == THINKING_REQUIRED:
            raise
    prompt = apply_chat_text(processor, text)
    ids = inner(prompt, add_special_tokens=False)["input_ids"]
    return _token_len(ids)


def encode_prompt(processor, text: str, device: Any) -> dict[str, Any]:
    import torch

    inner = inner_tokenizer(processor)
    if getattr(inner, "pad_token", None) is None:
        inner.pad_token = inner.eos_token
    try:
        encoded = _apply_chat_template(
            processor,
            text,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except RuntimeError as exc:
        if str(exc) == THINKING_REQUIRED:
            raise
        prompt = apply_chat_text(processor, text)
        encoded = inner(prompt, return_tensors="pt", add_special_tokens=False)
    tensors: dict[str, Any] = {}
    for key, value in encoded.items():
        if torch.is_tensor(value):
            tensors[key] = value.to(device)
        else:
            tensors[key] = value
    return tensors


def prepare_generate_inputs(encoded: dict[str, Any]) -> dict[str, Any]:
    import torch

    keep = {}
    for key in ("input_ids", "token_type_ids"):
        if key in encoded:
            keep[key] = encoded[key]
    mask = encoded.get("attention_mask")
    if mask is not None and torch.is_tensor(mask) and not bool(mask.all()):
        keep["attention_mask"] = mask
    return keep


def generate_completion(
    model: Any,
    processor,
    user_text: str,
    device: Any,
    max_new_tokens: int,
) -> tuple[str, int, int]:
    import gc

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    encoded = prepare_generate_inputs(encode_prompt(processor, user_text, device=device))
    prompt_len = _token_len(encoded["input_ids"])
    print(f"  generating prompt_tokens={prompt_len}", flush=True)
    inner = inner_tokenizer(processor)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    pad_id = getattr(inner, "pad_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    sdpa_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    output_ids = None
    try:
        with torch.inference_mode():
            with sdpa_kernel(sdpa_backends):
                output_ids = model.generate(**encoded, **gen_kwargs)
        new_ids = output_ids[0, prompt_len:].detach().cpu()
        text = inner.decode(new_ids, skip_special_tokens=False).strip()
        n_new = int(new_ids.shape[-1])
    finally:
        del encoded, output_ids
        gc.collect()
        torch.cuda.empty_cache()
    return text, prompt_len, n_new


def write_results(
    payload: dict[str, Any],
    completion: str,
    stem: str = "generation",
) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = RESULTS_DIR / f"{stem}.txt"
    json_path = RESULTS_DIR / f"{stem}.json"
    txt_path.write_text(completion + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return txt_path, json_path


def main() -> None:
    args = parse_args()
    if args.mode == "prompt":
        user_text = PROMPT_ONLY
        instruction = PROMPT_ONLY
        result_stem = "generation_prompt"
        count_meta: dict[str, Any] = {}
    else:
        chapters = load_chapters()
        user_text = oneshot_user_text(concatenated_script(chapters))
        instruction = INSTRUCTION
        result_stem = "generation"
        count_meta = {"n_chapters": len(chapters)}
    processor = load_tokenizer(MODEL_ID)
    prompt_tokens = count_prompt_tokens(processor, user_text)
    total_tokens = prompt_tokens + args.max_new_tokens
    if total_tokens > MAX_POSITION:
        raise RuntimeError(
            f"prompt_tokens={prompt_tokens} + max_new_tokens={args.max_new_tokens} "
            f"exceeds MAX_POSITION={MAX_POSITION}"
        )

    print(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "mode": args.mode,
                **count_meta,
                "n_prompt_tokens": prompt_tokens,
                "max_new_tokens": args.max_new_tokens,
                "max_position": MAX_POSITION,
                "enable_thinking": True,
            },
            indent=2,
        )
    )
    if args.count_only:
        return

    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    from recon.model import check_cuda, load_model, peak_vram_gb

    env = check_cuda()
    print(json.dumps(env, indent=2), flush=True)
    torch.cuda.reset_peak_memory_stats()
    loaded = load_model(seq_len=total_tokens)
    print(
        f"loaded {MODEL_ID} via {loaded.source}; device={loaded.device}",
        flush=True,
    )
    completion, used_prompt_tokens, n_new = generate_completion(
        loaded.model,
        loaded.tokenizer,
        user_text,
        device=loaded.device,
        max_new_tokens=args.max_new_tokens,
    )

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_source": loaded.source,
        "n_prompt_tokens": used_prompt_tokens,
        "n_new_tokens": n_new,
        "peak_vram_gb": peak_vram_gb(),
        "mode": args.mode,
        "enable_thinking": True,
        "max_new_tokens": args.max_new_tokens,
        "instruction": instruction,
        "env": env,
        "completion": completion,
    }
    if args.mode == "oneshot":
        payload["chapters"] = CHAPTER_FILES

    txt_path, json_path = write_results(payload, completion, stem=result_stem)
    print(completion)
    print(f"wrote {txt_path}")
    print(f"wrote {json_path}")
    print(f"peak_vram_gb={payload['peak_vram_gb']:.3f} n_new_tokens={n_new}")


if __name__ == "__main__":
    main()
