from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from recon.config import ATTN_IMPLEMENTATION, MODEL_ID


def check_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    info = {
        "device_name": name,
        "capability": list(cap),
        "is_blackwell": cap[0] == 12,
        "is_gb10_sm121": cap == (12, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
    }
    if cap not in {(12, 0), (12, 1)}:
        print(f"warning: expected Blackwell sm_120/sm_121, got {name} sm_{cap[0]}{cap[1]}")
    return info


def _backend_enum(name: str):
    from torch.nn.attention import SDPBackend

    mapping = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    if name not in mapping:
        raise ValueError(f"unknown SDPA backend {name}")
    return mapping[name]


def candidate_sdpa_backends() -> list[str]:
    return ["auto"]


@contextmanager
def sdpa_context(backend_name: str | None) -> Iterator[str]:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    auto = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    if backend_name in (None, "auto", "flash"):
        with sdpa_kernel(auto):
            yield "auto"
        return
    with sdpa_kernel(_backend_enum(backend_name)):
        yield backend_name


def _find_language_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    paths = [
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
    ]
    for path in paths:
        obj: Any = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.ModuleList) and len(obj) >= 2:
                return obj
        except AttributeError:
            continue
    raise RuntimeError("could not find language decoder layers")


def _use_gdn_kernels() -> bool:
    if os.environ.get("RECON_USE_KERNELS", "") == "1":
        return True
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (12, 1)


def _load_transformers(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda",
        "attn_implementation": ATTN_IMPLEMENTATION,
    }
    if _use_gdn_kernels():
        kwargs["use_kernels"] = True
        kwargs["trust_remote_code"] = True
        print("loading Gated DeltaNet Hub kernels (GB10 / RECON_USE_KERNELS=1)")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, "transformers"


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: Any
    layers: torch.nn.ModuleList
    source: str
    device: torch.device
    dtype: torch.dtype


def load_model(seq_len: int) -> LoadedModel:
    del seq_len
    try:
        loaded = _load_transformers(MODEL_ID)
    except Exception as exc:
        raise RuntimeError(f"failed to load {MODEL_ID}") from exc

    model, tokenizer, source = loaded
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    layers = _find_language_layers(model)
    device = next(model.parameters()).device
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        source=source,
        device=device,
        dtype=torch.bfloat16,
    )


def register_resid_hooks(
    layers: torch.nn.ModuleList,
    layer_indices: list[int],
) -> tuple[dict[int, torch.Tensor], list[Any]]:
    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden
            return output

        return _hook

    handles = []
    for idx in layer_indices:
        handles.append(layers[idx].register_forward_hook(make_hook(idx)))
    return captured, handles


def _model_forward(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
    kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
    if attention_mask is not None and not bool(attention_mask.all()):
        kwargs["attention_mask"] = attention_mask
    try:
        model(**kwargs)
    except TypeError:
        model(input_ids=input_ids, attention_mask=attention_mask)


@torch.inference_mode()
def probe_sdpa_backend(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> str:
    last_error: Exception | None = None
    for name in candidate_sdpa_backends():
        try:
            with sdpa_context(name):
                _model_forward(model, input_ids, attention_mask)
            return name
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"SDPA backend {name} failed: {type(exc).__name__}: {exc}")
    raise RuntimeError("no working SDPA backend") from last_error


@torch.inference_mode()
def forward_text(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    attn_backend: str | None = None,
) -> str:
    with sdpa_context(attn_backend):
        _model_forward(model, input_ids, attention_mask)
    return attn_backend or "default"


def peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)
