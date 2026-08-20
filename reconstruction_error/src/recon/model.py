from __future__ import annotations

import os
import re
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


class _ShardLoadFallback(Exception):
    """Checkpoint keys or format are not a direct CUDA shard load; use from_pretrained."""


def _remote_kwargs() -> dict[str, Any]:
    if _use_gdn_kernels():
        return {"trust_remote_code": True}
    return {}


def _apply_gdn_kernels(model: torch.nn.Module) -> None:
    if not _use_gdn_kernels():
        return
    print("loading Gated DeltaNet Hub kernels (GB10 / RECON_USE_KERNELS=1)")
    model.set_use_kernels(True)


def _resolve_safetensor_files(model_id: str) -> tuple[list[str], set[str]]:
    from transformers.modeling_utils import _get_resolved_checkpoint_files

    checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
        model_id,
        variant=None,
        gguf_file=None,
        use_safetensors=True,
        user_agent=None,
        is_remote_code=False,
    )
    if not checkpoint_files:
        raise _ShardLoadFallback("no checkpoint files")
    if any(not path.endswith(".safetensors") for path in checkpoint_files):
        raise _ShardLoadFallback("checkpoint is not safetensors")

    if sharded_metadata and sharded_metadata.get("weight_map"):
        ckpt_keys = set(sharded_metadata["weight_map"])
    else:
        from safetensors import safe_open

        ckpt_keys = set()
        for path in checkpoint_files:
            with safe_open(path, framework="pt") as handle:
                ckpt_keys.update(handle.keys())
    return checkpoint_files, ckpt_keys


def _model_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    tensors.update(model.named_parameters())
    tensors.update(model.named_buffers())
    return tensors


def _storage_id(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return id(tensor)
    return tensor.data_ptr()


def _map_ckpt_key(key: str) -> str:
    # qwen3_5_text PrefixChange(prefix_to_remove="language_model", model_prefix="model")
    infix = "model.language_model."
    if key.startswith(infix):
        return "model." + key[len(infix) :]
    return key


def _skip_unexpected_ckpt_key(model: torch.nn.Module, key: str) -> bool:
    patterns = getattr(type(model), "_keys_to_ignore_on_load_unexpected", None) or []
    return any(re.search(pattern, key) for pattern in patterns)


def _checkpoint_key_map(model: torch.nn.Module, ckpt_keys: set[str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for key in ckpt_keys:
        if _skip_unexpected_ckpt_key(model, key):
            continue
        mapped[key] = _map_ckpt_key(key)
    return mapped


def _checkpoint_keys_compatible(model: torch.nn.Module, mapped_keys: set[str]) -> None:
    model_keys = set(model.state_dict().keys())
    unexpected = mapped_keys - model_keys
    if unexpected:
        sample = ", ".join(sorted(unexpected)[:4])
        raise _ShardLoadFallback(f"unexpected checkpoint keys (e.g. {sample})")

    tensors = _model_tensors(model)
    aliases: dict[int, list[str]] = {}
    for name, tensor in tensors.items():
        aliases.setdefault(_storage_id(tensor), []).append(name)

    missing = model_keys - mapped_keys
    ignore_missing = getattr(type(model), "_keys_to_ignore_on_load_missing", None) or []
    for name in missing:
        if any(re.search(pattern, name) for pattern in ignore_missing):
            continue
        tensor = tensors.get(name)
        if tensor is None:
            raise _ShardLoadFallback(f"missing checkpoint key {name}")
        names = aliases[_storage_id(tensor)]
        if not any(alias in mapped_keys for alias in names):
            raise _ShardLoadFallback(f"missing checkpoint key {name}")


def _copy_shard_tensors(
    checkpoint_files: list[str],
    tensors: dict[str, torch.Tensor],
    ckpt_to_model: dict[str, str],
) -> None:
    from safetensors import safe_open
    from tqdm import tqdm

    filled: set[int] = set()
    print(f"loading {len(checkpoint_files)} safetensor shards onto cuda")
    for path in tqdm(checkpoint_files, desc="Loading shards"):
        with safe_open(path, framework="pt", device="cuda") as handle:
            for key in handle.keys():
                dst_name = ckpt_to_model.get(key)
                if dst_name is None:
                    continue
                dst = tensors[dst_name]
                storage = _storage_id(dst)
                if storage in filled:
                    continue
                src = handle.get_tensor(key)
                if src.shape != dst.shape:
                    raise _ShardLoadFallback(
                        f"shape mismatch for {dst_name}: {tuple(src.shape)} vs {tuple(dst.shape)}"
                    )
                if src.dtype != dst.dtype:
                    src = src.to(dtype=dst.dtype)
                dst.data.copy_(src)
                filled.add(storage)
                del src


def _empty_cuda_model(model_id: str):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.initialization import no_init_weights

    remote = _remote_kwargs()
    config = AutoConfig.from_pretrained(model_id, **remote)
    with no_init_weights(), torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(
            config,
            dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
            **remote,
        )
    model.tie_weights()
    return model


def _load_cuda_shards(model_id: str) -> torch.nn.Module:
    checkpoint_files, ckpt_keys = _resolve_safetensor_files(model_id)
    model = _empty_cuda_model(model_id)
    try:
        ckpt_to_model = _checkpoint_key_map(model, ckpt_keys)
        _checkpoint_keys_compatible(model, set(ckpt_to_model.values()))
        _copy_shard_tensors(checkpoint_files, _model_tensors(model), ckpt_to_model)
    except _ShardLoadFallback:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    return model


def _load_transformers_pretrained(model_id: str) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
        "attn_implementation": ATTN_IMPLEMENTATION,
        **_remote_kwargs(),
    }
    if _use_gdn_kernels():
        kwargs["use_kernels"] = True
        print("loading Gated DeltaNet Hub kernels (GB10 / RECON_USE_KERNELS=1)")
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def _load_transformers(model_id: str):
    from transformers import AutoTokenizer

    try:
        model = _load_cuda_shards(model_id)
        model.eval()
        _apply_gdn_kernels(model)
    except _ShardLoadFallback as exc:
        print(f"gpu shard load fallback to from_pretrained: {exc}")
        model = _load_transformers_pretrained(model_id)
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
