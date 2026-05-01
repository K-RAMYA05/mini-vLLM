"""Minimal multi-LoRA support for serving-time adapter selection.

The implementation is intentionally pragmatic:
  - adapters are loaded once at startup;
  - each scheduler step serves one adapter cohort at a time;
  - all wrapped modules share a single active adapter name.

This gives the engine real multi-adapter serving without requiring mixed-
adapter fused kernels inside one forward pass.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn


_LORA_KEY_RE = re.compile(r"^(.*)\.lora_(A|B)(?:\.[^.]+)?\.weight$")
_PREFIXES_TO_STRIP = (
    "base_model.model.",
    "base_model.",
)


@dataclass
class LoRAAdapterWeights:
    rank: int
    alpha: float
    tensors: Dict[str, Dict[str, torch.Tensor]]


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.adapters = nn.ModuleDict()
        self.adapter_scalings: dict[str, float] = {}
        self.active_adapter: str | None = None

    def add_adapter(
        self,
        name: str,
        a_weight: torch.Tensor,
        b_weight: torch.Tensor,
        alpha: float,
    ) -> None:
        rank = int(a_weight.shape[0])
        if a_weight.shape != (rank, self.in_features):
            raise ValueError(
                f"adapter {name!r} has mismatched A shape {tuple(a_weight.shape)} "
                f"for layer with in_features={self.in_features}"
            )
        if b_weight.shape != (self.out_features, rank):
            raise ValueError(
                f"adapter {name!r} has mismatched B shape {tuple(b_weight.shape)} "
                f"for layer with out_features={self.out_features}, rank={rank}"
            )
        params = nn.ParameterDict({
            "lora_A": nn.Parameter(a_weight.detach().to(device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype), requires_grad=False),
            "lora_B": nn.Parameter(b_weight.detach().to(device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype), requires_grad=False),
        })
        self.adapters[name] = params
        self.adapter_scalings[name] = float(alpha) / max(rank, 1)

    def set_active_adapter(self, name: str | None) -> None:
        self.active_adapter = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        if self.active_adapter is None:
            return out
        if self.active_adapter not in self.adapters:
            return out
        params = self.adapters[self.active_adapter]
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        delta = (x_2d @ params["lora_A"].transpose(0, 1)) @ params["lora_B"].transpose(0, 1)
        delta = delta.reshape(*orig_shape[:-1], self.out_features)
        return out + delta * self.adapter_scalings[self.active_adapter]


class LoRAManager:
    def __init__(self, model: nn.Module):
        self.model = model
        self.available_adapters: dict[str, Path] = {}
        self.active_adapter: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.available_adapters)

    def set_active_adapter(self, name: str | None) -> None:
        if name is not None and name not in self.available_adapters:
            raise KeyError(f"unknown LoRA adapter {name!r}")
        self.active_adapter = name
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.set_active_adapter(name)

    def load_adapter(self, name: str, path: str | Path) -> None:
        adapter_path = Path(path)
        weights = _load_adapter_weights(adapter_path)
        named_modules = dict(self.model.named_modules())
        touched = 0
        for module_name, pair in weights.tensors.items():
            target = _normalize_module_name(module_name)
            module = named_modules.get(target)
            if module is None:
                continue
            if not isinstance(module, (nn.Linear, LoRALinear)):
                continue
            if isinstance(module, LoRALinear):
                wrapped = module
            else:
                wrapped = LoRALinear(module)
                _set_submodule(self.model, target, wrapped)
                named_modules[target] = wrapped
            if "A" not in pair or "B" not in pair:
                continue
            wrapped.add_adapter(name, pair["A"], pair["B"], alpha=weights.alpha)
            touched += 1
        if touched == 0:
            raise ValueError(f"adapter {name!r} at {str(adapter_path)!r} did not match any linear layers")
        self.available_adapters[name] = adapter_path


def apply_lora_adapters(model: nn.Module, adapters: Iterable[str]) -> LoRAManager:
    manager = LoRAManager(model)
    for item in adapters:
        name, _, path = item.partition("=")
        if not name or not path:
            raise ValueError(f"invalid LoRA adapter spec {item!r}; expected name=path")
        manager.load_adapter(name, path)
    manager.set_active_adapter(None)
    return manager


def _load_adapter_weights(path: Path) -> LoRAAdapterWeights:
    config = {}
    if path.is_dir():
        config_path = path / "adapter_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
        weights_path = _find_weights_file(path)
    else:
        weights_path = path
        config_candidate = path.with_name("adapter_config.json")
        if config_candidate.exists():
            config = json.loads(config_candidate.read_text())

    state_dict = _load_state_dict(weights_path)
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    inferred_rank = None
    for raw_key, tensor in state_dict.items():
        match = _LORA_KEY_RE.match(raw_key)
        if match is None:
            continue
        module_name, which = match.groups()
        pairs.setdefault(module_name, {})["A" if which == "A" else "B"] = tensor.detach().cpu()
        if which == "A":
            inferred_rank = int(tensor.shape[0])
    if not pairs:
        raise ValueError(f"no LoRA tensors found in {str(weights_path)!r}")
    rank = int(config.get("r", inferred_rank or 0))
    alpha = float(config.get("lora_alpha", rank or 1))
    return LoRAAdapterWeights(rank=rank, alpha=alpha, tensors=pairs)


def _find_weights_file(path: Path) -> Path:
    for candidate in (
        path / "adapter_model.safetensors",
        path / "adapter_model.bin",
        path / "adapter_model.pt",
        path / "pytorch_model.bin",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find adapter weights under {str(path)!r}")


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "loading .safetensors LoRA adapters requires `safetensors` to be installed"
            ) from exc
        return load_file(path)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"unsupported adapter checkpoint format in {str(path)!r}")
    return state


def _normalize_module_name(name: str) -> str:
    for prefix in _PREFIXES_TO_STRIP:
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.startswith("model.model."):
        name = name[len("model."):]
    return name


def _set_submodule(model: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, replacement)
