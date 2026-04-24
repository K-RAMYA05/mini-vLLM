"""Dataset for sharded teacher-generated distillation data.

Lazy-loads shards on demand; holds no more than `max_cached_shards` in
memory at any time. Random access via (shard_idx, within_shard_idx).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class DistillDataset(Dataset):
    def __init__(self, data_dir: str, max_cached_shards: int = 2):
        self.data_dir = Path(data_dir)
        self.shard_paths = sorted(self.data_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files found in {data_dir}")

        # Index every example: (shard_idx, within_shard_idx).
        self.index: List[tuple[int, int]] = []
        self.shard_sizes: List[int] = []
        for si, path in enumerate(self.shard_paths):
            # Peek size without loading everything by checking input_ids shape.
            head = torch.load(path, map_location="cpu", weights_only=False)
            n = head["input_ids"].shape[0]
            self.shard_sizes.append(n)
            for i in range(n):
                self.index.append((si, i))
            # Release immediately — we may or may not cache it below.
            del head

        self.max_cached_shards = max_cached_shards
        self._cache: Dict[int, dict] = {}
        self._cache_order: List[int] = []

    def __len__(self) -> int:
        return len(self.index)

    def _get_shard(self, si: int) -> dict:
        if si in self._cache:
            return self._cache[si]
        data = torch.load(self.shard_paths[si], map_location="cpu", weights_only=False)
        self._cache[si] = data
        self._cache_order.append(si)
        while len(self._cache) > self.max_cached_shards:
            evict = self._cache_order.pop(0)
            del self._cache[evict]
        return data

    def __getitem__(self, idx: int) -> dict:
        si, wi = self.index[idx]
        shard = self._get_shard(si)
        return {
            "input_ids": shard["input_ids"][wi],          # [T]
            "topk_values": shard["topk_values"][wi],      # [T, K]
            "topk_indices": shard["topk_indices"][wi],    # [T, K]
        }


def collate_distill(batch: List[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "topk_values": torch.stack([b["topk_values"] for b in batch]),
        "topk_indices": torch.stack([b["topk_indices"] for b in batch]),
    }
