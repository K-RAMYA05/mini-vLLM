from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from typing import List, Tuple

from mini_vllm.block_manager import KVCache


class PrefixCache:
    """Caches full prompt blocks by cumulative prefix digest."""

    def __init__(self, kv_cache: KVCache, block_size: int, max_entries: int):
        self.kv_cache = kv_cache
        self.block_size = block_size
        self.max_entries = max_entries
        self._entries: OrderedDict[bytes, int] = OrderedDict()

    def _extend_digest(self, prev: bytes, block_tokens: List[int]) -> bytes:
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(prev)
        hasher.update(struct.pack(f"<{len(block_tokens)}I", *block_tokens))
        return hasher.digest()

    def lookup(self, prompt_token_ids: List[int]) -> Tuple[List[int], int]:
        reusable_blocks = (max(len(prompt_token_ids) - 1, 0)) // self.block_size
        if reusable_blocks == 0 or self.max_entries == 0:
            return [], 0

        matched: List[int] = []
        digest = b""
        for block_idx in range(reusable_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            digest = self._extend_digest(digest, prompt_token_ids[start:end])
            block_id = self._entries.get(digest)
            if block_id is None:
                break
            matched.append(block_id)
            self._entries.move_to_end(digest)
        return matched, len(matched) * self.block_size

    def register(self, prompt_token_ids: List[int], block_ids: List[int]) -> int:
        reusable_blocks = (max(len(prompt_token_ids) - 1, 0)) // self.block_size
        if reusable_blocks == 0 or self.max_entries == 0:
            return 0

        added = 0
        digest = b""
        for block_idx in range(min(reusable_blocks, len(block_ids))):
            start = block_idx * self.block_size
            end = start + self.block_size
            digest = self._extend_digest(digest, prompt_token_ids[start:end])
            if digest in self._entries:
                self._entries.move_to_end(digest)
                continue
            block_id = block_ids[block_idx]
            self.kv_cache.retain_blocks([block_id])
            self._entries[digest] = block_id
            added += 1
            self._evict_if_needed()
        return added

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            _, block_id = self._entries.popitem(last=False)
            self.kv_cache.allocator.free(block_id)

    @property
    def num_entries(self) -> int:
        return len(self._entries)
