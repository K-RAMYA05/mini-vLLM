from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple

from mini_vllm.block_manager import KVCache


@dataclass
class _Entry:
    block_id: int
    hits: int  # hit count for LFU; ignored under LRU


class PrefixCache:
    """Caches full prompt blocks by cumulative prefix digest.

    Two eviction policies:
      - 'lru' : standard recency-based. Cheap and good for single tenant.
      - 'lfu' : evict least-frequently-used; recency is a tiebreaker. Wins
                when a few prefixes (e.g. shared system prompts) are hit
                often and a flood of unique prompts would otherwise churn
                them out of an LRU cache.
    """

    def __init__(
        self,
        kv_cache: KVCache,
        block_size: int,
        max_entries: int,
        eviction: str = "lru",
    ):
        self.kv_cache = kv_cache
        self.block_size = block_size
        self.max_entries = max_entries
        if eviction not in ("lru", "lfu"):
            raise ValueError(f"eviction must be 'lru' or 'lfu' (got {eviction})")
        self.eviction = eviction
        # OrderedDict preserves recency for both policies (we re-insert on
        # touch so the most-recent entry is always at the end).
        self._entries: OrderedDict[bytes, _Entry] = OrderedDict()
        # Counters for observability / benchmarking.
        self.lookups = 0
        self.hits = 0

    def _extend_digest(self, prev: bytes, block_tokens: List[int]) -> bytes:
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(prev)
        hasher.update(struct.pack(f"<{len(block_tokens)}I", *block_tokens))
        return hasher.digest()

    def _touch(self, digest: bytes, entry: _Entry) -> None:
        entry.hits += 1
        self._entries.move_to_end(digest)

    def lookup(self, prompt_token_ids: List[int]) -> Tuple[List[int], int]:
        self.lookups += 1
        reusable_blocks = (max(len(prompt_token_ids) - 1, 0)) // self.block_size
        if reusable_blocks == 0 or self.max_entries == 0:
            return [], 0

        matched: List[int] = []
        digest = b""
        for block_idx in range(reusable_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            digest = self._extend_digest(digest, prompt_token_ids[start:end])
            entry = self._entries.get(digest)
            if entry is None:
                break
            matched.append(entry.block_id)
            self._touch(digest, entry)
        if matched:
            self.hits += 1
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
            existing = self._entries.get(digest)
            if existing is not None:
                self._touch(digest, existing)
                continue
            block_id = block_ids[block_idx]
            self.kv_cache.retain_blocks([block_id])
            self._entries[digest] = _Entry(block_id=block_id, hits=1)
            added += 1
            self._evict_if_needed()
        return added

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            victim_key = self._pick_victim()
            entry = self._entries.pop(victim_key)
            self.kv_cache.allocator.free(entry.block_id)

    def _pick_victim(self) -> bytes:
        if self.eviction == "lru":
            # Oldest = first item (move_to_end shifts touched entries to end).
            return next(iter(self._entries))
        # LFU: minimum hits, with insertion order (== oldest among ties) as
        # the tiebreaker. OrderedDict iteration is insertion order, so the
        # first entry we encounter at the minimum count is the right victim.
        min_hits = None
        victim = None
        for key, entry in self._entries.items():
            if min_hits is None or entry.hits < min_hits:
                min_hits = entry.hits
                victim = key
                if min_hits == 1:
                    break  # can't do better than this
        return victim

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0
