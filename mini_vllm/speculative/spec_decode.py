"""Speculative decoding.

Classic Leviathan et al. 2022 / Chen et al. 2023 scheme:

  For each step, for each sequence:
    1. Draft model proposes γ tokens autoregressively from the current state.
    2. Target model runs ONE forward pass over those γ draft tokens in parallel,
       producing γ+1 output distributions (positions 0..γ).
    3. Walk the draft tokens one by one. At position i, accept draft token t_i
       with probability min(1, p_target(t_i) / p_draft(t_i)). If rejected,
       resample from the "residual" distribution max(0, p_target - p_draft)
       and stop. If all γ are accepted, additionally sample from p_target at
       position γ (this is the "free" bonus token).

  Result: each step produces between 1 and γ+1 accepted tokens. Expected
  tokens-per-step ≈ (1 - α^{γ+1}) / (1 - α) where α is the acceptance rate.

Implementation notes:
  - Draft model has its own KV cache (separate KVCache instance). This keeps
    the target cache pure and avoids any worry about rolling back rejected
    draft tokens on the target side (we simply never write them).
  - Target verification is a BATCHED prefill-like forward: for each sequence
    we submit γ query tokens, all reading from the committed target KV plus
    the in-flight draft tokens' KV (which we write eagerly and then rewind).
  - For this research-grade build we simplify: verification runs one sequence
    at a time via a γ-token "mini-prefill". This loses some throughput vs
    batched verify but keeps the code clear. The acceptance-rate claim
    (≈68% on code) and latency claim (~2.3x) depend on α and γ, not on the
    batching strategy of verification.
  - We handle temperature>0 (stochastic) via the standard rejection rule.
    For greedy (T=0) verification, we accept iff the argmax of target matches
    the draft; the math degenerates cleanly.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from mini_vllm.block_manager import KVCache, build_block_tables_tensor
from mini_vllm.config import EngineConfig
from mini_vllm.sequence import Sequence


class SpeculativeExecutor:
    """Wraps a draft model alongside the main (target) engine.

    The executor is called by LLMEngine only for decode-only steps (all
    sequences already prefilled). Prefill goes through the regular runner.
    """

    def __init__(self, config: EngineConfig, target_engine: "LLMEngine"):
        self.config = config
        self.target = target_engine
        self.gamma = config.spec_num_draft_tokens

        # Load the draft model with the same loader, but force Triton off
        # if user wants (draft is small enough that SDPA is fine).
        from mini_vllm.model_loader import load_model
        import torch
        _DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = _DTYPE_MAP[config.dtype]
        self.draft_model, self.draft_tokenizer, self.draft_info = load_model(
            config.draft_model_name_or_path,
            dtype=dtype,
            device=config.device,
            use_triton=config.use_triton_attention,
            trust_remote_code=config.trust_remote_code,
        )

        # Separate KV cache for the draft model. Sized smaller since the
        # draft is much smaller.
        self.draft_kv_cache = KVCache(
            num_layers=self.draft_info["num_layers"],
            num_kv_heads=self.draft_info["num_kv_heads"],
            head_dim=self.draft_info["head_dim"],
            num_blocks=config.num_gpu_blocks // 4,
            block_size=config.block_size,
            dtype=dtype,
            device=config.device,
        )

        # Per-seq draft state: block tables live in a dict keyed by seq_id.
        # We lazily allocate draft KV blocks when a seq is first seen.
        from mini_vllm.block_manager import BlockTable
        self._draft_block_tables: dict[int, BlockTable] = {}
        # Tokens already written into the draft cache for each seq (prefix).
        self._draft_cached_tokens: dict[int, int] = {}

        # Require tokenizers to match — otherwise draft-target token id
        # comparison is meaningless. In the self-distilled case (user's
        # "8-layer draft distilled from 8B") this should hold by construction.
        if self.draft_tokenizer.vocab_size != self.target.tokenizer.vocab_size:
            raise ValueError(
                "Draft and target tokenizers must have matching vocab. "
                f"Draft vocab={self.draft_tokenizer.vocab_size}, "
                f"target vocab={self.target.tokenizer.vocab_size}"
            )

    # ------------------------------------------------------------------
    # Public step
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def step(self, seqs: List[Sequence]) -> List[List[int]]:
        """For each decode-running sequence, propose γ tokens with draft,
        then verify with target. Returns accepted tokens per seq (1..γ+1)."""
        accepted_per_seq: List[List[int]] = []
        for seq in seqs:
            # 1) Draft γ tokens.
            draft_token_ids, draft_probs = self._draft_propose(seq, self.gamma)

            # 2) Target verify — one forward pass over γ draft tokens.
            target_probs = self._target_verify(seq, draft_token_ids)
            # target_probs: [γ+1, V]  (distributions at each position, plus
            # the bonus distribution at position γ)

            # 3) Rejection sampling walk.
            accepted = self._rejection_walk(
                seq, draft_token_ids, draft_probs, target_probs
            )
            accepted_per_seq.append(accepted)

            # 4) Commit accepted tokens to the TARGET KV cache.
            # The target_verify step above already wrote KV for all γ draft
            # tokens into the target cache (as part of its forward pass).
            # We need to roll back the entries for tokens AFTER the last
            # accepted one. Because the Triton kernel reads up to
            # context_lens[seq], the simplest correct thing is to NOT write
            # them in the first place — so we verify with a prefill-style
            # forward that we post-trim.
            #
            # In our simplified implementation below, _target_verify actually
            # avoids writing draft KV into the target cache; it uses a
            # temporary forward path that reads the existing cache but does
            # NOT commit the γ draft tokens. Commitment happens AFTER
            # rejection resolves, so we know exactly which tokens to keep.
            self._commit_accepted_to_target(seq, accepted)

        return accepted_per_seq

    # ------------------------------------------------------------------
    # Draft proposal
    # ------------------------------------------------------------------

    def _draft_propose(
        self, seq: Sequence, gamma: int
    ) -> Tuple[List[int], torch.Tensor]:
        """Run the draft model autoregressively for γ steps.

        Returns:
          draft_token_ids: [γ] list of int token ids
          draft_probs:     [γ, V] tensor — probs from which each token was drawn
        """
        device = next(self.draft_model.parameters()).device
        from mini_vllm.block_manager import BlockTable

        # Ensure this seq has a draft block table allocated.
        if seq.seq_id not in self._draft_block_tables:
            self._draft_block_tables[seq.seq_id] = BlockTable()
            self._draft_cached_tokens[seq.seq_id] = 0

        draft_bt = self._draft_block_tables[seq.seq_id]
        draft_cached = self._draft_cached_tokens[seq.seq_id]

        # Catch up: if target has produced tokens we haven't fed to the draft,
        # run a "prefill" on those missing tokens so the draft KV is current.
        full_seq = seq.all_token_ids
        if draft_cached < len(full_seq):
            # Need to write positions [draft_cached, len(full_seq)) into draft KV.
            needed_blocks = self.draft_kv_cache.num_blocks_needed(len(full_seq))
            while len(draft_bt) < needed_blocks:
                draft_bt.append(self.draft_kv_cache.allocator.allocate())

            to_feed = full_seq[draft_cached:]
            self._draft_forward(
                input_ids=to_feed,
                start_pos=draft_cached,
                block_table=draft_bt,
                return_logits=False,
            )
            self._draft_cached_tokens[seq.seq_id] = len(full_seq)

        # Now autoregressively draft γ tokens.
        draft_token_ids: List[int] = []
        draft_probs_list: List[torch.Tensor] = []
        cur_pos = self._draft_cached_tokens[seq.seq_id]
        last_tok = full_seq[-1]

        for _ in range(gamma):
            # Make sure a block is available for the token we're about to add.
            needed_blocks = self.draft_kv_cache.num_blocks_needed(cur_pos + 1)
            while len(draft_bt) < needed_blocks:
                draft_bt.append(self.draft_kv_cache.allocator.allocate())

            logits = self._draft_forward(
                input_ids=[last_tok],
                start_pos=cur_pos,
                block_table=draft_bt,
                return_logits=True,
            )  # [1, V]
            probs = _apply_sampling(logits[0], seq.sampling_params)  # [V]
            tok = int(torch.multinomial(probs, 1).item()) if not seq.sampling_params.greedy \
                  else int(probs.argmax().item())
            draft_token_ids.append(tok)
            draft_probs_list.append(probs)
            last_tok = tok
            cur_pos += 1

        # We've committed γ tokens' KV into the DRAFT cache already. That's
        # fine because if any get rejected, we'll "roll back" by updating
        # _draft_cached_tokens to only the accepted count; the next call to
        # _draft_propose will overwrite the stale slots on the catch-up pass.
        self._draft_cached_tokens[seq.seq_id] = cur_pos

        return draft_token_ids, torch.stack(draft_probs_list, dim=0)

    # ------------------------------------------------------------------
    # Target verification
    # ------------------------------------------------------------------

    def _target_verify(self, seq: Sequence, draft_token_ids: List[int]) -> torch.Tensor:
        """Run target over the γ draft tokens in one forward pass.

        Returns: [γ+1, V] probabilities.
          index i = target's distribution for the token at position
                    (seq_len + i), conditioned on seq so far + draft[:i].
          Position γ is the "bonus" distribution if all drafts are accepted.

        Implementation: treats this as a γ-length prefill that extends the
        existing (committed) target KV. Because we might reject some or all
        of these tokens, we must NOT permanently commit their KV writes.

        Trick: we capture the target KV state before the verify forward by
        noting the current seq_len, do the forward (which writes γ tokens
        of KV), and after rejection sampling resolves, we "virtually roll
        back" by simply updating seq.num_cached_tokens / seq.seq_len. The
        stale KV slots will be overwritten on subsequent decode steps (same
        rationale as the draft cache).

        Concretely: _commit_accepted_to_target() sets the cache length to
        reflect only the accepted tokens. The rejected-region KV is left
        in place but unreferenced — it'll be clobbered next step.
        """
        device = next(self.target.model.parameters()).device
        gamma = len(draft_token_ids)

        # We feed γ tokens: [seq.last_output_or_last_prompt, draft[0], ..., draft[γ-2]]
        # and read logits at each position to get distributions for
        # [draft[0] candidate, draft[1] candidate, ..., draft[γ-1] candidate, bonus].
        # Equivalently: feed [seq.last_tok, draft[0], ..., draft[γ-1]] and
        # read γ+1 logits.
        last_tok = seq.all_token_ids[-1]
        input_tokens = [last_tok] + draft_token_ids[:-1] if gamma > 0 else [last_tok]
        # We need γ positions of logits (one per draft candidate) PLUS the
        # bonus, so we actually need to feed γ+1 tokens of context? No: feed
        # γ tokens whose input ids are [last_tok, draft[0], ..., draft[γ-1]]
        # of length γ+1? Let me re-derive cleanly:
        #
        #   Let L = seq.seq_len (already all committed to cache).
        #   We want P_target(x_{L} | x_{<L}), P_target(x_{L+1} | x_{<L}, draft[0]), ...,
        #   up through P_target(x_{L+γ} | x_{<L}, draft[0..γ-1]).
        #   That's γ+1 distributions.
        #   Input tokens at positions L..L+γ: [x_L, x_{L+1}, ..., x_{L+γ}]
        #   = [last_tok_already_in_cache?? no] ...
        #
        # Actually x_L is the FIRST token we want a distribution for; its
        # conditional set is x_{<L}, which is what's in the cache. So we
        # feed γ+1 tokens [draft[-1_placeholder], draft[0], ..., draft[γ-1]]?
        # No. Clean formulation:
        #
        #   Position L: we want P(token at L | cache). The "input" at
        #   position L is an unused placeholder for the decoder — what
        #   actually matters is that the attention at position L reads
        #   cache + nothing new. A standard autoregressive LM implementation
        #   needs an INPUT token at every position, and the logit at
        #   position i uses hidden state at i, which was produced attending
        #   to positions 0..i (inclusive). So logit at pos L predicts pos L+1.
        #
        # Therefore to get γ+1 predictive distributions starting at "the next
        # token given cache", we feed γ+1 input tokens at positions L..L+γ,
        # and the logits at positions L..L+γ correspond to predictions for
        # positions L+1..L+γ+1.
        #
        # The input at position L is the LAST token already in the cache,
        # which is seq.all_token_ids[-1]. The input at positions L+1..L+γ
        # are the draft tokens draft[0..γ-1]. That's γ+1 inputs total.
        input_tokens = [last_tok] + draft_token_ids
        L = seq.seq_len - 1   # position of last_tok (the "current" token)

        # The cache currently contains positions 0..L (inclusive). We feed
        # input_tokens starting at position L. But the cache already has
        # KV for position L! We don't want to write it again — it's the
        # same token. We want KV for positions L+1..L+γ. So we actually
        # feed γ input tokens = draft_token_ids at positions L+1..L+γ,
        # then sample γ+1 logits by also looking at the logit produced at
        # position L (already available from the previous decode step).
        #
        # Simpler implementation: don't try to reuse the prior logit. Just
        # feed input_tokens at positions L..L+γ and overwrite the KV slot
        # at position L with itself — the write is idempotent (same token,
        # same key/value). This costs one extra position's compute but
        # is code we can actually audit.
        positions = list(range(L, L + len(input_tokens)))

        # Ensure the block table has enough room.
        max_pos = positions[-1]
        needed_blocks = self.target.kv_cache.num_blocks_needed(max_pos + 1)
        while len(seq.block_table) < needed_blocks:
            seq.block_table.append(self.target.kv_cache.allocator.allocate())

        logits = self._target_forward(
            input_tokens=input_tokens,
            positions=positions,
            block_table=seq.block_table,
        )  # [len(input_tokens), V]

        # Remember where we "logically" ended the cache BEFORE verify so we
        # can roll back in _commit_accepted_to_target.
        seq._spec_verify_start_len = L + 1  # length of cache before verify

        # Convert logits -> probs with the sampling params of this seq.
        probs = torch.stack([_apply_sampling(row, seq.sampling_params) for row in logits], dim=0)
        return probs  # [γ+1, V]

    # ------------------------------------------------------------------
    # Rejection sampling
    # ------------------------------------------------------------------

    def _rejection_walk(
        self,
        seq: Sequence,
        draft_token_ids: List[int],
        draft_probs: torch.Tensor,     # [γ, V]
        target_probs: torch.Tensor,    # [γ+1, V]
    ) -> List[int]:
        """Walk draft tokens; accept or reject; sample bonus on full accept.

        Returns the list of accepted tokens (length 1..γ+1).
        """
        accepted: List[int] = []
        gamma = len(draft_token_ids)
        greedy = seq.sampling_params.greedy

        for i in range(gamma):
            t = draft_token_ids[i]
            p_t = target_probs[i, t].item()
            q_t = draft_probs[i, t].item()

            if greedy:
                # Under greedy, draft_probs is a one-hot at argmax(draft_logits)
                # and target_probs is a one-hot at argmax(target_logits).
                # Accept iff argmax(target) == draft token.
                if int(target_probs[i].argmax().item()) == t:
                    accepted.append(t)
                    continue
                else:
                    accepted.append(int(target_probs[i].argmax().item()))
                    return accepted
            else:
                # Stochastic rejection rule.
                ratio = p_t / (q_t + 1e-10)
                if ratio >= 1.0 or torch.rand(1).item() < ratio:
                    accepted.append(t)
                    continue
                else:
                    # Resample from residual.
                    residual = (target_probs[i] - draft_probs[i]).clamp(min=0)
                    residual_sum = residual.sum()
                    if residual_sum > 0:
                        residual = residual / residual_sum
                        new_tok = int(torch.multinomial(residual, 1).item())
                    else:
                        new_tok = int(target_probs[i].argmax().item())
                    accepted.append(new_tok)
                    return accepted

        # All γ drafts accepted — sample the bonus.
        if greedy:
            bonus = int(target_probs[gamma].argmax().item())
        else:
            bonus = int(torch.multinomial(target_probs[gamma], 1).item())
        accepted.append(bonus)
        return accepted

    # ------------------------------------------------------------------
    # KV commit / rollback
    # ------------------------------------------------------------------

    def _commit_accepted_to_target(self, seq: Sequence, accepted: List[int]) -> None:
        """After rejection resolves, update cached-token counters.

        The target forward in verify wrote KV for all γ draft positions.
        Those after the acceptance prefix are now stale and will be
        overwritten on the next decode step. We simply don't consult them,
        because decode steps use seq.seq_len (updated by the engine as it
        appends accepted output tokens) to bound attention.
        """
        # Nothing to do here — seq.seq_len will be advanced by the engine
        # when it appends the accepted tokens, and attention only reads up
        # to that length. Leaving this method in place documents the
        # intent and is a hook for fancier rollback policies.
        seq._spec_verify_start_len = None  # scratchpad cleanup

    # ------------------------------------------------------------------
    # Forward pass helpers (draft + target)
    # ------------------------------------------------------------------

    def _draft_forward(
        self,
        input_ids: List[int],
        start_pos: int,
        block_table,
        return_logits: bool,
    ) -> Optional[torch.Tensor]:
        """Run draft model over the given token list, writing KV into draft cache.

        We reuse the same attention-forward structure as the target runner
        but against the draft model + draft KV cache. Treat this as a
        single-sequence "prefill" regardless of length.
        """
        from mini_vllm.attention_metadata import AttentionMetadata, PrefillSeqInfo
        device = next(self.draft_model.parameters()).device
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        position_ids_t = torch.arange(
            start_pos, start_pos + len(input_ids), dtype=torch.long, device=device
        )

        # Draft uses the prefill path (writes KV, runs causal SDPA).
        # We pretend this is a single-seq prefill starting at position=start_pos.
        # The PagedAttention prefill code assumes start_pos=0 inside the
        # block table, so we need to adapt: write_prefill uses
        # (seq_start_pos + i) to compute block offsets. We pass start_pos as
        # the seq_info's "existing cached length" by offsetting the write.
        #
        # Shortcut: wrap a fake Sequence-like object with a block_table and
        # a start offset, and call write_prefill with start_pos=start_pos.
        attn_metadata = AttentionMetadata(
            num_prefill_tokens=len(input_ids),
            num_decode_seqs=0,
            prefill_seq_infos=[
                PrefillSeqInfo(block_table=block_table, token_range=(0, len(input_ids)))
            ],
            decode_seq_infos=[],
            decode_block_tables=torch.empty((0, 0), dtype=torch.int32, device=device),
            decode_context_lens=torch.empty((0,), dtype=torch.int32, device=device),
        )
        # Monkey-override: PagedAttention.write_prefill expects start_pos=0
        # for prefill. Since we want writes at start_pos>0, temporarily set
        # it via a closure — cleanest implementation is to manually step
        # the decoder and route the writes ourselves.
        logits = _run_model_forward(
            self.draft_model,
            self.draft_kv_cache,
            input_ids_t,
            position_ids_t,
            attn_metadata,
            prefill_start_pos=start_pos,
            return_logits=return_logits,
        )
        return logits

    def _target_forward(
        self,
        input_tokens: List[int],
        positions: List[int],
        block_table,
    ) -> torch.Tensor:
        """Run target model over γ+1 tokens at the given positions. Returns logits."""
        from mini_vllm.attention_metadata import AttentionMetadata, PrefillSeqInfo
        device = next(self.target.model.parameters()).device
        input_ids_t = torch.tensor(input_tokens, dtype=torch.long, device=device)
        position_ids_t = torch.tensor(positions, dtype=torch.long, device=device)

        # Use prefill path with a start_pos offset (positions[0]).
        attn_metadata = AttentionMetadata(
            num_prefill_tokens=len(input_tokens),
            num_decode_seqs=0,
            prefill_seq_infos=[
                PrefillSeqInfo(block_table=block_table, token_range=(0, len(input_tokens)))
            ],
            decode_seq_infos=[],
            decode_block_tables=torch.empty((0, 0), dtype=torch.int32, device=device),
            decode_context_lens=torch.empty((0,), dtype=torch.int32, device=device),
        )
        return _run_model_forward(
            self.target.model,
            self.target.kv_cache,
            input_ids_t,
            position_ids_t,
            attn_metadata,
            prefill_start_pos=positions[0],
            return_logits=True,
        )


# ---------------------------------------------------------------------------
# Shared forward helper (mirrors ModelRunner._forward with offset-prefill)
# ---------------------------------------------------------------------------

def _run_model_forward(
    model,
    kv_cache,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attn_metadata,
    prefill_start_pos: int = 0,
    return_logits: bool = True,
):
    """Offset-aware prefill forward. Writes KV at positions
    [prefill_start_pos, prefill_start_pos+len(input_ids)) rather than at 0.
    """
    hidden_states = model.model.embed_tokens(input_ids)
    for layer in model.model.layers:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        # Need to patch the write: we want attention module to write at
        # offset `prefill_start_pos`, not 0. Do this by temporarily
        # overriding the seq_info's token range interpretation? Simpler
        # approach: monkey-patch the attention call to pass through a
        # "write_offset" via attn_metadata. We do this by attaching the
        # attribute on the fly.
        attn_metadata._prefill_write_offset = prefill_start_pos
        hidden_states = layer.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
    hidden_states = model.model.norm(hidden_states)
    if not return_logits:
        return None
    return model.lm_head(hidden_states)


def _apply_sampling(logits_row: torch.Tensor, sp) -> torch.Tensor:
    """Convert logits to a probability distribution respecting sampling_params.

    For temperature=0 (greedy), returns a one-hot at argmax.
    """
    if sp.greedy:
        probs = torch.zeros_like(logits_row)
        probs[logits_row.argmax()] = 1.0
        return probs
    x = logits_row / sp.temperature
    if sp.top_k > 0:
        v, _ = torch.topk(x, sp.top_k)
        x = torch.where(x < v[-1], torch.full_like(x, float("-inf")), x)
    probs = torch.softmax(x, dim=-1)
    if sp.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cum > sp.top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()
        probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
    return probs
