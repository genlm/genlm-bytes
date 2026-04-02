import asyncio
import torch
import numpy as np
from arsenal import colors
from dataclasses import dataclass
from functools import cached_property
from genlm.backend.tokenization.bytes import get_byte_vocab

from ..util import logsumexp, LazyByteProbs
from ..trie import AsyncTokenByteTrie
from .trie_state import LazyTrieState, TrieMode
from .lm_state import StatefulByteLM
from .heal import TokenHealer


@dataclass
class BeamParams:
    """Parameters for byte-level beam summing algorithm.

    Args:
        K (int): Beam width - maximum number of candidates to maintain.
        prune_threshold (float, optional): Probability threshold for pruning candidates.
            Candidates with probability below this are removed. Defaults to 0.0
        verbose (bool, optional): Whether to print the beam state at each step. Defaults to False
        eos_tokens (list[bytes], optional): List of tokens that should be treated as EOS. When configured,
            EOS tokens will terminate generation when sampled. Defaults to None
        heal (bool, optional): Whether to enable adaptive token healing. Defaults to True
        heal_max_backoff (int, optional): Maximum number of bytes to back off when healing. Defaults to None
        heal_max_splits (int, optional): Maximum number of intra-suffix commits allowed during a single healing attempt. Defaults to None
    """

    K: int
    prune_threshold: float = 0.0
    verbose: bool = False
    eos_tokens: list[bytes] = None
    heal: bool = True
    heal_max_backoff: int | None = None
    # Optional cap on how many intra-partial commits are allowed during a
    # single healing attempt. None means unlimited. Set to 0 to disable
    # multi-split behavior (i.e., single-split only).
    heal_max_splits: int | None = None

    def __post_init__(self):
        if self.prune_threshold < 0:
            raise ValueError(
                f"prune_threshold must be non-negative, got {self.prune_threshold}"
            )
        self.log_prune_threshold = (
            np.log(self.prune_threshold) if self.prune_threshold > 0 else -np.inf
        )
        self.eos_tokens = set(self.eos_tokens) if self.eos_tokens else set()


class ByteBeamState(StatefulByteLM):
    """Represents the state of the beam during byte-level language modeling.

    Tracks multiple candidate states and their probabilities, pruning low-probability
    candidates.

    Args:
        states (list[LazyTrieState]): List of candidate states to track
        params (BeamParams): Parameters controlling beam search behavior
    """

    def __init__(self, states, params):
        self.states = sorted(states, key=lambda b: -b.weight)
        self.params = params

    @classmethod
    async def initial(cls, llm, params, trie_opts=None):
        """Creates initial beam state.

        Args:
            llm (StatefulTokenizedLM): Token-level language model to use.
            params (BeamParams): Beam search parameters.
            trie_opts (dict, optional): Additional keyword arguments passed to
                AsyncTokenByteTrie.from_vocab. For example, {"max_batch_size": 100}.

        Returns:
            (ByteBeamState): Initial beam state.
        """
        # Handle EOS tokens
        trie_opts = trie_opts or {}
        trie_opts["eos_tokens"] = params.eos_tokens

        async_trie = AsyncTokenByteTrie.from_vocab(
            get_byte_vocab(llm.tokenizer), **trie_opts
        )
        state = LazyTrieState.initial(llm, async_trie, mode=TrieMode.WITH_EOS)
        return cls([await state.materialize()], params)

    def __iter__(self):
        return iter(self.states)

    def __len__(self):
        return len(self.states)

    @cached_property
    def logZ(self):
        """Estimate of the partition function (sum of weights) for current beam.
        This is the estimate of the prefix probability of the bytes consumed so far.
        """
        return logsumexp([state.weight for state in self])

    async def __lshift__(self, a):
        """Advances the beam state with a new byte.

        Args:
            a (int): Byte to add to states.

        Returns:
            (ByteBeamState): New beam state after processing the byte.
        """
        new_states = []
        for state in self:
            if new_state := state << a:
                new_states.append(new_state)

        logZ = logsumexp([s.weight for s in new_states]) if new_states else -np.inf
        for state in await self.extend(logZ):
            if new_state := state << a:
                new_states.append(new_state)

        new_state = ByteBeamState(new_states, self.params)

        # If advancing would empty the beam, do adaptive healing if enabled
        if self.params.heal and len(new_state) == 0:
            healed = await self._adaptive_heal(a)
            if healed is not None:
                if self.params.verbose:
                    print("[heal] Applied adaptive token healing")
                return healed

        if self.params.verbose:
            print()
            print(new_state)

        return new_state

    async def logp_next(self):
        """Computes log probabilities for the next byte across all beam candidates.

        Returns:
            (LazyByteProbs): Log probabilities for next possible bytes.
        """
        assert len(self) > 0, "Beam is empty"

        logqs = []
        for state in self:
            logqs.append(state.logp_next.ps + state.weight)

        for state in await self.extend(self.logZ):
            logqs.append(state.logp_next.ps + state.weight)

        logqs = torch.stack(logqs, dim=0)  # shape: (num_states, 258)
        # mask EOT positions of non-extended (EOT is at index 256)
        logqs[: len(self), -2] = -float('inf')
        logps = torch.logsumexp(logqs, dim=0)

        return LazyByteProbs(logps - torch.logsumexp(logps, dim=0))

    async def extend(self, logZ):
        """Attempts to advance each candidate in the beam by a token (EOT).

        For each candididate with EOT available, this ends the current token and
        starts a new one in preparation for the next byte.

        Args:
            logZ (float): Current estimated of the partition function for pruning

        Returns:
            (list[LazyTrieState]): New candidate states after extension
        """
        extends = []
        for state in self:
            if new_state := state.extend():
                logZ = np.logaddexp(logZ, new_state.weight)
                extends.append(new_state)

        to_materialize = [
            state for state in extends
            if state.weight - logZ > self.params.log_prune_threshold
        ]

        return await LazyTrieState.batch_materialize(to_materialize)

    def prune(self):
        """Prunes beam to maintain beam width and probability threshold.

        Returns:
            (ByteBeamState): New state with pruned candidates.
        """
        new_states = [
            state
            for state in self
            if state.weight - self.logZ > self.params.log_prune_threshold
        ][: self.params.K]
        return ByteBeamState(new_states, self.params)

    def __repr__(self):
        desc = colors.bold % f"Z: {self.logZ}\n" + colors.bold % "Candidates:\n"
        for state in self:
            P = np.exp(state.weight - self.logZ)
            color = colors.green if P > self.params.prune_threshold else colors.red
            desc += f"({color % f'{P:.4f}'}) {repr(state)}\n"
        return desc

    def with_mode(self, mode):
        """Create a new beam state with specified trie mode.

        Args:
            mode (TrieMode): Trie mode for the new beam state

        Returns:
            (ByteBeamState): New beam state with updated mode
        """
        return ByteBeamState(
            states=[state.with_mode(mode) for state in self.states],
            params=self.params,
        )

    async def prefill(self, bs):
        """Prefill the beam on a sequence of bytes.

        During prefilling, EOS tokens are treated as normal tokens and don't cause termination.
        Uses batched prefill to minimize LLM calls by tracing trie paths ahead of time
        and batching all materialization calls together.

        Args:
            bs (bytes): Byte sequence to prefill on

        Returns:
            (ByteBeamState): New beam state after prefilling
        """
        no_eos_beam = self.with_mode(TrieMode.WITHOUT_EOS)
        no_eos_beam = await no_eos_beam._batched_prefill(bs)
        return no_eos_beam.with_mode(TrieMode.WITH_EOS)

    async def _batched_prefill(self, bs):
        """Batched prefill: trace trie paths, batch LLM calls, propagate weights.

        Instead of processing bytes one-at-a-time (each potentially triggering K LLM calls),
        we trace all candidates through the trie for multiple bytes, identify all token
        boundaries, then batch all LLM calls together.
        """
        if len(bs) == 0:
            return self

        trie = self.states[0].trie
        trie_data = trie.trie
        children = trie_data.children
        eot = trie_data.eot_token
        root = trie_data.root
        K = self.params.K
        candidate_budget = max(4 * K, 16)
        mode = self.states[0].mode

        # Candidates: list of (lm_state, node, weight, mass)
        candidates = [
            (s.lm_state, s.node, s.weight, s._mass)
            for s in self.states
        ]

        pos = 0
        while pos < len(bs):
            # Phase 1: Trace through bytes without LLM calls.
            # Each candidate is a 4-tuple: (lm_state, node, weight, mass)
            # Weight is the parent's weight (deferred update). Mass is inherited
            # from parent for descended states, or None for extended states.
            # We also track: parent_node (for weight calc) and optionally mat_idx
            # (index into needs_materialize for extended states).
            needs_materialize = []
            # needs_materialize entries: (lm_state, eot_node, parent_node, parent_weight, parent_mass)

            # Traced candidates: 4-tuple + metadata for weight propagation
            # Format: (lm_st, node, weight, mass, parent_node, mat_idx_or_None)
            # mat_idx_or_None: None for descended, int for extended
            traced = [(lm, nd, w, m, nd, None) for lm, nd, w, m in candidates]
            chunk_start = pos

            while pos < len(bs):
                b = bs[pos]
                next_traced = []

                for lm_st, node, weight, mass, _pnode, _midx in traced:
                    child = children[node].get(b)

                    # Path A: Descend
                    if child is not None:
                        next_traced.append((lm_st, child, weight, mass, node, None))

                    # Path B: Extend + descend from root
                    eot_node = children[node].get(eot)
                    if eot_node is not None:
                        token_id = int(trie_data.leaf2token_id[eot_node])
                        new_lm_st = lm_st << token_id
                        mat_idx = len(needs_materialize)
                        needs_materialize.append(
                            (new_lm_st, eot_node, node, weight, mass)
                        )
                        root_child = children[root].get(b)
                        if root_child is not None:
                            next_traced.append(
                                (new_lm_st, root_child, weight, None, node, mat_idx)
                            )

                if not next_traced:
                    # All candidates stuck — healing needed
                    break

                # Budget enforcement
                if len(next_traced) > candidate_budget:
                    next_traced.sort(key=lambda c: -c[2])
                    next_traced = next_traced[:candidate_budget]

                traced = next_traced
                pos += 1

                if len(needs_materialize) > candidate_budget:
                    break

            # Phase 2: Batch materialize
            mat_masses = [None] * len(needs_materialize)
            if needs_materialize:
                context_map = {}
                for mat_idx, (lm_st, eot_nd, parent_nd, parent_w, parent_mass) in enumerate(needs_materialize):
                    ctx_key = tuple(lm_st.context)
                    if ctx_key not in context_map:
                        context_map[ctx_key] = (lm_st, [])
                    context_map[ctx_key][1].append(mat_idx)

                unique_contexts = list(context_map.values())
                logp_nexts = await asyncio.gather(
                    *[lm_st.logp_next() for lm_st, _ in unique_contexts]
                )

                ws_batch = [torch.exp(logp) for logp in logp_nexts]
                batch_masses = trie_data.batch_weight_sum(ws_batch, mode=mode)

                for (_, mat_indices), mass_tensor in zip(unique_contexts, batch_masses):
                    log_mass = torch.log(mass_tensor)
                    for idx in mat_indices:
                        mat_masses[idx] = log_mass

            # Phase 3: Propagate weights
            final_candidates = []
            for lm_st, node, parent_weight, mass, parent_node, mat_idx in traced:
                if mat_idx is None:
                    # Descended: use inherited mass
                    if mass is not None:
                        weight = parent_weight + (mass[node] - mass[parent_node]).item()
                    else:
                        weight = parent_weight
                    final_candidates.append((lm_st, node, weight, mass))
                else:
                    # Extended: compute EOT weight from parent mass, descent from new mass
                    _, eot_nd, orig_parent_nd, _, parent_mass = needs_materialize[mat_idx]
                    new_mass = mat_masses[mat_idx]
                    if parent_mass is not None and new_mass is not None:
                        eot_w = parent_weight + (parent_mass[eot_nd] - parent_mass[orig_parent_nd]).item()
                        weight = eot_w + (new_mass[node] - new_mass[root]).item()
                    else:
                        weight = parent_weight
                    final_candidates.append((lm_st, node, weight, new_mass))

            # Prune
            if final_candidates:
                final_candidates.sort(key=lambda c: -c[2])
                logZ_val = logsumexp([c[2] for c in final_candidates])
                final_candidates = [
                    c for c in final_candidates
                    if c[2] - logZ_val > self.params.log_prune_threshold
                ][:K]

            candidates = final_candidates

            # Empty beam: healing fallback — process remaining bytes sequentially
            if not candidates:
                beam = await self._rebuild_beam(
                    [(s.lm_state, s.node, s.weight, s._mass) for s in self.states],
                    trie, mode,
                )
                for b_val in bs[chunk_start:]:
                    beam = await (beam.prune() << b_val)
                return beam

        return await self._rebuild_beam(candidates, trie, mode)

    async def _rebuild_beam(self, candidates, trie, mode):
        """Convert traced candidates back to LazyTrieState objects.

        Materializes any candidates that lack mass tensors.
        """
        states = []
        for lm_st, node, weight, mass in candidates:
            state = LazyTrieState(
                lm_state=lm_st,
                trie=trie,
                node=node,
                weight=weight,
                mass=mass,
                mode=mode,
            )
            states.append(state)
        # Materialize any states missing mass
        await LazyTrieState.batch_materialize(states)
        return ByteBeamState(states, self.params)

    async def cleanup(self):
        """Cleans up resources used by the candidates."""
        await asyncio.gather(*[state.cleanup() for state in self])

    async def _adaptive_heal(self, next_byte: int):
        """Attempt adaptive token healing using TokenHealer.

        Returns a new beam advanced by `next_byte` if healing succeeds, else None.
        """
        healer = TokenHealer(
            max_backoff=self.params.heal_max_backoff,
            max_splits=self.params.heal_max_splits,
            verbose=self.params.verbose,
        )

        for state in self.states:
            healed_state = await healer.try_heal(state, next_byte)
            if healed_state is not None:
                return ByteBeamState([healed_state], self.params)

        return None
