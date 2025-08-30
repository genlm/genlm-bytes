import asyncio
import numpy as np
from arsenal import colors
from dataclasses import dataclass
from scipy.special import logsumexp as scipy_logsumexp
from functools import cached_property
from genlm.backend.tokenization.bytes import get_byte_vocab

from ..util import logsumexp, LazyByteProbs
from ..trie import AsyncTokenByteTrie
from .trie_state import LazyTrieState, TrieMode
from .lm_state import StatefulByteLM


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
    """

    K: int
    prune_threshold: float = 0.0
    verbose: bool = False
    eos_tokens: list[bytes] = None
    heal: bool = True
    heal_max_backoff: int | None = None
    # When true, ensure that every canonical tokenization (per BPE canonicality)
    # remains on the beam even if it would be pruned by K/threshold.
    keep_all_canonical: bool = False

    def __post_init__(self):
        if self.prune_threshold < 0:
            raise ValueError(
                f"prune_threshold must be non-negative, got {self.prune_threshold}"
            )
        self.log_prune_threshold = (
            np.log(self.prune_threshold) if self.prune_threshold > 0 else -np.inf
        )
        self.eos_tokens = set(self.eos_tokens) if self.eos_tokens else {}
        # Placeholder for optional canonicality helper (populated in ByteBeamState.initial)
        self._canonical_filter = None


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
        # Optionally prepare a canonicality filter using genlm-control if requested
        if getattr(params, "keep_all_canonical", False):
            try:
                # Import locally to avoid a hard dependency at install time
                from genlm.control.potential.built_in.canonical import (
                    FastCanonicalityFilterBPE,
                )

                filt = FastCanonicalityFilterBPE.from_tokenizer(llm.tokenizer)
                # Best-effort overrides for some tokenizers (e.g., GPT‑2)
                try:  # pragma: no cover - depends on tokenizer
                    name = getattr(llm.tokenizer, "name_or_path", None)
                    if name:
                        filt.set_overrides(name)
                except Exception:
                    pass
                params._canonical_filter = filt
            except Exception as e:  # pragma: no cover - optional feature path
                raise RuntimeError(
                    "keep_all_canonical=True requires genlm-control's canonical module."
                ) from e
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
                if self.params.verbose:  # pragma: no cover - diagnostics only
                    print("[heal] Applied adaptive token healing inside __lshift__")
                return healed

        if self.params.verbose:
            print()
            print(f"[beam] size={len(new_state)} (K={self.params.K})")
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

        logqs = np.stack(logqs, axis=0)  # shape: (num_states, array_size)
        # mask EOT positions of non-extended (EOT is at index 256)
        logqs[: len(self), -2] = -np.inf
        logps = scipy_logsumexp(logqs, axis=0)

        return LazyByteProbs(logps - logsumexp(logps))

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

        coros = []
        for state in extends:
            if state.weight - logZ > self.params.log_prune_threshold:
                coros.append(state.materialize())

        return await asyncio.gather(*coros)

    def prune(self):
        """Prunes beam to maintain beam width and probability threshold.

        Returns:
            (ByteBeamState): New state with pruned candidates.
        """
        # Standard K/threshold pruning
        new_states = [
            state
            for state in self
            if state.weight - self.logZ > self.params.log_prune_threshold
        ][: self.params.K]
        # Optionally union with all canonical tokenizations so far
        if getattr(self.params, "keep_all_canonical", False) and getattr(
            self.params, "_canonical_filter", None
        ) is not None:
            canonicals = []
            seen = set()

            def key_of(s):
                # Uniqueness by (context, node) pair
                return (tuple(s.lm_state.context), int(s.node))

            for s in self.states:
                if self._is_canonical_state(s):
                    k = key_of(s)
                    if k not in seen:
                        seen.add(k)
                        canonicals.append(s)

            # Merge and deduplicate with the pruned set
            merged = []
            seen = set()
            for s in new_states + canonicals:
                k = key_of(s)
                if k not in seen:
                    seen.add(k)
                    merged.append(s)
            return ByteBeamState(merged, self.params)

        return ByteBeamState(new_states, self.params)

    def _is_canonical_state(self, s):
        """Check if a state's completed tokenization is canonical.

        Uses the optional canonicality filter from genlm-control to verify that
        each adjacent pair of completed tokens is allowed under canonical BPE
        segmentation. Only considers completed tokens (i.e., those already
        committed to `lm_state.context`).
        """
        filt = getattr(self.params, "_canonical_filter", None)
        if filt is None:
            return False

        try:
            decode = s.trie.trie.decode
            bos_id = getattr(s.lm_state.model.tokenizer, "bos_token_id", None)
            # Map context token IDs to their byte strings; drop BOS if present
            ids = [t for t in s.lm_state.context if bos_id is None or t != bos_id]
            toks = [decode[t] for t in ids if t is not None]

            # Sequences with 0/1 tokens are trivially canonical
            if len(toks) <= 1:
                return True

            for i in range(1, len(toks)):
                left = toks[i - 1]
                right = toks[i]
                # If tokens are not in the encoding table, skip strict checking
                if left not in filt._encode or right not in filt._encode:
                    continue
                mask = filt((None, left))
                rid = filt._encode[right]
                if not bool(mask[rid]):
                    return False
            return True
        except Exception:
            # Be conservative: if we cannot check, do not force-include
            return False

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

        Args:
            bs (bytes): Byte sequence to prefill on

        Returns:
            (ByteBeamState): New beam state after prefilling
        """
        # Create no_eos beam for prefill (EOS tokens treated as normal)
        no_eos_beam = self.with_mode(TrieMode.WITHOUT_EOS)

        # Do prefill operations on no_eos beam
        for b in bs:
            no_eos_beam = await (no_eos_beam.prune() << b)

        # Return as with_eos beam (EOS tokens get special handling after prefill)
        return no_eos_beam.with_mode(TrieMode.WITH_EOS)

    async def cleanup(self):
        """Cleans up resources used by the candidates."""
        await asyncio.gather(*[state.cleanup() for state in self])

    async def _adaptive_heal(self, next_byte: int):
        """Attempt byte-preserving adaptive token healing across current candidates.

        Returns a new beam advanced by `next_byte` if healing succeeds, else None.
        """
        verbose = bool(getattr(self.params, "verbose", False))

        # Try each state in descending weight order
        for s in self.states:
            s = await s.materialize()

            trie = s.trie.trie
            children = trie.children

            P = s.partial  # bytes since last token boundary

            if verbose:  # pragma: no cover - diagnostics only
                try:
                    nb_disp = repr(bytes([next_byte])) if 0 <= next_byte <= 255 else str(next_byte)
                except Exception:
                    nb_disp = str(next_byte)
                print(
                    f"[heal] Start: next_byte={nb_disp}, P={repr(bytes(P))}, max_backoff={self.params.heal_max_backoff}"
                )

            # Base weight at the start of the current token
            base_weight = s.weight - (s.mass[s.node] - s.mass[trie.root])

            # Build path nodes along P from root
            path_nodes = [trie.root]
            node = trie.root
            ok = True
            for b in P:
                nxt = children[node].get(b)
                if nxt is None:
                    ok = False
                    break
                path_nodes.append(nxt)
                node = nxt
            if not ok:
                continue

            L = len(P)
            max_bk = self.params.heal_max_backoff
            min_k = max(0, L - (max_bk if max_bk is not None else L))

            # Structural precheck: find nearest k with EOT at P[:k], suffix P[k:] reachable from root,
            # and next_byte reachable from node(P[k:]). No LM calls here.
            chosen_k = None
            for k in range(L, min_k - 1, -1):
                anc_node = path_nodes[k]
                eot_node = children[anc_node].get(trie.eot_token)
                if eot_node is None:
                    if verbose:  # pragma: no cover - diagnostics only
                        print(f"[heal] k={k}: no EOT at prefix {repr(bytes(P[:k]))}")
                    continue

                # Try to follow suffix structurally from root
                node2 = trie.root
                suffix_ok = True
                for b in P[k:]:
                    nxt = children[node2].get(b)
                    if nxt is None:
                        suffix_ok = False
                        break
                    node2 = nxt
                if not suffix_ok:
                    if verbose:  # pragma: no cover - diagnostics only
                        print(f"[heal] k={k}: replay failed structurally (unreachable suffix)")
                    continue

                # Check next_byte structurally
                if children[node2].get(next_byte) is None:
                    if verbose:  # pragma: no cover - diagnostics only
                        print(f"[heal] k={k}: next_byte unreachable structurally; continue")
                    continue

                chosen_k = k
                break

            if chosen_k is None:
                # No structurally valid backoff for this state
                continue

            # Now do exactly one materialize for the chosen_k
            anc_node = path_nodes[chosen_k]
            eot_node = children[anc_node].get(trie.eot_token)
            token_id = int(trie.leaf2token_id[eot_node])
            w_after_eot = base_weight + (s.mass[eot_node] - s.mass[anc_node])

            committed = LazyTrieState(
                lm_state=(s.lm_state << token_id),
                trie=s.trie,
                node=trie.root,
                weight=w_after_eot,
                mass=None,
                mode=s.mode,
                terminated=False,
            )
            committed = await committed.materialize()

            if verbose:  # pragma: no cover - diagnostics only
                tok_bytes = trie.decode[token_id]
                print(
                    f"[heal] k={chosen_k}: commit token={repr(tok_bytes)}, base→w={w_after_eot:.2f}; replay suffix={repr(bytes(P[chosen_k:]))}"
                )

            # Replay suffix under new masses to get the exact weight
            node2 = trie.root
            weight2 = committed.weight
            for b in P[chosen_k:]:
                nxt = children[node2].get(b)
                weight2 = weight2 + (committed.mass[nxt] - committed.mass[node2])
                node2 = nxt

            # Advance by next_byte (without recursion)
            child = children[node2].get(next_byte)
            weight3 = weight2 + (committed.mass[child] - committed.mass[node2])
            healed_state = LazyTrieState(
                lm_state=committed.lm_state,
                trie=s.trie,
                node=child,
                weight=weight3,
                mass=committed.mass,
                mode=s.mode,
                terminated=(next_byte == 257),
            )
            if verbose:  # pragma: no cover - diagnostics only
                print(f"[heal] SUCCESS at k={chosen_k}: will consume {nb_disp} next; new_weight={weight3:.2f}")
            return ByteBeamState([healed_state], self.params)

        if verbose:  # pragma: no cover - diagnostics only
            print("[heal] FAILED: no valid backoff found")
        return None
