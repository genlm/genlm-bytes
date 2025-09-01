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
        self.eos_tokens = set(self.eos_tokens) if self.eos_tokens else {}


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
                    print("[heal] Applied adaptive token healing inside __lshift__")
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

            if verbose:
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

            # Commit planning handled by ByteBeamState._plan_commits

            chosen_k = None
            commit_plan: list[int] | None = None

            # Unified planner: for each valid k (EOT at P[:k]), compute a plan over S=P[k:].
            # Accept empty plan (single-split) or multi-split plan subject to heal_max_splits.
            for k in range(L, min_k - 1, -1):
                anc_node = path_nodes[k]
                eot_node = children[anc_node].get(trie.eot_token)
                if eot_node is None:
                    if verbose:
                        print(f"[heal] k={k}: no EOT at prefix {repr(bytes(P[:k]))}")
                    continue
                plan = self._plan_commits(trie, bytes(P[k:]), next_byte, self.params.heal_max_splits)
                if plan is None:
                    if verbose:
                        print(f"[heal] k={k}: no plan found (unreachable suffix or next_byte)")
                    continue
                chosen_k = k
                commit_plan = plan
                break

            if chosen_k is None:
                # No structurally valid backoff for this state
                continue

            # Apply the chosen plan and return healed state if successful
            healed = await self._apply_commit_plan(
                s=s,
                trie=trie,
                children=children,
                P=P,
                chosen_k=chosen_k,
                plan_positions=commit_plan or [],
                next_byte=next_byte,
                base_weight=base_weight,
                path_nodes=path_nodes,
                verbose=verbose,
            )
            if healed is not None:
                return healed

        if verbose:
            print("[heal] FAILED: no valid backoff found")
        return None

    def _plan_commits(
        self,
        trie,
        S: bytes,
        next_byte: int,
        heal_max_splits: int | None,
    ) -> list[int] | None:
        """Plan commit positions inside suffix S so that next_byte is structurally reachable.

        Returns strictly increasing absolute positions within S at which to commit EOT,
        or None if impossible under the structural trie.
        """
        children = trie.children
        commits: list[int] = []
        if len(S) == 0:
            return commits if children[trie.root].get(next_byte) is not None else None

        seg_start = 0
        node2 = trie.root
        last_eot_in_seg: int | None = None  # relative to seg_start

        # Phase 1: make S structurally reachable by greedy longest-match with commits at last EOT when stuck.
        i = seg_start
        while i < len(S):
            nxt = children[node2].get(S[i])
            if nxt is not None:
                node2 = nxt
                i += 1
                if children[node2].get(trie.eot_token) is not None:
                    last_eot_in_seg = i - seg_start
            else:
                if last_eot_in_seg is None:
                    return None
                if heal_max_splits is not None and len(commits) >= heal_max_splits:
                    return None
                # Commit at the last EOT within this segment and restart next segment
                abs_pos = seg_start + last_eot_in_seg
                commits.append(abs_pos)
                seg_start = abs_pos
                node2 = trie.root
                last_eot_in_seg = None
                i = seg_start

        # At end of S, node2 is the node after consuming the tail segment.
        # Phase 2: ensure next_byte is reachable. If not, iteratively commit the last EOT of the current tail segment.
        while children[node2].get(next_byte) is None:
            # Find last EOT in the current tail segment [seg_start:len(S))
            n = trie.root
            last_e = None
            for j in range(seg_start, len(S)):
                nxt = children[n].get(S[j])
                if nxt is None:
                    break
                n = nxt
                if children[n].get(trie.eot_token) is not None:
                    last_e = j + 1  # absolute position
            if last_e is None:
                return None
            if heal_max_splits is not None and len(commits) >= heal_max_splits:
                return None
            commits.append(last_e)
            seg_start = last_e
            # Replay the new tail segment head to get node2
            n = trie.root
            for j in range(seg_start, len(S)):
                nxt = children[n].get(S[j])
                if nxt is None:
                    break
                n = nxt
            node2 = n

        return commits

    async def _apply_commit_plan(
        self,
        s: LazyTrieState,
        trie,
        children,
        P: list[int],
        chosen_k: int,
        plan_positions: list[int],
        next_byte: int,
        base_weight: float,
        path_nodes: list[int],
        verbose: bool,
    ):
        """Commit P[:chosen_k], then planned intra-suffix commits, then consume next_byte.

        Returns a new ByteBeamState if successful, else None (if the plan proves invalid).
        """
        # Initial commit of P[:k]
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
                f"[heal] k={chosen_k}: commit token={repr(tok_bytes)}, base→w={w_after_eot:.2f}; plan commits at {plan_positions}; suffix={repr(bytes(P[chosen_k:]))}"
            )

        # Replay suffix with optional multi-split commits under new masses to get the exact weight
        node2 = trie.root
        weight2 = committed.weight

        S = list(P[chosen_k:])
        last = 0
        # Helper to follow bytes [i:j) under current committed.mass and update (node2, weight2)
        def _follow_span(i: int, j: int):
            nonlocal node2, weight2
            for bb in S[i:j]:
                nxt = children[node2].get(bb)
                weight2 = weight2 + (committed.mass[nxt] - committed.mass[node2])
                node2 = nxt

        for cp in plan_positions:
            _follow_span(last, cp)
            # Commit EOT at node2
            eot_n = children[node2].get(trie.eot_token)
            if eot_n is None:  # Defensive: plan should only include valid EOT points
                if verbose:  # pragma: no cover - diagnostics only
                    print(f"[heal] plan invalid at {cp}: no EOT at node; abort healing for this state")
                return None
            tok_id = int(trie.leaf2token_id[eot_n])
            weight2 = weight2 + (committed.mass[eot_n] - committed.mass[node2])
            # Advance LM and materialize new masses
            committed = LazyTrieState(
                lm_state=(committed.lm_state << tok_id),
                trie=s.trie,
                node=trie.root,
                weight=weight2,
                mass=None,
                mode=s.mode,
                terminated=False,
            )
            committed = await committed.materialize()
            if verbose:  # pragma: no cover - diagnostics only
                print(f"[heal] commit@{cp}: token={repr(trie.decode[tok_id])}, new_w={weight2:.2f}")
            # Reset traversal under updated masses
            node2 = trie.root
            last = cp

        # Follow remaining residual
        _follow_span(last, len(S))
        # Advance by next_byte under current committed.mass
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
            try:
                nb_disp = repr(bytes([next_byte])) if 0 <= next_byte <= 255 else str(next_byte)
            except Exception:
                nb_disp = str(next_byte)
            print(f"[heal] SUCCESS at k={chosen_k}: will consume {nb_disp} next; new_weight={weight3:.2f}")
        return ByteBeamState([healed_state], self.params)
