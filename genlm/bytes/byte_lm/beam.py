import asyncio
import numpy as np
from arsenal import colors
from dataclasses import dataclass
from scipy.special import logsumexp as scipy_logsumexp
from functools import cached_property
from genlm.backend.tokenization.bytes import get_byte_vocab

from ..util import logsumexp, LazyByteProbs
from ..trie import AsyncTokenByteTrie
from .trie_state import LazyTrieState, TrieMode, EOS
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

    def _format_byte(self, byte_val: int) -> str:
        """Format byte value for verbose logging."""
        try:
            return repr(bytes([byte_val])) if 0 <= byte_val <= 255 else str(byte_val)
        except Exception:
            return str(byte_val)

    def _traverse_bytes(self, children: dict, eot_token: int, start_node: int, bytes_seq, track_eot: bool = False):
        """Traverse trie following bytes_seq from start_node.
        
        Returns:
            tuple: (final_node, last_eot_pos) where last_eot_pos is None if not tracking EOT
                   or if no EOT found. Returns (None, None) if traversal fails.
        """
        node = start_node
        last_eot_pos = None
        
        for i, byte_val in enumerate(bytes_seq):
            next_node = children[node].get(byte_val)
            if next_node is None:
                return (None, None)
            node = next_node
            if track_eot and children[node].get(eot_token) is not None:
                last_eot_pos = i + 1
        
        return (node, last_eot_pos)

    def _build_path_nodes(self, children: dict, root: int, partial_bytes: list[int]) -> list[int] | None:
        """Build path of nodes along partial_bytes from root.
        
        Returns list of nodes or None if path is invalid.
        """
        path_nodes = [root]
        current_node = root
        
        for byte_val in partial_bytes:
            next_node = children[current_node].get(byte_val)
            if next_node is None:
                return None
            path_nodes.append(next_node)
            current_node = next_node
        
        return path_nodes

    async def _adaptive_heal(self, next_byte: int):
        """Attempt byte-preserving adaptive token healing across current candidates.

        Returns a new beam advanced by `next_byte` if healing succeeds, else None.
        """
        verbose = self.params.verbose

        # Try each state in descending weight order
        for state in self.states:
            state = await state.materialize()
            trie = state.trie.trie
            children = trie.children
            partial_bytes = state.partial

            if verbose:
                byte_disp = self._format_byte(next_byte)
                print(
                    f"[heal] Start: next_byte={byte_disp}, P={repr(bytes(partial_bytes))}, "
                    f"max_backoff={self.params.heal_max_backoff}"
                )

            # Base weight at the start of the current token
            base_weight = state.weight - (state.mass[state.node] - state.mass[trie.root])

            # Build path nodes along partial_bytes from root
            path_nodes = self._build_path_nodes(children, trie.root, partial_bytes)
            if path_nodes is None:
                continue

            # Find valid backoff point and plan commits
            partial_len = len(partial_bytes)
            max_backoff = self.params.heal_max_backoff
            min_k = max(0, partial_len - (max_backoff if max_backoff is not None else partial_len))

            chosen_k, commit_plan = self._find_heal_plan(
                trie, children, path_nodes, partial_bytes, next_byte, min_k, verbose
            )
            if chosen_k is None:
                continue

            # Apply the chosen plan and return healed state if successful
            healed = await self._apply_commit_plan(
                state=state,
                trie=trie,
                children=children,
                partial_bytes=partial_bytes,
                chosen_k=chosen_k,
                plan_positions=commit_plan,
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

    def _find_heal_plan(
        self, trie, children, path_nodes, partial_bytes, next_byte, min_k, verbose
    ) -> tuple[int | None, list[int] | None]:
        """Find a valid healing plan by trying backoff positions.
        
        Returns:
            tuple: (chosen_k, commit_plan) or (None, None) if no valid plan found.
        """
        partial_len = len(partial_bytes)
        
        for k in range(partial_len, min_k - 1, -1):
            ancestor_node = path_nodes[k]
            eot_node = children[ancestor_node].get(trie.eot_token)
            if eot_node is None:
                if verbose:
                    print(f"[heal] k={k}: no EOT at prefix {repr(bytes(partial_bytes[:k]))}")
                continue
            
            suffix = bytes(partial_bytes[k:])
            plan = self._plan_commits(trie, suffix, next_byte, self.params.heal_max_splits)
            if plan is None:
                if verbose:
                    print(f"[heal] k={k}: no plan found (unreachable suffix or next_byte)")
                continue
            
            return (k, plan)
        
        return (None, None)

    def _plan_commits(
        self,
        trie,
        suffix_bytes: bytes,
        next_byte: int,
        heal_max_splits: int | None,
    ) -> list[int] | None:
        """Plan commit positions inside suffix_bytes so that next_byte is structurally reachable.

        Returns strictly increasing absolute positions within suffix_bytes at which to commit EOT,
        or None if impossible under the structural trie.
        """
        children = trie.children
        commits: list[int] = []
        
        # Empty suffix case: check if next_byte is reachable from root
        if len(suffix_bytes) == 0:
            return commits if children[trie.root].get(next_byte) is not None else None

        seg_start = 0
        current_node = trie.root
        last_eot_in_seg: int | None = None  # relative to seg_start

        # Phase 1: make suffix_bytes structurally reachable by greedy longest-match with commits at last EOT when stuck.
        i = seg_start
        while i < len(suffix_bytes):
            next_node = children[current_node].get(suffix_bytes[i])
            if next_node is not None:
                current_node = next_node
                i += 1
                if children[current_node].get(trie.eot_token) is not None:
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
                current_node = trie.root
                last_eot_in_seg = None
                i = seg_start

        # Phase 2: ensure next_byte is reachable. If not, iteratively commit the last EOT of the current tail segment.
        while children[current_node].get(next_byte) is None:
            # Find last EOT in the current tail segment [seg_start:len(suffix_bytes))
            tail_segment = suffix_bytes[seg_start:]
            final_node, last_eot_pos = self._traverse_bytes(
                children, trie.eot_token, trie.root, tail_segment, track_eot=True
            )
            
            if final_node is None or last_eot_pos is None:
                return None
            
            if heal_max_splits is not None and len(commits) >= heal_max_splits:
                return None
            
            # Commit at the absolute position
            abs_eot_pos = seg_start + last_eot_pos
            commits.append(abs_eot_pos)
            seg_start = abs_eot_pos
            
            # Replay the new tail segment to get current_node
            new_tail = suffix_bytes[seg_start:]
            current_node, _ = self._traverse_bytes(
                children, trie.eot_token, trie.root, new_tail, track_eot=False
            )
            if current_node is None:
                return None

        return commits

    async def _apply_commit_plan(
        self,
        state: LazyTrieState,
        trie,
        children,
        partial_bytes: list[int],
        chosen_k: int,
        plan_positions: list[int],
        next_byte: int,
        base_weight: float,
        path_nodes: list[int],
        verbose: bool,
    ):
        """Commit partial_bytes[:chosen_k], then planned intra-suffix commits, then consume next_byte.

        Returns a new ByteBeamState if successful, else None (if the plan proves invalid).
        """
        # Initial commit of partial_bytes[:chosen_k]
        ancestor_node = path_nodes[chosen_k]
        eot_node = children[ancestor_node].get(trie.eot_token)
        token_id = int(trie.leaf2token_id[eot_node])
        weight_after_eot = base_weight + (state.mass[eot_node] - state.mass[ancestor_node])

        committed_state = LazyTrieState(
            lm_state=(state.lm_state << token_id),
            trie=state.trie,
            node=trie.root,
            weight=weight_after_eot,
            mass=None,
            mode=state.mode,
            terminated=False,
        )
        committed_state = await committed_state.materialize()

        if verbose:
            token_bytes = trie.decode[token_id]
            suffix_str = repr(bytes(partial_bytes[chosen_k:]))
            print(
                f"[heal] k={chosen_k}: commit token={repr(token_bytes)}, base→w={weight_after_eot:.2f}; "
                f"plan commits at {plan_positions}; suffix={suffix_str}"
            )

        # Replay suffix with optional multi-split commits under new masses to get the exact weight
        current_node = trie.root
        current_weight = committed_state.weight
        suffix_list = list(partial_bytes[chosen_k:])
        last_pos = 0

        # Helper to follow bytes [start:end) under current committed_state.mass and update (current_node, current_weight)
        def _follow_span(start: int, end: int):
            nonlocal current_node, current_weight
            for byte_val in suffix_list[start:end]:
                next_node = children[current_node].get(byte_val)
                current_weight = current_weight + (
                    committed_state.mass[next_node] - committed_state.mass[current_node]
                )
                current_node = next_node

        for commit_pos in plan_positions:
            _follow_span(last_pos, commit_pos)
            # Commit EOT at current_node
            eot_node_at_pos = children[current_node].get(trie.eot_token)
            if eot_node_at_pos is None:  # Defensive: plan should only include valid EOT points
                if verbose:
                    print(
                        f"[heal] plan invalid at {commit_pos}: no EOT at node; abort healing for this state"
                    )
                return None
            token_id_at_pos = int(trie.leaf2token_id[eot_node_at_pos])
            current_weight = current_weight + (
                committed_state.mass[eot_node_at_pos] - committed_state.mass[current_node]
            )
            # Advance LM and materialize new masses
            committed_state = LazyTrieState(
                lm_state=(committed_state.lm_state << token_id_at_pos),
                trie=state.trie,
                node=trie.root,
                weight=current_weight,
                mass=None,
                mode=state.mode,
                terminated=False,
            )
            committed_state = await committed_state.materialize()
            if verbose:
                print(
                    f"[heal] commit@{commit_pos}: token={repr(trie.decode[token_id_at_pos])}, "
                    f"new_w={current_weight:.2f}"
                )
            # Reset traversal under updated masses
            current_node = trie.root
            last_pos = commit_pos

        # Follow remaining residual
        _follow_span(last_pos, len(suffix_list))
        # Advance by next_byte under current committed_state.mass
        child_node = children[current_node].get(next_byte)
        final_weight = current_weight + (
            committed_state.mass[child_node] - committed_state.mass[current_node]
        )
        healed_state = LazyTrieState(
            lm_state=committed_state.lm_state,
            trie=state.trie,
            node=child_node,
            weight=final_weight,
            mass=committed_state.mass,
            mode=state.mode,
            terminated=(next_byte == EOS),
        )
        if verbose:
            byte_disp = self._format_byte(next_byte)
            print(
                f"[heal] SUCCESS at k={chosen_k}: will consume {byte_disp} next; "
                f"new_weight={final_weight:.2f}"
            )
        return ByteBeamState([healed_state], self.params)
