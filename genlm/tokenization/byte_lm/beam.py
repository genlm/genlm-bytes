import asyncio
import numpy as np
from arsenal import colors
from dataclasses import dataclass
from functools import cached_property
from genlm.backend.tokenization.bytes import get_byte_vocab

from genlm.tokenization.trie import AsyncTokenByteTrie
from genlm.tokenization.util import Chart, logsumexp, flatten, logsubexp

from .lm_state import StatefulByteLM
from .trie_state import TrieState


@dataclass
class BeamParams:
    K: int
    step_extend_threshold: float | None = None
    logp_extend_threshold: float | None = None
    verbose: bool = False


class ByteBeamState(StatefulByteLM):
    def __init__(self, states, params, context=()):
        if not states:
            raise ValueError("Beam state must contain at least one state.")

        self.states = sorted(states, key=lambda b: -b.weight)
        self.params = params

        super().__init__(context)

    @classmethod
    async def initial(cls, llm, params):
        """
        Initialize a beam state.

        Args:
            llm (genlm.backend.AsyncLM): The token-level language model.
            K (int): The (maximum) size of the beam.
            extend_threshold (float, optional): The threshold for extending a candidate.
        """
        initial_state = await TrieState.initial(
            llm, AsyncTokenByteTrie.from_vocab(get_byte_vocab(llm.tokenizer))
        )
        return cls([initial_state], params)

    async def step(self, q):
        """
        Extend the beam state by one byte.

        Args:
            q (int): The next byte to extend the beam by.

        Returns:
            (ByteBeamState): The new beam state.
        """
        candidates = []
        extensions = []

        for state in self.states:
            if new_state := state << q:
                candidates.append(new_state)
            if self.maybe_extend(state, q):
                extensions.append(state.extend())

        if self.params.verbose:
            print()
            for state in candidates:
                print(colors.bold % "Filtered:", repr(state))

        for state in await asyncio.gather(*extensions):
            if new_state := state << q:
                candidates.append(new_state)
                if self.params.verbose:
                    print(colors.bold % "Ext+Filt:", repr(new_state))

        return self.spawn(candidates, new_context=(self.context, q))

    def prune(self):
        """Prune the beam to the top K states."""
        return self.spawn(self.states[: self.params.K])

    def maybe_extend(self, state, q):
        """Determine whether to extend the state with an End-of-Token (EOT)."""

        if not state.has_EOT():
            return False

        if q not in state.children[state.root]:
            return False

        if self.params.step_extend_threshold is None:
            return True

        children = state.children[state.node]
        if q not in children:
            return True

        return (
            np.exp(state.mass[children[q]] - state.mass[children[None]])
            < self.params.step_extend_threshold
        )

    @cached_property
    async def logp_next(self):
        """
        Get the log probability distribution of the next byte.

        Returns:
            (Chart): The log probability distribution of the next byte.
        """
        if self.params.verbose:
            print(colors.bold % f"logp_next {colors.green % repr(self.flat_context)}:")

        Q = Chart(-np.inf)  # Log distribution over next bytes
        W = logsumexp([s.weight for s in self.states])  # Total weight
        Z = -np.inf  # Adjusted total weight (accounting for pruned extensions)
        extensions = []

        # First pass: Handle immediate next-byte probabilities
        for state in self.states:
            logq = state.logp_next
            logp = state.weight
            for k in logq:
                if k is not None:
                    Q[k] = np.logaddexp(Q[k], logp + logq[k])

            if state.has_EOT() and (
                self.params.logp_extend_threshold is None
                or
                # Don't extend if the extended state's contribution to the total weight is less than the threshold
                np.exp(state.weight + logq[None] - W)
                >= self.params.logp_extend_threshold
            ):
                Z = np.logaddexp(Z, state.weight)
                extensions.append(state.extend())
            else:
                # Remove the contribution of the extended state from the total weight
                Z = np.logaddexp(Z, logsubexp(state.weight, state.weight + logq[None]))

        # Second pass: Handle extended states
        extended = await asyncio.gather(*extensions)
        for state in extended:
            logq = state.logp_next
            logp = state.weight
            for k in logq:
                assert k is not None
                Q[k] = np.logaddexp(Q[k], logp + logq[k])

        Q_norm = Chart(-np.inf)
        for k in Q:
            Q_norm[k] = Q[k] - Z

        if self.params.verbose:
            self._print_logp_debug_info(W, Z, Q_norm, Q)

        return Q_norm

    def spawn(self, states, new_context=None):
        """Spawn a new beam state from the current one."""
        return ByteBeamState(
            states=states,
            params=self.params,
            context=new_context or self.context,
        )

    @property
    def flat_context(self):
        return bytes(flatten(self.context))

    def _print_logp_debug_info(self, W, Z, Q_norm, Q):
        """Helper method to print debug information."""
        print(
            "\n".join(
                (colors.green % f"{repr(bytes([k])):<3} ({k:3})") + f" {v:.8f}"
                for k, v in Q_norm.map_values(np.exp).top(10).items()
            )
            + "\n...\n"
        )
        print(colors.bold % f"log W: {W}")
        print(colors.bold % f"log Z: {Z}")
        print(colors.bold % f"log Σ: {logsumexp(list(Q.values()))}")
        print(
            colors.blue
            % f"pruned {np.exp(logsubexp(W, Z) - W) * 100:.2f}% of total weight"
        )

    def __repr__(self):
        return (
            colors.bold % "Current context: "
            + colors.green % repr(self.flat_context)
            + "\n"
            + colors.bold % "Candidates:"
            + "\n"
            + (
                "\n".join(
                    f"\033[38;5;247m{repr(state)}\033[0m"
                    if i >= self.params.K
                    else repr(state)
                    for i, state in enumerate(self.states)
                )
            )
        )

    async def cleanup(self):
        """Async clean up method."""
        await asyncio.gather(*[s.cleanup() for s in self.states])
