import asyncio
import numpy as np
from arsenal import colors
from dataclasses import dataclass
from functools import cached_property

from genlm.backend.tokenization.bytes import get_byte_vocab

from genlm.tokenization.trie import AsyncTokenByteTrie
from genlm.tokenization.util import Chart, logsumexp, unflatten

from .lm import StatefulByteLM
from .trie_state import TrieState


@dataclass
class BeamParams:
    K: int
    step_extend_threshold: float | None = None
    logp_extend_threshold: float | None = None


class ByteBeamState(StatefulByteLM):
    def __init__(self, states, V, params, context=()):
        if not states:
            raise ValueError("Beam state must contain at least one state.")

        self.states = sorted(states, key=lambda b: -b.weight)
        self.params = params

        super().__init__(V, context)

    @classmethod
    async def initial(cls, llm, params):
        """
        Initialize a beam state.

        Args:
            llm (genlm.backend.AsyncLM): The token-level language model.
            K (int): The (maximum) size of the beam.
            extend_threshold (float, optional): The threshold for extending a candidate.
        """
        decode = get_byte_vocab(llm.tokenizer)
        async_trie = AsyncTokenByteTrie.from_vocab(decode)
        states = [await TrieState.initial(llm, async_trie)]
        V = set(b"".join(decode))
        return cls(states, V, params)

    async def step(self, q, verbose=False):
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

        if verbose:
            print()
            for state in candidates:
                print(colors.bold % "Filtered:", repr(state))

        for state in await asyncio.gather(*extensions):
            if new_state := state << q:
                candidates.append(new_state)
                if verbose:
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

    def spawn(self, states, new_context=None):
        """Spawn a new beam state from the current one."""
        return ByteBeamState(
            states=states,
            V=self.V,
            params=self.params,
            context=new_context or self.context,
        )

    @cached_property
    async def logp_next(self):
        """
        Get the log probability distribution of the next byte.

        Returns:
            (Chart): The log probability distribution of the next byte.
        """
        extensions = []
        Q = Chart(-np.inf)
        Z = logsumexp([s.weight for s in self.states])
        for state in self.states:
            logq = state.logp_next
            logp = state.weight
            for k in logq:
                if k is not None:
                    Q[k] = np.logaddexp(Q[k], logp + logq[k])

            if state.has_EOT():
                if (
                    self.params.logp_extend_threshold is None
                    or np.exp(state.weight + logq[None] - Z)
                    > self.params.logp_extend_threshold
                ):
                    extensions.append(state.extend())

        # Handle Nones that were filtered above.
        extended = await asyncio.gather(*extensions)
        for state in extended:
            logq = state.logp_next
            logp = state.weight
            for k in logq:
                assert k is not None
                Q[k] = np.logaddexp(Q[k], logp + logq[k])

        for k in Q:
            Q[k] = Q[k] - Z

        _Z = logsumexp(list(Q.values()))
        assert np.isclose(_Z, 0, atol=1e-5), _Z

        return Q

    async def cleanup(self):
        """Async clean up method."""
        await asyncio.gather(*[s.cleanup() for s in self.states])

    def __repr__(self):
        return (
            colors.bold % "Current context: "
            + colors.green % repr(bytes(unflatten(self.context)))
            + "\n"
            + colors.bold % "Candidates:"
            + "\n"
            + (
                "\n".join(
                    f"\033[38;5;247m{repr(state)}\033[0m"
                    if i >= self.K
                    else repr(state)
                    for i, state in enumerate(self.states)
                )
            )
        )
