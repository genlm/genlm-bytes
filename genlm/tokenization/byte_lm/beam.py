import asyncio
import numpy as np
from arsenal import colors
from functools import cached_property

from genlm.backend import load_model_by_name
from genlm.backend.tokenization.bytes import get_byte_vocab

from genlm.tokenization.byte_lm.trie_state import TrieState
from genlm.tokenization.util import Chart, load_async_trie, logsumexp, unflatten

from .lm import StatefulByteLM


class ByteBeamState(StatefulByteLM):
    def __init__(self, states, K, V, extend_threshold=None, context=()):
        if not states:
            raise ValueError("Beam state must contain at least one state.")

        self.states = sorted(states, key=lambda b: -b.weight)
        self.K = K
        self.extend_threshold = extend_threshold

        super().__init__(V, context)

    @classmethod
    async def initial(cls, llm, K, extend_threshold=None):
        """
        Initialize a beam state.

        Args:
            llm (genlm.backend.AsyncLM): The token-level language model.
            K (int): The (maximum) size of the beam.
            extend_threshold (float, optional): The threshold for extending a candidate.
        """
        decode = get_byte_vocab(llm.tokenizer)
        async_trie = load_async_trie(decode)
        states = [await TrieState.initial(llm, async_trie)]
        V = set(b"".join(decode))
        return cls(states, K, V, extend_threshold)

    @classmethod
    async def initial_from_name(cls, name, K, extend_threshold=None, **kwargs):
        """
        Initialize a beam state from a token-level language model name.

        Args:
            name (str): The name of the model to load.
            K (int): The (maximum) size of the beam.
            extend_threshold (float, optional): The threshold for extending a candidate.
            **kwargs: Additional arguments to pass to `load_model_by_name`.
        """
        llm = load_model_by_name(name, **kwargs)
        return await cls.initial(llm, K, extend_threshold)

    async def step(self, q: int, verbose=False):
        """
        Extend the beam state by one byte.

        Example:
            >>> state = await ByteBeamState.initial(llm, K=5)
            >>> state = await state.step(97) # "a"
            >>> state = await state.step(98) # "b"
            >>> state.context # The bytes we have consumed so far.
            ((97,), 98)
            >>> await state.logp_next # The log probability distribution of the next byte.
            Chart(...)

        Args:
            q (int): The next byte to extend the beam by.

        Returns:
            (ByteBeamState): The new beam state.
        """
        candidates = []
        extensions = []

        if verbose:
            print()

        for state in self.states:
            # Filter for curr_byte.
            if new_state := state << q:
                candidates.append(new_state)
                if verbose:
                    print(colors.bold % "Filtered:", repr(new_state))
            # Maybe extend.
            if self.maybe_extend(state, q):
                extensions.append(state.extend())

        # Extend concurrently.
        for state in await asyncio.gather(*extensions):
            if new_state := state << q:
                new_state.parent = state
                candidates.append(new_state)
                if verbose:
                    print(colors.bold % "Ext+Filt:", repr(new_state))

        return self.spawn(candidates, new_context=(self.context, q))

    def prune(self):
        """Prune the beam to the top K states."""
        return self.spawn(sorted(self.states, key=lambda b: -b.weight)[: self.K])

    def maybe_extend(self, state, q):
        """Determine whether to extend the state with an End-of-Token (EOT)."""

        if not state.has_EOT():
            return False

        if q not in state.children[state.root]:
            return False

        if self.extend_threshold is None:
            return True

        children = state.children[state.node]
        if q not in children:
            return True

        return (
            np.exp(state.mass[children[q]] - state.mass[children[None]])
            < self.extend_threshold
        )

    def spawn(self, states, new_context=None):
        """Spawn a new beam state from the current one."""
        return ByteBeamState(
            states=states,
            K=self.K,
            V=self.V,
            extend_threshold=self.extend_threshold,
            context=new_context or self.context,
        )

    @cached_property
    async def logp_next(self):
        """
        Get the log probability distribution of the next byte.

        Returns:
            (Chart): The log probability distribution of the next byte.
        """
        Q = Chart(-np.inf)
        extensions = []
        for state in self.states:
            logq = state.logp_next
            logp = state.weight
            for k in logq:
                if k is not None:
                    Q[k] = np.logaddexp(Q[k], logp + logq[k])

            if state.has_EOT():
                extensions.append(state.extend())

        # Handle Nones that were filtered above.
        extended = await asyncio.gather(*extensions)
        for state in extended:
            logq = state.logp_next
            logp = state.weight
            for k in logq:
                assert k is not None
                Q[k] = np.logaddexp(Q[k], logp + logq[k])

        Z = logsumexp(
            [s.weight for s in self.states]
        )  # Z = logsumexp(list(Q.values())) # This is equivalent.
        for k in Q:
            Q[k] = Q[k] - Z

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
