import asyncio
import numpy as np
from functools import cached_property

from genlm.backend.tokenization.bytes import get_byte_vocab

from genlm.tokenization.byte_lm.trie_state import TrieState
from genlm.tokenization.util import Chart, load_async_trie

from .stream import ByteStreamLM


class BeamByteStream(ByteStreamLM):
    def __init__(self, states, K, V, context=()):
        self.states = states
        self.K = K
        super().__init__(V, context)

    @classmethod
    async def initial(cls, llm, K):
        """
        Initialize a beam.

        Args:
            llm (genlm.backend.AsyncLM): The token-level language model.
            K (int): The (maximum) size of the beam.
        """
        decode = get_byte_vocab(llm.tokenizer)
        async_trie = load_async_trie(decode)
        states = [await TrieState.initial(llm, async_trie)]
        V = set(b"".join(decode))
        return cls(states, K, V)

    async def step(self, q):
        candidates = []

        # Try to add curr_byte to all states.
        for state in self.states:
            new_state = state << q
            if new_state:  # curr_byte can come next.
                candidates.append(new_state)

        # Try to extend states.
        extended = await asyncio.gather(*[s.extend() for s in self.states])
        for state in extended:
            if state:  # EOT was available.
                new_state = state << q
                if new_state:  # curr_byte can come next.
                    new_state.parent = state
                    candidates.append(new_state)

        top_K = sorted(candidates, key=lambda b: -b.weight)[: self.K]
        return BeamByteStream(
            states=top_K, K=self.K, V=self.V, context=(self.context, q)
        )

    @cached_property
    async def logp_next(self):
        Q = Chart(-np.inf)
        for state in self.states:
            logq = state.logp_next
            logppath = state.weight
            for k in logq:
                if k is not None:
                    Q[k] = np.logaddexp(Q[k], logppath + logq[k])

        extended = await asyncio.gather(*[s.extend() for s in self.states])
        for state in extended:
            if state:  # EOT is available.
                logq = state.logp_next
                logppath = state.weight
                for k in logq:
                    if k is not None:
                        Q[k] = np.logaddexp(Q[k], logppath + logq[k])

        return Q

    async def cleanup(self) -> None:
        await asyncio.gather(*[s.cleanup() for s in self.states])
