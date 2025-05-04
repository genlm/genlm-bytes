import asyncio
import numpy as np
from genlm.backend import AsyncTokenCharacterTrie
from genlm.backend.tokenization.bytes import get_byte_vocab

from .lm import ByteLM
from .trie_state import TrieState
from ..util import Chart, LRUCache


class ByteBeam(ByteLM):
    def __init__(self, llm, K, beam_cache_size=float("inf")):
        """
        Initialize a byte-level language model using the beam summing algorithm with a beam of size K.

        Args:
            llm (genlm.backend.AsyncLM): The token-level language model.
            K (int): The size of the beam.
            beam_cache_size (int): The size of the beam cache. If None, the cache is disabled.
                Defaults to float('inf').
        """
        self.llm = llm
        self.K = K
        token_V = get_byte_vocab(self.llm.tokenizer)
        self.async_trie = AsyncTokenCharacterTrie.from_vocab(token_V)

        lookup = {}
        for i, v in enumerate(token_V):
            if v in lookup:
                raise ValueError(
                    f"Token {v!r} maps to multiple token_ids ({lookup[v]}, {i})."
                )
            lookup[v] = i

        self.async_trie.lookup = lookup
        self.beam_cache = LRUCache(beam_cache_size)

        super().__init__(set(b"".join(token_V)))

    async def beam(self, qs):
        """
        Get the beam for the given context of bytes.

        Input:
            qs (bytes): Byte context.

        Returns:
            (list[Bundle]): A list of bundles of size at most K representing the beam.
        """
        if not qs:
            return [await TrieState.initial(self.llm, self.async_trie)]

        key = tuple(qs)

        try:
            return self.beam_cache[key]
        except KeyError:
            pass

        beam = await self.beam(qs[:-1])
        if not beam:
            return []

        beam = await self.extend_beam(beam, qs[-1])
        if not beam:
            return []

        self.beam_cache[key] = beam
        return beam

    async def extend_beam(self, beam, curr_byte):
        """
        Extend the beam by one byte.

        Args:
            beam (list[TrieState]): A list of TrieStates representing the beam.
            curr_byte (int): The byte to extend the beam by.

        Returns:
            (list[TrieState]): A list of TrieStates of size at most K representing the extended beam.
        """
        assert curr_byte in self.V, curr_byte

        candidates = []

        # Try to add curr_byte to all states.
        for state in beam:
            new_state = state << curr_byte
            if new_state:  # curr_byte can come next.
                candidates.append(new_state)

        # Try to extend states.
        extended = await asyncio.gather(*[s.extend() for s in beam])
        for state in extended:
            if state:  # EOT was available.
                new_state = state << curr_byte
                if new_state:  # curr_byte can come next.
                    new_state.parent = state
                    candidates.append(new_state)

        return sorted(candidates, key=lambda b: -b.weight)[: self.K]

    async def logp_next(self, qs):
        """
        Get the log probability of the next byte.

        Input:
            qs (bytes): Byte context.

        Returns:
            (Chart): A probability distribution over the next byte.
        """
        beam = await self.beam(qs)
        if not beam:
            return Chart(-np.inf)

        Q = Chart(-np.inf)
        for state in beam:
            logq = state.logp_next
            logppath = state.weight
            for k in logq:
                if k is not None:
                    Q[k] = np.logaddexp(Q[k], logppath + logq[k])

        extended = await asyncio.gather(*[s.extend() for s in beam])
        for state in extended:
            if state:  # EOT is available.
                logq = state.logp_next
                logppath = state.weight
                for k in logq:
                    if k is not None:
                        Q[k] = np.logaddexp(Q[k], logppath + logq[k])

        return Q

    async def cleanup(self):
        await self.async_trie.cleanup()
