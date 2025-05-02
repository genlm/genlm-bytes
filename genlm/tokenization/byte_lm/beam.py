import asyncio
import numpy as np
from genlm.backend import AsyncTokenCharacterTrie
from genlm.backend.tokenization.bytes import get_byte_vocab

from .lm import ByteLM
from .bundle import Bundle
from genlm.tokenization.util import Chart, logsumexp


class ByteBeam(ByteLM):
    def __init__(self, llm, K):
        """
        Initialize a byte-level language model using the beam summing algorithm with a beam of size K.

        Args:
            llm (genlm.backend.AsyncLM): The token-level language model.
            K (int): The size of the beam.
        """
        self.llm = llm
        self.K = K
        token_V = get_byte_vocab(self.llm.tokenizer)
        self.async_trie = AsyncTokenCharacterTrie.from_vocab(token_V)
        self.async_trie.trie.lookup = {
            v: i for i, v in enumerate(token_V)
        }  # TODO: expose this in backend.
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
            return [await Bundle.create(self.llm, self.async_trie)]

        beam = await self.beam(qs[:-1])
        if not beam:
            return []

        return await self.extend(beam, qs[-1])

    async def extend(self, beam, curr_byte):
        """
        Extend the beam by one byte.

        Args:
            beam (list[Bundle]): A list of bundles representing the beam.
            curr_byte (int): The byte to extend the beam by.

        Returns:
            (list[Bundle]): A list of bundles of size at most K representing the extended beam.
        """
        assert curr_byte in self.V, curr_byte

        candidates = []
        for b in beam:
            new_b = b.filter(curr_byte)
            if new_b:  # curr_byte can come next.
                candidates.append(new_b)

        bs = await asyncio.gather(*[b.extend() for b in beam])
        for b in bs:
            if b:  # At a leaf.
                new_b = b.filter(curr_byte)
                if new_b:  # curr_byte can come next.
                    candidates.append(new_b)

        return sorted(candidates, key=lambda b: -b.weight)[: self.K]

    async def p_next(self, context):
        """
        Get the probability of the next byte.

        Input:
            context (bytes): Byte context.

        Returns:
            (Chart): A probability distribution over the next byte.
        """
        beam = await self.beam(context)
        if not beam:
            return Chart(0.0)

        Q = Chart(0.0)
        Z = logsumexp([b.weight for b in beam])

        for b in beam:
            q = b.p_next()
            ppath = np.exp(b.weight - Z)
            for k in q:
                if k is not None:  # k is a byte.
                    Q[k] += ppath * q[k]

        bs = await asyncio.gather(*[b.extend() for b in beam])
        for b in bs:
            if b:  # At a leaf.
                q = b.p_next()
                ppath = np.exp(b.weight - Z)
                for k in q:
                    if k is not None:  # k is a byte.
                        Q[k] += ppath * q[k]

        return Q
