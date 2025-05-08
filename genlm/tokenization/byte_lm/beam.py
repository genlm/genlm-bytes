import asyncio
import numpy as np
from arsenal import colors
from dataclasses import dataclass
from functools import cached_property
from genlm.backend.tokenization.bytes import get_byte_vocab

from ..util import logsumexp, Chart
from ..trie import AsyncTokenByteTrie
from .trie_state import LazyTrieState
from .lm_state import StatefulByteLM


@dataclass
class BeamParams:
    K: int
    prune_threshold: float = 0.0
    verbose: bool = False


class ByteBeamState(StatefulByteLM):
    def __init__(self, states, params):
        self.states = sorted(states, key=lambda b: -b.weight)
        self.params = params

    @classmethod
    async def initial(cls, llm, params):
        state = LazyTrieState.initial(
            llm, AsyncTokenByteTrie.from_vocab(get_byte_vocab(llm.tokenizer))
        )
        return cls([await state.materialize()], params)

    def __iter__(self):
        return iter(self.states)

    def __len__(self):
        return len(self.states)

    @cached_property
    def logZ(self):
        return logsumexp([state.weight for state in self])

    async def __lshift__(self, a):
        new_states = []
        for state in self:
            if new_state := state << a:
                new_states.append(new_state)

        logZ = logsumexp([s.weight for s in new_states])
        for state in await self.extend(logZ):
            if new_state := state << a:
                new_states.append(new_state)

        new_state = ByteBeamState(new_states, self.params)

        if self.params.verbose:
            print()
            print(new_state)

        return new_state

    async def logp_next(self):
        Q = Chart(-np.inf)
        for state in self:
            for b, logp in state.logp_next.items():
                if b is not None:
                    Q[b] = np.logaddexp(Q[b], state.weight + logp)

        for state in await self.extend(self.logZ):
            for b, logp in state.logp_next.items():
                assert b is not None
                Q[b] = np.logaddexp(Q[b], state.weight + logp)

        Z = logsumexp(list(Q.values()))
        for b in Q:
            Q[b] -= Z

        return Q

    async def extend(self, logZ):
        extends = []
        for state in self:
            if new_state := state.extend:
                logZ = np.logaddexp(logZ, new_state.weight)
                extends.append(new_state)

        coros = []
        for state in extends:
            if np.exp(state.weight - logZ) > self.params.prune_threshold:
                coros.append(state.materialize())

        return await asyncio.gather(*coros)

    def prune(self):
        new_states = [
            state
            for state in self
            if np.exp(state.weight - self.logZ) > self.params.prune_threshold
        ][: self.params.K]
        return ByteBeamState(new_states, self.params)

    def __repr__(self):
        desc = colors.bold % f"Z: {self.logZ}\n" + colors.bold % "Candidates:\n"
        for state in self:
            P = np.exp(state.weight - self.logZ)
            color = colors.green if P > self.params.prune_threshold else colors.red
            desc += f"({color % f'{P:.4f}'}) {repr(state)}\n"
        return desc
