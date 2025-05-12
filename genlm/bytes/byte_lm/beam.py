import asyncio
import numpy as np
from arsenal import colors
from dataclasses import dataclass
from scipy.special import logsumexp as scipy_logsumexp
from functools import cached_property
from genlm.backend.tokenization.bytes import get_byte_vocab

from ..util import logsumexp, LazyByteProbs
from ..trie import AsyncTokenByteTrie
from .trie_state import LazyTrieState
from .lm_state import StatefulByteLM


@dataclass
class BeamParams:
    K: int
    prune_threshold: float = 0.0
    verbose: bool = False

    def __post_init__(self):
        if self.prune_threshold < 0:
            raise ValueError(
                f"prune_threshold must be non-negative, got {self.prune_threshold}"
            )
        self.log_prune_threshold = (
            np.log(self.prune_threshold) if self.prune_threshold > 0 else -np.inf
        )


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
        logqs = []
        for state in self:
            logqs.append(state.logp_next.ps + state.weight)

        for state in await self.extend(self.logZ):
            logqs.append(state.logp_next.ps + state.weight)

        logqs = np.stack(logqs, axis=0)  # shape: (num_states, 257)
        logqs[: len(self), -1] = -np.inf  # mask EOT positions of non-extended
        logps = scipy_logsumexp(logqs, axis=0)

        return LazyByteProbs(logps - logsumexp(logps))

    async def extend(self, logZ):
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

    async def cleanup(self):
        await asyncio.gather(*[state.cleanup() for state in self])
