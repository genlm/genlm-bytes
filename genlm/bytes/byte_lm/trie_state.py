import torch
import numpy as np
from functools import cached_property
from arsenal import colors
from .lm_state import StatefulTokenizedLM
from ..util import escape, LazyByteProbs


class LazyTrieState:
    def __init__(self, lm_state, trie, node, weight, mass=None):
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.weight = weight
        self._mass = mass
        self._extend = None
        self.root = self.trie.trie.root
        self.children = self.trie.trie.children

    @classmethod
    def initial(cls, lm, trie):
        return cls(
            trie=trie,
            node=trie.trie.root,
            lm_state=StatefulTokenizedLM.initial(lm),
            weight=0.0,
        )

    @property
    def partial(self):
        return self.trie.trie.node2prefix[self.node]

    @property
    def mass(self):
        if self._mass is None:
            raise ValueError("State is not yet materialized.")
        return self._mass

    def actions(self):
        return self.children[self.node]

    def get_EOT(self):
        return self.children[self.node].get(self.trie.trie.eot_token)

    def __lshift__(self, b):
        if node := self.children[self.node].get(b):
            mass = self.mass
            return LazyTrieState(
                lm_state=self.lm_state,
                trie=self.trie,
                mass=mass,
                node=node,
                weight=self.weight + mass[node] - mass[self.node],
            )

    def extend(self):
        if self._extend is None:
            if eot_node := self.get_EOT():
                mass = self.mass
                self._extend = LazyTrieState(
                    lm_state=self.lm_state
                    << int(self.trie.trie.leaf2token_id[eot_node]),
                    trie=self.trie,
                    node=self.root,
                    weight=self.weight + mass[eot_node] - mass[self.node],
                )
        return self._extend

    @cached_property
    def logp_next(self):
        logps = np.full(257, -np.inf)  # 257 for EOT
        mass = self.mass
        logZ = mass[self.node]
        for byte, node in self.actions().items():
            logps[byte if byte is not None else 256] = mass[node] - logZ
        return LazyByteProbs(logps)

    async def materialize(self):
        if self._mass is None:
            logp_next = await self.lm_state.logp_next
            mass = torch.log(await self.trie.weight_sum(torch.exp(logp_next)))
            self._mass = mass.detach().cpu().numpy()
        return self

    def __repr__(self):
        context = colors.green % ("|" + escape(bytes(self.partial)))
        return f"{self.weight:.2f}: {self.lm_state}" + context

    async def cleanup(self):
        await self.trie.cleanup()
