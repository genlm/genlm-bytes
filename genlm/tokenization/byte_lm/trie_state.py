import torch
import numpy as np
from functools import cached_property
from arsenal import colors
from .lm_state import StatefulTokenizedLM
from ..util import Chart, escape


class LazyTrieState:
    def __init__(self, lm_state, trie, node, weight, mass=None):
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.weight = weight

        self._mass = mass
        self._extend = None
        self._logp_next = None

        self.root = self.trie.trie.root
        self.children = self.trie.trie.children

    @classmethod
    def initial(cls, lm, trie):
        return cls(
            trie=trie,
            node=trie.trie.root,
            lm_state=StatefulTokenizedLM.initial(lm),
            weight=0,
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

    def __lshift__(self, b):
        if node := self.children[self.node].get(b):
            return LazyTrieState(
                lm_state=self.lm_state,
                trie=self.trie,
                mass=self.mass,
                node=node,
                weight=self.weight + self.mass[node] - self.mass[self.node],
            )

    @cached_property
    def extend(self):
        if eot_node := self.children[self.node].get(None):
            return LazyTrieState(
                lm_state=self.lm_state << int(self.trie.trie.leaf2token_id[eot_node]),
                trie=self.trie,
                node=self.root,
                weight=self.weight + self.mass[eot_node] - self.mass[self.node],
            )

    @cached_property
    def logp_next(self):
        if self._logp_next is None:
            logZ = self.mass[self.node]
            self._logp_next = Chart(
                -np.inf, {a: self.mass[i] - logZ for a, i in self.actions().items()}
            )
        return self._logp_next

    async def materialize(self):
        if self._mass is None:
            self._mass = (
                torch.log(
                    await self.trie.weight_sum(torch.exp(await self.lm_state.logp_next))
                )
                .cpu()
                .numpy()
            )
        return LazyTrieState(
            lm_state=self.lm_state,
            trie=self.trie,
            node=self.node,
            mass=self._mass,
            weight=self.weight,
        )

    def __repr__(self):
        context = colors.green % ("|" + escape(bytes(self.partial)))
        return f"{self.weight:.2f}: {self.lm_state}" + context

    async def cleanup(self):
        self.trie.cleanup()
