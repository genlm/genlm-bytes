import torch
import numpy as np
from functools import cached_property

from .lm import StateLM
from ..util import Chart


class TrieState:
    def __init__(self, lm_state, trie, node, mass, weight, parent, _extend=None):
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.mass = mass
        self.weight = weight
        self.parent = parent
        self.root = self.trie.trie.root
        self._extend = _extend
        self.children = self.trie.trie.children

    @classmethod
    async def initial(cls, lm, trie):
        lm_state = StateLM.initial(lm)
        mass = np.log(await trie.weight_sum(torch.exp(await lm_state.logp_next())))
        return cls(
            trie=trie,
            node=trie.trie.root,
            lm_state=lm_state,
            mass=mass,
            weight=0,
            parent=None,
        )

    def __lshift__(self, curr_byte):
        """
        Add a new byte to the TrieState.

        Example:
        ```python
            >>> state = await TrieState.initial(lm, trie)
            >>> state = state << 104 # "h"
            >>> state = state << 105 # "i"
            >>> bytes(state.partial)
            b"hi"
        ```

        Args:
            curr_byte (int): The byte to add to the TrieState.

        Returns:
            (TrieState|None): A new TrieState with the added byte, or None if the byte cannot come next.
        """
        if curr_byte not in self.children[self.node]:
            return
        return TrieState(
            lm_state=self.lm_state,
            trie=self.trie,
            node=self.children[self.node][curr_byte],
            mass=self.mass,
            weight=self.weight,
            parent=self,
        )

    async def extend(self):
        """
        Extend the bundle by one token if it is at a leaf (EOT).

        Returns:
            (Bundle|None): A new bundle with the extended key, mass, and weight, or None if the bundle is not at a leaf.
        """
        if self._extend:
            return self._extend

        eot_node = self.children[self.node].get(None)
        if not eot_node:
            return

        next_tok = self.trie.trie.leaf2word[eot_node]
        lm_state = self.lm_state << self.trie.lookup[next_tok]
        new_mass = np.log(
            await self.trie.weight_sum(torch.exp(await lm_state.logp_next()))
        )

        # We sometimes call extend in logp_next. This
        # helps us avoid recomputing the same state twice.
        self._extend = TrieState(
            lm_state=lm_state,
            trie=self.trie,
            node=self.root,
            mass=new_mass,
            weight=self.weight + self.mass[eot_node] - self.mass[self.node],
            parent=self,
        )

        return self._extend

    def actions(self):
        return self.children[self.node]

    @cached_property
    def logp_next(self):
        logZ = self.mass[self.node]
        return Chart(
            -np.inf, {a: self.mass[i] - logZ for a, i in self.actions().items()}
        )

    @property
    def partial(self):
        return self.trie.trie.node2prefix[self.node]

    def __repr__(self):
        return f"TrieState(partial={self.partial}, weight={self.weight}, lm_state={self.lm_state})"

    def clone(self):
        return TrieState(
            lm_state=self.lm_state.clone(),
            trie=self.trie,
            node=self.node,
            mass=self.mass,
            weight=self.weight,
            parent=self.parent,
            _extend=self._extend.clone() if self._extend else None,
        )

    async def cleanup(self):
        self.trie.cleanup()

    # TODO: add viz methods
