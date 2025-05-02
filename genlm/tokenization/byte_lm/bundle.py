import torch
import numpy as np

from genlm.tokenization.util import Chart


class Bundle:
    def __init__(self, llm, key, wpath, mass, node, async_trie):
        self.llm = llm
        self.key = key
        self.wpath = wpath
        self.node = node
        self.mass = mass
        self.weight = wpath + np.log(mass[node])
        self.async_trie = async_trie
        self.trie = async_trie.trie
        self._extend = None

    @classmethod
    async def create(cls, llm, async_trie):
        """
        Create a new bundle for the initial state.

        Returns:
            (Bundle): A new bundle for the initial state.
        """
        start_ctx = [llm.tokenizer.bos_token_id]
        logprobs = await llm.next_token_logprobs(start_ctx)
        mass = await async_trie.weight_sum(torch.exp(logprobs))
        return cls(llm, start_ctx, 0, mass, async_trie.trie.root, async_trie)

    def filter(self, curr_byte):
        """
        Filter the bundle by a new byte.

        Args:
            curr_byte (int): The byte to filter the bundle by.

        Returns:
            (Bundle|None): A new bundle with the filtered byte, or None if the byte cannot come next.
        """
        if curr_byte not in self.trie.children[self.node]:
            return
        return Bundle(
            llm=self.llm,
            key=self.key,
            mass=self.mass,
            node=self.trie.children[self.node][curr_byte],
            async_trie=self.async_trie,
            wpath=self.wpath,
        )

    async def extend(self):
        """
        Extend the bundle by one token if it is at a leaf (EOT).

        Returns:
            (Bundle|None): A new bundle with the extended key, mass, and weight, or None if the bundle is not at a leaf.
        """
        if self._extend:
            return self._extend

        node = self.trie.children[self.node].get(None)
        if not node:
            return

        context = self.key + [self.trie.lookup[self.trie.leaf2word[node]]]  # Slow?
        logprobs = await self.llm.next_token_logprobs(context)
        new_mass = await self.async_trie.weight_sum(torch.exp(logprobs))

        self._extend = Bundle(
            llm=self.llm,
            key=context,
            node=self.trie.root,
            mass=new_mass,
            async_trie=self.async_trie,
            wpath=self.wpath + np.log(self.mass[node]),
        )

        return self._extend

    def p_next(self):
        """
        Get the probability of the next byte.

        Returns:
            (Chart): A chart of the probability of the next byte.
        """
        return Chart(
            0, {byte: self.mass[i] for byte, i in self.trie.children[self.node].items()}
        )

    def items(self):
        """
        Generate the items of the bundle. Used to inspect the bundle.

        Returns:
            (Generator): A generator of the items of the bundle.
        """
        agenda = [self.node]
        while agenda:
            i = agenda.pop()
            for symbol, j in self.trie.children[i].items():
                if symbol is None:
                    yield (j, self.mass[j])
                else:
                    agenda.append(j)
