import torch
import numpy as np
from functools import cached_property
from arsenal import colors
from .lm_state import StatefulTokenizedLM
from ..util import escape, LazyByteProbs


class LazyTrieState:
    """A lazy-evaluated state of a TokenByteTrie traversal.

    This class maintains the state of a language model while traversing a trie structure,
    lazily evaluating probabilities and maintaining the weight of the current path through the trie
    for beam search.

    Args:
        lm_state (StatefulTokenizedLM): Current language model state
        trie (TokenByteTrie): Trie structure mapping tokens to byte sequences
        node (int): Current node in the trie
        weight (float): Cumulative log probability of the path to this node
        mass (numpy.ndarray, optional): Masses for each node in the trie for the current state
    """

    def __init__(self, lm_state, trie, node, weight, mass=None, generation_mode=True):
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.weight = weight
        self._mass = mass
        self._extend = None
        self.generation_mode = generation_mode
        self.terminated = False
        self.root = self.trie.trie.root
        self.children = self.trie.trie.children

    @classmethod
    def initial(cls, lm, trie, generation_mode=True):
        """Creates an initial trie state.

        Args:
            lm (genlm.backend.AsyncLM): Language model to use
            trie (TokenByteTrie): TokenByteTrie structure for byte-to-token mapping
            generation_mode (bool): Whether this state is for generation or conditioning

        Returns:
            (LazyTrieState): Initial state at root of trie with weight 0.0
        """
        return cls(
            trie=trie,
            node=trie.trie.root,
            lm_state=StatefulTokenizedLM.initial(lm),
            weight=0.0,
            generation_mode=generation_mode,
        )

    @property
    def partial(self):
        """Returns the byte sequence corresponding to the current node in the trie."""
        return self.trie.trie.node2prefix[self.node]

    @property
    def mass(self):
        """Returns the log mass for each node in the trie.

        The mass at a node corresponds to the sum of the probabilities of all
        tokens which share the prefix (`self.partial`) represented by that node.

        Raises:
            ValueError: If state hasn't been materialized yet
        """
        if self._mass is None:
            raise ValueError("State is not yet materialized.")
        return self._mass

    def actions(self):
        """Returns possible byte transitions from current node."""
        return self.children[self.node]

    def get_EOT(self):
        """Returns the end-of-token node if available from current position in the trie."""
        return self.children[self.node].get(self.trie.trie.eot_token)

    def __lshift__(self, b):
        """Transitions to a new state by consuming a byte.

        Args:
            b (int): Byte to consume

        Returns:
            (LazyTrieState|None): New state after consuming byte, or None if transition invalid
        """
        if b == 257:  # EOS byte
            if self.generation_mode and getattr(self.trie.trie, 'eos_node', None) is not None:
                # Create terminated state
                mass = self.mass
                new_state = LazyTrieState(
                    lm_state=self.lm_state,
                    trie=self.trie,
                    node=self.node,  # Stay at current node
                    weight=self.weight + mass[self.trie.trie.eos_node] - mass[self.node],
                    mass=mass,
                    generation_mode=self.generation_mode
                )
                new_state.terminated = True
                return new_state
            else:
                # During conditioning or when EOS is disabled, EOS is not available
                return None
        
        # normal byte transition
        if node := self.children[self.node].get(b):
            mass = self.mass
            return LazyTrieState(
                lm_state=self.lm_state,
                trie=self.trie,
                mass=mass,
                node=node,
                weight=self.weight + mass[node] - mass[self.node],
                generation_mode=self.generation_mode,
            )

    def extend(self):
        """Extends current state by consuming an end-of-token if possible.

        Returns:
            (LazyTrieState|None): New state after consuming EOT, or None if not possible
        """
        if self._extend is None:
            if eot_node := self.get_EOT():
                mass = self.mass
                self._extend = LazyTrieState(
                    lm_state=self.lm_state
                    << int(self.trie.trie.leaf2token_id[eot_node]),
                    trie=self.trie,
                    node=self.root,
                    weight=self.weight + mass[eot_node] - mass[self.node],
                    generation_mode=self.generation_mode,
                )
        return self._extend

    @cached_property
    def logp_next(self):
        """Computes log probabilities for next possible transitions.

        Returns:
            (LazyByteProbs): Lazy log probability distribution over possible next bytes
        """
        if self.terminated:
            # Terminated states have no next transitions
            return LazyByteProbs(np.full(258, -np.inf), generation_mode=self.generation_mode)
        
        logps = np.full(258, -np.inf)  # 258 for bytes + EOT + EOS
        mass = self.mass
        logZ = mass[self.node]
        
        # Regular transitions
        for byte, node in self.actions().items():
            if byte is None:  # EOT
                logps[256] = mass[node] - logZ
            elif isinstance(byte, int) and 0 <= byte <= 255:
                logps[byte] = mass[node] - logZ
        
        # EOS transition (only during generation)
        if self.generation_mode and getattr(self.trie.trie, 'eos_node', None) is not None:
            logps[257] = mass[self.trie.trie.eos_node] - logZ
        
        return LazyByteProbs(logps, generation_mode=self.generation_mode)

    async def materialize(self):
        """Materializes the masses for each node in the trie for the current state.

        This makes a call to the language model and the underlying trie.

        Returns:
            (LazyTrieState): Self with materialized masses
        """
        if self._mass is None:
            logp_next = await self.lm_state.logp_next()
            # Use EOS-aware weight sum if available
            if hasattr(self.trie.trie, 'weight_sum_with_eos'):
                log_mass = self.trie.trie.weight_sum_with_eos(
                    torch.exp(logp_next), 
                    generation_mode=self.generation_mode
                )
                mass = torch.log(torch.tensor(log_mass, dtype=torch.float32))
            else:
                log_mass = await self.trie.weight_sum(torch.exp(logp_next))
                mass = torch.log(log_mass)
            self._mass = mass.cpu().numpy()
        return self

    def __repr__(self):
        context = colors.green % ("|" + escape(bytes(self.partial)))
        return f"{self.weight:.2f}: {self.lm_state}" + context

    async def cleanup(self):
        """Cleans up resources used by the trie."""
        await self.trie.cleanup()
