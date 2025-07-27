import torch
import numpy as np
from enum import Enum
from functools import cached_property
from arsenal import colors
from .lm_state import StatefulTokenizedLM
from ..util import escape, LazyByteProbs

# EOS byte constant - using 257 as the virtual EOS byte
EOS = 257


class TrieMode(Enum):
    """Modes for trie state behavior."""
    NO_EOS = "no_eos"              # EOS tokens behave like normal tokens  
    PROPAGATE_EOS = "propagate_eos" # EOS tokens get special handling (aggregated to EOS node)
    TERMINATED = "terminated"       # EOS consumed, sequence ended


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

    def __init__(self, lm_state, trie, node, weight, mass=None, mode=TrieMode.PROPAGATE_EOS):
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.weight = weight
        self._mass = mass
        self._extend = None
        self.mode = mode
        self.root = self.trie.trie.root
        self.children = self.trie.trie.children



    @classmethod
    def initial(cls, lm, trie, mode=TrieMode.PROPAGATE_EOS):
        """Creates an initial trie state.

        Args:
            lm (genlm.backend.AsyncLM): Language model to use
            trie (TokenByteTrie): TokenByteTrie structure for byte-to-token mapping
            mode (TrieMode): Trie mode to use

        Returns:
            (LazyTrieState): Initial state at root of trie with weight 0.0
        """
        return cls(
            trie=trie,
            node=trie.trie.root,
            lm_state=StatefulTokenizedLM.initial(lm),
            weight=0.0,
            mode=mode,
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
        # Terminated states cannot consume any more bytes
        if self.mode == TrieMode.TERMINATED:
            return None

        # EOS transition with weight update
        if b == EOS and self.mode == TrieMode.PROPAGATE_EOS:
            # EOS is only available from root node
            if self.node != self.root or self.trie.trie.eos_node is None:
                return None
            
            # Proper EOS transition with weight update
            mass = self.mass
            return LazyTrieState(
                lm_state=self.lm_state,
                trie=self.trie,
                mass=mass,
                node=self.node,  # Stay at root
                weight=self.weight + mass[self.trie.trie.eos_node] - mass[self.node],
                mode=TrieMode.TERMINATED,
            )

        # Handle normal byte transitions
        if node := self.children[self.node].get(b):
            mass = self.mass
            return LazyTrieState(
                lm_state=self.lm_state,
                trie=self.trie,
                mass=mass,
                node=node,
                weight=self.weight + mass[node] - mass[self.node],
                mode=self.mode,
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
                    mode=self.mode,
                )
        return self._extend

    @cached_property
    def logp_next(self):
        """Computes log probabilities for next possible transitions.

        Returns:
            (LazyByteProbs): Lazy log probability distribution over possible next bytes
        """
        logps = np.full(258, -np.inf)  # 258 for EOT, EOS + 256 for normal bytes
        mass = self.mass
        logZ = mass[self.node]

        for byte, node in self.actions().items():
            logps[byte if byte is not None else 256] = mass[node] - logZ

        # Add EOS probability if in propagate_eos mode and EOS node exists
        # EOS is only available from root (where the EOS node is connected)
        if (
            self.mode == TrieMode.PROPAGATE_EOS
            and hasattr(self.trie.trie, "eos_node")
            and self.trie.trie.eos_node is not None
            and self.node == self.root
        ):
            logps[EOS] = mass[self.trie.trie.eos_node] - logZ

        return LazyByteProbs(logps)

    async def materialize(self):
        """Materializes the masses for each node in the trie for the current state.

        This makes a call to the language model and the underlying trie.

        Returns:
            (LazyTrieState): Self with materialized masses
        """
        if self._mass is None:
            logp_next = await self.lm_state.logp_next()
            log_mass = await self.trie.weight_sum_with_eos(
                torch.exp(logp_next), self.mode
            )
            mass = torch.log(log_mass)
            self._mass = mass.cpu().numpy()
        return self

    def __repr__(self):
        context = colors.green % ("|" + escape(bytes(self.partial)))
        return f"{self.weight:.2f}: {self.lm_state}" + context

    async def cleanup(self):
        """Cleans up resources used by the trie."""
        await self.trie.cleanup()
