import torch
import numpy as np
from functools import cached_property
from arsenal import colors
from .lm_state import StatefulTokenizedLM
from ..util import escape, LazyByteProbs
from ..trie import TrieMode

# EOS byte constant - using 257 as the virtual EOS byte
EOS = 257


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
        mode (TrieMode): Trie mode to use
        terminated (bool): Whether the state is terminated (EOS has been consumed)
    """

    def __init__(
        self,
        lm_state,
        trie,
        node,
        weight,
        mass=None,
        mode=TrieMode.WITH_EOS,
        terminated=False,
    ):
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.weight = weight
        self._mass = mass
        self._extend = None
        self.mode = mode
        self.root = self.trie.trie.root
        self.children = self.trie.trie.children
        self.terminated = terminated

    @classmethod
    def initial(cls, lm, trie, mode=TrieMode.WITH_EOS):
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

    def with_mode(self, mode):
        """Returns a new state with the given mode."""
        return LazyTrieState(
            lm_state=self.lm_state,
            trie=self.trie,
            node=self.node,
            weight=self.weight,
            mass=self._mass,
            mode=mode,
            terminated=self.terminated,
        )

    def actions(self):
        """Returns possible byte transitions from current node."""
        return self.children[self.node]

    def get_all_EOT(self):
        """Returns all EOT edges from the current position in the trie.
        
        With duplicate tokens, multiple token IDs can map to the same byte string,
        resulting in multiple EOT edges at the same node.

        Returns:
            list[tuple[int, int]]: List of (eot_node, token_id) tuples for each EOT edge.
                Empty list if no EOT edges exist.
        """
        eot_sentinel = self.trie.trie.eot_sentinel
        results = []
        for key, node in self.children[self.node].items():
            # EOT edges are tuples: (eot_sentinel, token_id)
            if isinstance(key, tuple) and key[0] == eot_sentinel:
                token_id = key[1]
                results.append((node, token_id))
        return results

    def __lshift__(self, b):
        """Transitions to a new state by consuming a byte.

        Args:
            b (int): Byte to consume

        Returns:
            (LazyTrieState|None): New state after consuming byte, or None if transition invalid (terminated or EOS)
        """
        if self.terminated:
            return None

        if node := self.children[self.node].get(b):
            mass = self.mass
            return LazyTrieState(
                lm_state=self.lm_state,
                trie=self.trie,
                mass=mass,
                node=node,
                weight=self.weight + mass[node] - mass[self.node],
                mode=self.mode,
                terminated=b == EOS,
            )

    def extend_all(self):
        """Extends current state by consuming an end-of-token, returning all possible extensions.

        With duplicate tokens (multiple token IDs with the same byte string), there can be
        multiple valid extensions at the same position. Each extension corresponds to a
        different token being committed, which affects future LM predictions.

        Returns:
            list[LazyTrieState]: List of new states after consuming EOT, one per possible token.
                Empty list if no EOT edges exist.
        """
        if self._extend is None:
            extensions = []
            mass = self.mass
            for eot_node, token_id in self.get_all_EOT():
                extensions.append(LazyTrieState(
                    lm_state=self.lm_state << int(token_id),
                    trie=self.trie,
                    node=self.root,
                    weight=self.weight + mass[eot_node] - mass[self.node],
                    mode=self.mode,
                ))
            self._extend = extensions
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

        for key, node in self.actions().items():
            # Handle different edge types:
            # - tuple: (eot_sentinel, token_id) for EOT edges to leaves
            # - int 0-255: byte transitions
            # - int 257: EOS transition
            if isinstance(key, tuple):
                # EOT edge - use index 256
                # For duplicates, sum their masses using logaddexp
                if logps[256] == -np.inf:
                    logps[256] = mass[node] - logZ
                else:
                    logps[256] = np.logaddexp(logps[256], mass[node] - logZ)
            elif isinstance(key, int):
                logps[key] = mass[node] - logZ
            else:  # pragma: no cover
                raise ValueError(
                    f"Unexpected edge key type: {type(key).__name__} (value: {key!r}). "
                    f"Expected tuple (EOT edge) or int (byte/EOS transition)."
                )

        return LazyByteProbs(logps)

    async def materialize(self):
        """Materializes the masses for each node in the trie for the current state.

        This makes a call to the language model and the underlying trie.

        Returns:
            (LazyTrieState): Self with materialized masses
        """
        if self._mass is None:
            logp_next = await self.lm_state.logp_next()
            log_mass = await self.trie.weight_sum(torch.exp(logp_next), self.mode)
            mass = torch.log(log_mass)
            self._mass = mass.cpu().numpy()
        return self

    def __repr__(self):  # pragma: no cover
        context = colors.green % ("|" + escape(bytes(self.partial)))
        if self.terminated:
            context += colors.yellow % "<EOS>"
        return f"{self.weight:.2f}: {self.lm_state}" + context

    async def cleanup(self):
        """Cleans up resources used by the trie."""
        await self.trie.cleanup()
