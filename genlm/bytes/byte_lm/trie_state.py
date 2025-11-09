import torch
import numpy as np
from functools import cached_property
from arsenal import colors
from .lm_state import StatefulTokenizedLM
from ..util import escape, LazyByteProbs
from ..trie import TrieMode

# EOS byte constant - using 257 as the virtual EOS byte
EOS = 257


def _torch_dtype_from(mass_dtype):
    if isinstance(mass_dtype, torch.dtype):
        return mass_dtype
    if isinstance(mass_dtype, str):
        dtype = getattr(torch, mass_dtype, None)
        if dtype is None:
            raise ValueError(f"Unknown torch dtype string: {mass_dtype}")
        return dtype
    if mass_dtype is None:
        return torch.float32
    raise ValueError(f"Unsupported mass dtype specification: {mass_dtype}")


def _numpy_dtype(torch_dtype):
    mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.bfloat16: np.float16,  # use float16 for storage when bf16 requested
    }
    if hasattr(torch, "float8_e4m3fn"):
        mapping[torch.float8_e4m3fn] = np.float16
    if hasattr(torch, "float8_e5m2"):
        mapping[torch.float8_e5m2] = np.float16
    return mapping.get(torch_dtype, np.float32)


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
        mass (numpy.ndarray|torch.Tensor, optional): Masses for each node in the trie for the current state
        mode (TrieMode): Trie mode to use
        terminated (bool): Whether the state is terminated (EOS has been consumed)
        mass_dtype (torch.dtype|str|None): Desired dtype for stored masses (default float32)
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
        mass_dtype=torch.float32,
        store_mass_cpu=False,
    ):
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.weight = weight
        self._mass = None
        self._mass_cpu = None
        self._extend = None
        self.mode = mode
        self.root = self.trie.trie.root
        self.children = self.trie.trie.children
        self.terminated = terminated
        self.mass_dtype = _torch_dtype_from(mass_dtype)
        self.mass_dtype_np = _numpy_dtype(self.mass_dtype)
        self.store_mass_cpu = store_mass_cpu

        if mass is not None:
            self._assign_mass(mass)

    @classmethod
    def initial(
        cls,
        lm,
        trie,
        mode=TrieMode.WITH_EOS,
        mass_dtype=torch.float32,
        store_mass_cpu=False,
    ):
        """Creates an initial trie state."""
        return cls(
            trie=trie,
            node=trie.trie.root,
            lm_state=StatefulTokenizedLM.initial(lm),
            weight=0.0,
            mode=mode,
            mass_dtype=mass_dtype,
            store_mass_cpu=store_mass_cpu,
        )

    @property
    def partial(self):
        """Returns the byte sequence corresponding to the current node in the trie."""
        return self.trie.trie.node2prefix[self.node]

    @property
    def mass(self):
        """Returns the log mass for each node in the trie."""
        if self._mass is None:
            raise ValueError("State is not yet materialized.")
        return self._mass

    def with_mode(self, mode):
        """Returns a new state with the given mode."""
        new_state = LazyTrieState(
            lm_state=self.lm_state,
            trie=self.trie,
            node=self.node,
            weight=self.weight,
            mass=self._mass,
            mode=mode,
            terminated=self.terminated,
            mass_dtype=self.mass_dtype,
            store_mass_cpu=self.store_mass_cpu,
        )
        if self._mass_cpu is not None:
            new_state._mass_cpu = self._mass_cpu
        return new_state

    def actions(self):
        """Returns possible byte transitions from current node."""
        return self.children[self.node]

    def get_EOT(self):
        """Returns the end-of-token node if available from current position in the trie."""
        return self.children[self.node].get(self.trie.trie.eot_token)

    def __lshift__(self, b):
        if self.terminated:
            return None

        if node := self.children[self.node].get(b):
            delta = self._mass_scalar(node) - self._mass_scalar(self.node)
            new_state = LazyTrieState(
                lm_state=self.lm_state,
                trie=self.trie,
                mass=self._mass,
                node=node,
                weight=self.weight + delta,
                mode=self.mode,
                terminated=b == EOS,
                mass_dtype=self.mass_dtype,
                store_mass_cpu=self.store_mass_cpu,
            )
            if self._mass_cpu is not None:
                new_state._mass_cpu = self._mass_cpu
            return new_state

    def extend(self):
        if self._extend is None:
            if eot_node := self.get_EOT():
                delta = self._mass_scalar(eot_node) - self._mass_scalar(self.node)
                new_state = LazyTrieState(
                    lm_state=self.lm_state
                    << int(self.trie.trie.leaf2token_id[eot_node]),
                    trie=self.trie,
                    node=self.root,
                    weight=self.weight + delta,
                    mode=self.mode,
                    mass_dtype=self.mass_dtype,
                    store_mass_cpu=self.store_mass_cpu,
                )
                if self._mass_cpu is not None:
                    new_state._mass_cpu = self._mass_cpu
                self._extend = new_state
        return self._extend

    @cached_property
    def logp_next(self):
        logps = np.full(258, -np.inf)
        logZ = self._mass_scalar(self.node)

        for byte, node in self.actions().items():
            logps[byte if byte is not None else 256] = self._mass_scalar(node) - logZ

        return LazyByteProbs(logps)

    async def materialize(self):
        if self._mass is None:
            logp_next = await self.lm_state.logp_next()
            log_mass = await self.trie.weight_sum(torch.exp(logp_next), self.mode)
            self._assign_mass(torch.log(log_mass))
        return self

    def __repr__(self):
        context = colors.green % ("|" + escape(bytes(self.partial)))
        return f"{self.weight:.2f}: {self.lm_state}" + context

    async def cleanup(self):
        await self.trie.cleanup()

    def _assign_mass(self, mass):
        if torch.is_tensor(mass):
            mass = mass.to(dtype=self.mass_dtype)
            self._mass = mass
            if self.store_mass_cpu:
                cpu_tensor = mass.detach().cpu().to(torch.float16)
                self._mass_cpu = cpu_tensor.numpy().astype(
                    self.mass_dtype_np, copy=False
                )
            else:
                self._mass_cpu = None
        elif mass is None:
            self._mass = None
            self._mass_cpu = None
        else:
            np_mass = np.asarray(mass, dtype=self.mass_dtype_np)
            self._mass = torch.from_numpy(np_mass)
            self._mass_cpu = np_mass if self.store_mass_cpu else None

    def _mass_scalar(self, node):
        if self._mass_cpu is not None:
            return float(self._mass_cpu[node])
        value = self.mass[node]
        return value.item() if hasattr(value, "item") else float(value)
