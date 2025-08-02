import asyncio
import numpy as np
from arsenal import colors
from dataclasses import dataclass
from scipy.special import logsumexp as scipy_logsumexp
from functools import cached_property
from genlm.backend.tokenization.bytes import get_byte_vocab

from ..util import logsumexp, LazyByteProbs
from ..trie import AsyncTokenByteTrie
from .trie_state import LazyTrieState, TrieMode
from .lm_state import StatefulByteLM


@dataclass
class BeamParams:
    """Parameters for byte-level beam summing algorithm.

    Args:
        K (int): Beam width - maximum number of candidates to maintain.
        prune_threshold (float, optional): Probability threshold for pruning candidates.
            Candidates with probability below this are removed. Defaults to 0.0
        verbose (bool, optional): Whether to print the beam state at each step. Defaults to False
        eos_tokens (list[bytes], optional): List of tokens that should be treated as EOS. When configured,
            EOS tokens will terminate generation when sampled. Defaults to None
    """

    K: int
    prune_threshold: float = 0.0
    verbose: bool = False
    eos_tokens: list[bytes] = None

    def __post_init__(self):
        if self.prune_threshold < 0:
            raise ValueError(
                f"prune_threshold must be non-negative, got {self.prune_threshold}"
            )
        self.log_prune_threshold = (
            np.log(self.prune_threshold) if self.prune_threshold > 0 else -np.inf
        )
        self.eos_tokens = set(self.eos_tokens) if self.eos_tokens else {}


class ByteBeamState(StatefulByteLM):
    """Represents the state of the beam during byte-level language modeling.

    Tracks multiple candidate states and their probabilities, pruning low-probability
    candidates.

    Args:
        states (list[LazyTrieState]): List of candidate states to track
        params (BeamParams): Parameters controlling beam search behavior
    """

    def __init__(self, states, params):
        self.states = sorted(states, key=lambda b: -b.weight)
        self.params = params

    @classmethod
    async def initial(cls, llm, params, trie_opts=None):
        """Creates initial beam state.

        Args:
            llm (StatefulTokenizedLM): Token-level language model to use.
            params (BeamParams): Beam search parameters.
            trie_opts (dict, optional): Additional keyword arguments passed to
                AsyncTokenByteTrie.from_vocab. For example, {"max_batch_size": 100}.

        Returns:
            (ByteBeamState): Initial beam state.
        """
        # Handle EOS tokens
        trie_opts = trie_opts or {}
        trie_opts["eos_tokens"] = params.eos_tokens

        async_trie = AsyncTokenByteTrie.from_vocab(
            get_byte_vocab(llm.tokenizer), **trie_opts
        )
        state = LazyTrieState.initial(llm, async_trie, mode=TrieMode.WITH_EOS)
        return cls([await state.materialize()], params)

    def __iter__(self):
        return iter(self.states)

    def __len__(self):
        return len(self.states)

    @cached_property
    def logZ(self):
        """Estimate of the partition function (sum of weights) for current beam.
        This is the estimate of the prefix probability of the bytes consumed so far.
        """
        return logsumexp([state.weight for state in self])

    async def __lshift__(self, a):
        """Advances the beam state with a new byte.

        Args:
            a (int): Byte to add to states.

        Returns:
            (ByteBeamState): New beam state after processing the byte.
        """
        new_states = []
        for state in self:
            if new_state := state << a:
                new_states.append(new_state)

        logZ = logsumexp([s.weight for s in new_states]) if new_states else -np.inf
        for state in await self.extend(logZ):
            if new_state := state << a:
                new_states.append(new_state)

        new_state = ByteBeamState(new_states, self.params)

        if self.params.verbose:
            print()
            print(new_state)

        return new_state

    async def logp_next(self):
        """Computes log probabilities for the next byte across all beam candidates.

        Returns:
            (LazyByteProbs): Log probabilities for next possible bytes.
        """
        assert len(self) > 0, "Beam is empty"

        logqs = []
        for state in self:
            logqs.append(state.logp_next.ps + state.weight)

        for state in await self.extend(self.logZ):
            logqs.append(state.logp_next.ps + state.weight)

        logqs = np.stack(logqs, axis=0)  # shape: (num_states, array_size)
        # mask EOT positions of non-extended (EOT is at index 256)
        logqs[: len(self), -2] = -np.inf
        logps = scipy_logsumexp(logqs, axis=0)

        return LazyByteProbs(logps - logsumexp(logps))

    async def extend(self, logZ):
        """Attempts to advance each candidate in the beam by a token (EOT).

        For each candididate with EOT available, this ends the current token and
        starts a new one in preparation for the next byte.

        Args:
            logZ (float): Current estimated of the partition function for pruning

        Returns:
            (list[LazyTrieState]): New candidate states after extension
        """
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
        """Prunes beam to maintain beam width and probability threshold.

        Returns:
            (ByteBeamState): New state with pruned candidates.
        """
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

    def with_mode(self, mode):
        """Create a new beam state with specified trie mode.

        Args:
            mode (TrieMode): Trie mode for the new beam state

        Returns:
            (ByteBeamState): New beam state with updated mode
        """
        return ByteBeamState(
            states=[state.with_mode(mode) for state in self.states],
            params=self.params,
        )

    async def prefill(self, bs):
        """Prefill the beam on a sequence of bytes.

        During prefilling, EOS tokens are treated as normal tokens and don't cause termination.

        Args:
            bs (bytes): Byte sequence to prefill on

        Returns:
            (ByteBeamState): New beam state after prefilling
        """
        # Create no_eos beam for prefill (EOS tokens treated as normal)
        no_eos_beam = self.with_mode(TrieMode.WITHOUT_EOS)

        # Do prefill operations on no_eos beam
        for b in bs:
            no_eos_beam = await (no_eos_beam.prune() << b)

        # Return as with_eos beam (EOS tokens get special handling after prefill)
        return no_eos_beam.with_mode(TrieMode.WITH_EOS)

    async def cleanup(self):
        """Cleans up resources used by the candidates."""
        await asyncio.gather(*[state.cleanup() for state in self])
