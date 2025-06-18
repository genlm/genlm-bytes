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
    """Parameters for byte-level beam summing algorithm.

    Args:
        K (int): Beam width - maximum number of candidates to maintain.
        prune_threshold (float, optional): Probability threshold for pruning candidates.
            Candidates with probability below this are removed. Defaults to 0.0
        verbose (bool, optional): Whether to print the beam state at each step. Defaults to False
        eos_tokens (set, optional): Set of tokens that should be treated as EOS. Defaults to None
        terminate_on_eos (bool, optional): Whether to terminate generation on EOS. Defaults to True
    """

    K: int
    prune_threshold: float = 0.0
    verbose: bool = False
    eos_tokens: set = None
    terminate_on_eos: bool = True

    def __post_init__(self):
        if self.prune_threshold < 0:
            raise ValueError(
                f"prune_threshold must be non-negative, got {self.prune_threshold}"
            )
        self.log_prune_threshold = (
            np.log(self.prune_threshold) if self.prune_threshold > 0 else -np.inf
        )
        if self.eos_tokens is None:
            self.eos_tokens = set()


class ByteBeamState(StatefulByteLM):
    """Represents the state of the beam during byte-level language modeling.

    Tracks multiple candidate states and their probabilities, pruning low-probability
    candidates.

    Args:
        states (list[LazyTrieState]): List of candidate states to track
        params (BeamParams): Parameters controlling beam search behavior
    """

    def __init__(self, states, params, generation_mode=True):
        # Separate active and terminated states
        self.states = sorted([s for s in states if not getattr(s, 'terminated', False)], 
                           key=lambda b: -b.weight)
        self.terminated_states = [s for s in states if getattr(s, 'terminated', False)]
        self.params = params
        self.generation_mode = generation_mode

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
        # Pass EOS tokens to trie if specified
        trie_options = trie_opts or {}
        if params.eos_tokens:
            trie_options['eos_tokens'] = params.eos_tokens
        
        state = LazyTrieState.initial(
            llm,
            AsyncTokenByteTrie.from_vocab(
                get_byte_vocab(llm.tokenizer), **trie_options
            ),
            generation_mode=True,
        )
        return cls([await state.materialize()], params, generation_mode=True)

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
        if a == 257 and self.generation_mode and self.params.terminate_on_eos:
            # EOS byte - create terminated beam
            terminated_states = []
            for state in self.states:
                if new_state := state << a:
                    terminated_states.append(new_state)
            
            # Return beam with only terminated states
            new_beam = ByteBeamState(terminated_states, self.params, self.generation_mode)
            if self.params.verbose:
                print()
                print("EOS encountered - terminating beam")
                print(new_beam)
            return new_beam
        
        # Standard processing
        new_states = []
        for state in self:
            if new_state := state << a:
                new_states.append(new_state)

        logZ = logsumexp([s.weight for s in new_states]) if new_states else -np.inf
        for state in await self.extend(logZ):
            if new_state := state << a:
                new_states.append(new_state)

        new_state = ByteBeamState(new_states, self.params, self.generation_mode)

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

        logqs = np.stack(logqs, axis=0)  # shape: (num_states, 257)
        logqs[: len(self), -1] = -np.inf  # mask EOT positions of non-extended
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
        return ByteBeamState(new_states, self.params, self.generation_mode)

    def __repr__(self):
        desc = colors.bold % f"Z: {self.logZ}\n" + colors.bold % "Candidates:\n"
        for state in self:
            P = np.exp(state.weight - self.logZ)
            color = colors.green if P > self.params.prune_threshold else colors.red
            desc += f"({color % f'{P:.4f}'}) {repr(state)}\n"
        return desc

    async def prefill(self, bs):
        """Prefill with context, handling EOS tokens appropriately.
        
        During prefill (conditioning), EOS tokens are treated as normal tokens
        and don't cause termination.
        
        Args:
            bs (bytes): Byte sequence to prefill with
            
        Returns:
            (ByteBeamState): New beam state after prefilling
        """
        # Switch to conditioning mode temporarily
        old_mode = self.generation_mode
        self.generation_mode = False
        
        # Update all states to conditioning mode
        for state in self.states:
            state.generation_mode = False
        
        try:
            # Standard prefill process
            state = self
            for b in bs:
                state = await (state.prune() << b)
            
            # Switch back to generation mode
            state.generation_mode = True
            for s in state.states:
                s.generation_mode = True
            
            return state
        
        except Exception:
            # Restore mode on error
            self.generation_mode = old_mode
            for state in self.states:
                state.generation_mode = old_mode
            raise

    async def cleanup(self):
        """Cleans up resources used by the candidates."""
        await asyncio.gather(*[state.cleanup() for state in self])
