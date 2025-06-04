import asyncio
from typing import Callable, Literal, List, Tuple
from collections import defaultdict

import numpy as np
from cachetools import LRUCache
from arsenal.maths import logsumexp

from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams
from genlm.control import Potential
from genlm.control.sampler.token import TokenSampler
from genlm.control.util import fast_sample_logprobs
from genlm.control.constant import EOS


def convert_to_logop(op: Literal["sum", "prod", "min"]) -> Callable:
    """Convert a string operation to its log-space equivalent."""
    if op == "sum":
        return lambda x, y: logsumexp([x, y], axis=0)
    elif op == "prod":
        return lambda x, y: x + y
    elif op == "min":
        return lambda x, y: np.minimum(x, y)
    else:
        raise ValueError(f"Invalid operation: {op}. Choose from 'sum', 'prod', 'min'.")


class ByteEnsemble(Potential):
    """
    An ensemble potential combining two language models using a specified log-space operation.

    Attributes:
        p1, p2: The base LM potentials.
        op: A function to combine log-probabilities.
        data_dict_1, data_dict_2: Beam state caches keyed by context (bytes).
        vocabulary: Byte-level vocabulary.
    """

    def __init__(self, p1, p2, op: Callable, data_dict_1, data_dict_2, vocab):
        self.p1 = p1
        self.p2 = p2
        self.op = op
        self.data_dict_1 = data_dict_1
        self.data_dict_2 = data_dict_2
        super().__init__(vocabulary=vocab)

    @classmethod
    async def create(cls, llm1, llm2, op: str, prompt1: bytes, prompt2: bytes):
        """Factory method to initialize beam states from prompts and return a ByteEnsemble instance."""
        beam_params = BeamParams(K=5, prune_threshold=0.1, verbose=True)
        data_dict_1 = defaultdict()
        data_dict_2 = defaultdict()

        async def setup():
            beam1, beam2 = await asyncio.gather(
                ByteBeamState.initial(llm1, beam_params),
                ByteBeamState.initial(llm2, beam_params)
            )
            return await asyncio.gather(beam1.prefill(prompt1), beam2.prefill(prompt2))

        beam_state_1, beam_state_2 = await setup()
        data_dict_1[b""] = beam_state_1
        data_dict_2[b""] = beam_state_2
        return cls(llm1, llm2, convert_to_logop(op), data_dict_1, data_dict_2, vocab=list(range(256)))

    async def _cleanup_cache(self):
        """Remove old entries to avoid cache bloat."""
        max_len = max((len(k) for k in self.data_dict_1), default=0)
        min_len = max_len - 2
        for d in [self.data_dict_1, self.data_dict_2]:
            for k in list(d.keys()):
                if len(k) < min_len:
                    del d[k]

    async def get_beam_states(self, context: List[int]):
        """Fetch beam states for the current context."""
        ctx_bytes = bytes(context)
        await self._cleanup_cache()
        return self.data_dict_1[ctx_bytes], self.data_dict_2[ctx_bytes]
    
    async def prefix(self, context: List[int]):
        """Stub for abstract method."""
        return None  # or raise NotImplementedError if you're sure it's never needed

    async def complete(self, context: List[int]):
        """Stub for abstract method."""
        return None


class ByteEnsembleTokenSampler(TokenSampler):
    """
    Token sampler that draws from an ensemble of potentials using a log-space proposal strategy.

    Args:
        potential: The target ensemble potential.
        proposal: How to combine log-probabilities ('linear', 'abs', etc.).
        n_particles: Number of particles for SMC sampling.
        eos_tokens: List of end-of-sequence tokens.
        max_tokens: Maximum number of tokens to generate.
        models_equal: Flag whether the two potentials have the same base LM.
    """

    def __init__(
        self,
        potential: ByteEnsemble,
        proposal: Literal["linear", "abs", "square", "soft n"] = "linear",
        n_particles: int = 10,
        eos_tokens: List[int] = [],
        max_tokens: int = None,
        models_equal: bool = False,
    ):
        super().__init__(target=potential)
        self.potential = potential
        self.proposal = proposal
        self.n_particles = n_particles
        self.eos_tokens = eos_tokens
        self.max_tokens = max_tokens
        self.models_equal = models_equal

        self.prefix_cache_1 = LRUCache(maxsize=3 * n_particles)
        self.prefix_cache_2 = LRUCache(maxsize=3 * n_particles)
        self.particle_prefix_log_prob_1 = defaultdict()
        self.particle_prefix_log_prob_2 = defaultdict()

        self.prefix_cache_1[()] = 0.0
        self.prefix_cache_2[()] = 0.0

    async def start_weight(self) -> float:
        return 0.0

    async def sample(self, context: List[int]) -> Tuple[int, float, float]:
        """Sample one token from the ensemble distribution."""
        beam1, beam2 = await self.potential.get_beam_states(context)
        logp_1, logp_2 = await beam1.logp_next(), await beam2.logp_next()

        ctx_tuple = tuple(context)
        log_context_weight_1 = self.prefix_cache_1[ctx_tuple]
        log_context_weight_2 = self.prefix_cache_2[ctx_tuple]

        logws1 = log_context_weight_1 + logp_1.ps
        logws2 = log_context_weight_2 + logp_2.ps

        log_shaping_weight_prev = (
            0 if not context else self.potential.op(log_context_weight_1, log_context_weight_2)
        )

        proposal_weights = self.potential.op(logws1, logws2) - log_shaping_weight_prev
        logps = proposal_weights - logsumexp(proposal_weights)
        token_idx = fast_sample_logprobs(logps)[0]

        token = beam1.states[0].trie.trie.trie_decode[token_idx]
        assert token == beam2.states[0].trie.trie.trie_decode[token_idx]

        next_context = bytes(context + [token]) if isinstance(token, int) else bytes(context) + token
        self.potential.data_dict_1[next_context] = await (beam1.prune() << token)
        self.potential.data_dict_2[next_context] = await (beam2.prune() << token)

        new_ctx_tuple = ctx_tuple + (token,)
        self.prefix_cache_1[new_ctx_tuple] = logws1[token_idx]
        self.prefix_cache_2[new_ctx_tuple] = logws2[token_idx]

        if token in self.eos_tokens or (self.max_tokens and len(context) + 1 == self.max_tokens):
            token = EOS
            self.particle_prefix_log_prob_1[ctx_tuple + (EOS,)] = logws1[token_idx]
            self.particle_prefix_log_prob_2[ctx_tuple + (EOS,)] = logws2[token_idx]

        return token, proposal_weights[token_idx] - logps[token_idx], logps[token_idx]

    async def smc(self, n_particles: int, ess_threshold: float, max_tokens: int, critic=None, **kwargs):
        """Run Sequential Monte Carlo inference."""
        from genlm.control.sampler.sequence import EnsembleSMC
        return await EnsembleSMC(self, critic)(
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            max_tokens=max_tokens,
            **kwargs,
        )


async def main():
    llm1 = load_model_by_name("meta-llama/Llama-3.2-1B-Instruct")
    llm2 = load_model_by_name("meta-llama/Llama-3.2-1B-Instruct")

    prompt1 = b"London is "
    prompt2 = b"Paris is "

    ensemble = await ByteEnsemble.create(llm1, llm2, op="prod", prompt1=prompt1, prompt2=prompt2)

    eos_tokens = [
        llm1.byte_vocab[llm1.tokenizer.eos_token_id],
        llm2.byte_vocab[llm2.tokenizer.eos_token_id],
    ]
    sampler = ByteEnsembleTokenSampler(
        ensemble, max_tokens=20, eos_tokens=eos_tokens, n_particles=5
    )

    result = await sampler.smc(n_particles=5, ess_threshold=0.9, max_tokens=20)
    print(result.posterior)


if __name__ == "__main__":
    asyncio.run(main())