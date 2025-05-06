import numpy as np
from arsenal import colors
from abc import ABC, abstractmethod
from arsenal.maths import sample_dict
from genlm.tokenization.util import unflatten
from genlm.backend import load_model_by_name


class StatefulTokenizedLM:
    def __init__(self, model, context):
        self.model = model
        self.context = context
        self._n_calls = 0

    @classmethod
    def initial(cls, model, initial_context=None):
        if initial_context is None:
            initial_context = [model.tokenizer.bos_token_id]
        return cls(model, initial_context)

    @classmethod
    def initial_from_name(cls, name, initial_context=None, **kwargs):
        model = load_model_by_name(name, **kwargs)
        return cls.initial(model, initial_context)

    def __lshift__(self, token):
        assert isinstance(token, int)
        return StatefulTokenizedLM(self.model, self.context + [token])

    @property
    async def logp_next(self):
        self._n_calls += 1
        return await self.model.next_token_logprobs(self.context)

    def __repr__(self):
        return colors.purple % (
            "|".join([repr(self.model.byte_vocab[x]) for x in self.context])
        )

    def clone(self):
        return StatefulTokenizedLM(self.model, self.context.copy())


class StatefulByteLM(ABC):
    def __init__(self, V: set, context: tuple = ()):
        self.V = V
        self.context = context

    @abstractmethod
    async def step(self, q: int, verbose=False):
        pass

    def prune(self):
        return self

    async def consume(self, qs, verbose=False):
        state = self
        for q in qs:
            state = await state.step(q, verbose)
            state = state.prune()
        return state

    @property
    @abstractmethod
    async def logp_next(self):
        pass

    async def greedy(self, prompt, steps, verbose=False):
        state = await self.consume(list(prompt), verbose)
        for _ in range(steps):
            logp = await state.logp_next
            if verbose:
                print(state, logp.top(5))
            x = logp.argmax()
            state = await state.step(x, verbose)
            state = state.prune()
        return bytes(unflatten(state.context))

    async def sample(self, prompt, steps, verbose=False, draw=sample_dict):
        state = await self.consume(list(prompt), verbose)
        for _ in range(steps):
            logp = await state.logp_next
            if verbose:
                print(state, logp.top(5))
            x = draw(logp.map_values(np.exp))
            state = await state.step(x, verbose)
            state = state.prune()
        return bytes(unflatten(state.context))

    def __repr__(self):
        return f"{bytes(unflatten(self.context))}"

    async def cleanup(self):
        pass
