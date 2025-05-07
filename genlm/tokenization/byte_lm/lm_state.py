from numpy import exp
from arsenal import colors
from abc import ABC, abstractmethod
from arsenal.maths import sample_dict
from genlm.tokenization.util import flatten
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
    def __init__(self, context: tuple = ()):
        self.context = context

    @abstractmethod
    async def step(self, q: int):
        pass

    def prune(self):
        return self

    async def consume(self, qs):
        state = self
        for q in qs:
            state = state.prune()
            state = await state.step(q)
        return state

    @property
    @abstractmethod
    async def logp_next(self):
        pass

    async def greedy(self, prompt, steps):
        state = await self.consume(list(prompt))
        for _ in range(steps):
            logp = await state.logp_next
            x = logp.argmax()
            state = state.prune()
            state = await state.step(x)
        return bytes(flatten(state.context))

    async def sample(self, prompt, steps, draw=sample_dict):
        state = await self.consume(list(prompt))
        for _ in range(steps):
            logp = await state.logp_next
            x = draw(logp.map_values(exp))
            state = state.prune()
            state = await state.step(x)
        return bytes(flatten(state.context))

    async def cleanup(self):
        pass
