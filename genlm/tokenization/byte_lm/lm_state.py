from numpy import exp
from arsenal import colors
from abc import ABC, abstractmethod
from arsenal.maths import sample_dict

from ..util import escape


class StatefulTokenizedLM:
    def __init__(self, model, context, n_calls=0):
        self.model = model
        self.context = context
        self._n_calls = n_calls

    @classmethod
    def initial(cls, model, initial_context=None):
        if initial_context is None:
            initial_context = [model.tokenizer.bos_token_id]
        return cls(model, initial_context)

    def __lshift__(self, token):
        assert isinstance(token, int)
        return StatefulTokenizedLM(
            self.model, self.context + [token], n_calls=self._n_calls
        )

    @property
    async def logp_next(self):
        self._n_calls += 1
        return await self.model.next_token_logprobs(self.context)

    def __repr__(self):
        return colors.purple % (
            "|".join([escape(self.model.byte_vocab[x]) for x in self.context])
        )


class StatefulByteLM(ABC):
    @abstractmethod
    async def __lshift__(self, b: int):
        pass

    def prune(self):
        return self

    @abstractmethod
    async def logp_next(self):
        pass

    async def prefill(self, bs):
        "Prefill the beam with bytes"
        state = self
        for b in bs:
            state = await (state.prune() << b)
        return state

    async def greedy(self, context, steps):
        context = list(context)
        state = await self.prefill(context)
        for _ in range(steps):
            Q = await state.logp_next()
            b = Q.argmax()
            state = await (state.prune() << b)
            context.append(b)
        return bytes(context)

    async def sample(self, context, steps, draw=sample_dict):
        context = list(context)
        state = await self.prefill(context)
        for _ in range(steps):
            Q = await state.logp_next()
            b = draw(Q.map_values(exp))
            state = await (state.prune() << b)
            context.append(b)
        return bytes(context)

    async def cleanup(self):
        pass
