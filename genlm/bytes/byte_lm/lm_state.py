from numpy import exp
from arsenal import colors
from abc import ABC, abstractmethod
from arsenal.maths import sample_dict

from ..util import escape


class StatefulTokenizedLM:
    def __init__(self, model, context, n_calls=0, max_context_length=None):
        self.model = model
        self.context = context
        self._n_calls = n_calls
        self.max_context_length = max_context_length

    @classmethod
    def initial(cls, model, initial_context=None, max_context_length=None):
        if initial_context is None:
            initial_context = [model.tokenizer.bos_token_id]
        return cls(model, initial_context, max_context_length=max_context_length)

    def __lshift__(self, token):
        assert isinstance(token, int)
        if (
            self.max_context_length is not None
            and len(self.context) >= self.max_context_length
        ):
            self.context = self.context[-(self.max_context_length - 1) :]
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
            Q = (await state.logp_next()).materialize()
            b = Q.argmax()
            state = await (state.prune() << b)
            context.append(b)
        return bytes(context)

    async def sample(self, context, steps, draw=sample_dict):
        context = list(context)
        state = await self.prefill(context)
        for _ in range(steps):
            Q = (await state.logp_next()).materialize()
            b = draw(Q.map_values(exp))
            state = await (state.prune() << b)
            context.append(b)
        return bytes(context)

    async def cleanup(self):
        pass
