from abc import ABC, abstractmethod
from genlm.tokenization.util import unflatten


class ByteStreamLM(ABC):
    def __init__(self, V: set, context: list = None):
        self.V = V
        self.context = context

    @abstractmethod
    async def step(self, q):
        pass

    @property
    @abstractmethod
    async def logp_next(self):
        pass

    async def multi_step(self, qs):
        state = self
        for q in qs:
            state = await state.step(q)
        return state

    async def greedy(self, prompt, steps, verbose=False):
        state = await self.multi_step(list(prompt))
        for _ in range(steps):
            p = await state.logp_next
            if verbose:
                print(state, p.top(5))
            x = p.argmax()
            state = await state.step(x)
        return bytes(unflatten(state.context))

    def __repr__(self):
        return f"{self.__class__.__name__}({bytes(unflatten(self.context))})"

    async def cleanup(self):
        pass
