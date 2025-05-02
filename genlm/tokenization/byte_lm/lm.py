from abc import ABC, abstractmethod
from arsenal.maths import sample_dict


class ByteLM(ABC):
    def __init__(self, V):
        self.V = V

    @abstractmethod
    async def p_next(self, context):
        pass

    async def greedy(self, prompt, steps, verbose=False):
        """
        Generate character-by-character starting from `prompt` using LLM with
        the approximate conditional distribution.
        """
        context = list(prompt)
        for _ in range(steps):
            p = await self.p_next(context)
            if verbose:
                print(repr(context), p.top(5))
            x = p.argmax()
            context.append(x)
        return bytes(context)

    async def sample(self, prompt, steps, verbose=False, draw=sample_dict):
        """
        Generate character-by-character starting from `prompt` using LLM with
        the approximate conditional distribution.
        """
        context = list(prompt)
        for _ in range(steps):
            p = await self.p_next(context)
            if verbose:
                print(repr(context), p.top(5))
            x = draw(p)
            context.append(x)
        return bytes(context)
