from abc import ABC, abstractmethod
from arsenal.maths import sample_dict


class ByteLM(ABC):
    def __init__(self, V):
        self.V = V

    @abstractmethod
    async def logp_next(self, context):
        pass

    async def greedy(self, prompt, steps, verbose=False):
        """
        Generate character-by-character starting from `prompt` using LLM with
        the approximate conditional distribution.
        """
        context = list(prompt)
        for _ in range(steps):
            p = await self.logp_next(context)
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
            p = await self.logp_next(context)
            if verbose:
                print(repr(context), p.top(5))
            x = draw(p)
            context.append(x)
        return bytes(context)


class StateLM:
    def __init__(self, llm, context):
        self.llm = llm
        self.context = context
        self._n_calls = 0

    @classmethod
    def initial(cls, llm, initial_context=None):
        if initial_context is None:
            initial_context = [llm.tokenizer.bos_token_id]
        return cls(llm, initial_context)

    def __lshift__(self, token):
        return StateLM(self.llm, self.context + [token])

    async def logp_next(self):
        self._n_calls += 1
        return await self.llm.next_token_logprobs(self.context)

    def __repr__(self):
        context = [self.llm.byte_vocab[x] for x in self.context]
        return f"StateLM(context={context})"

    def clone(self):
        return StateLM(self.llm, self.context.copy())
