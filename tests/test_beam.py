import pytest
import time
import numpy as np
import asyncio
from genlm.backend import load_model_by_name
from genlm.tokenization.byte_lm import ByteBeamState


@pytest.fixture(scope="module")
def llm():
    return load_model_by_name("gpt2-medium")


@pytest.mark.asyncio
async def test_basics(llm):
    state = await ByteBeamState.initial(llm, K=5)

    try:
        result = await state.greedy(b"An apple a day keeps ", steps=20)
        print(result)
    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_generate(llm):
    state = await ByteBeamState.initial(llm, K=5)

    try:
        output = await state.greedy(
            b"An apple a day keeps the ", steps=12, verbose=True
        )
        print(repr(output))
        assert output == b"An apple a day keeps the doctor away."
        output = await state.sample(
            b"An apple a day keeps the ", steps=12, verbose=True
        )
        print(repr(output))
    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_async_batching(llm):
    state = await ByteBeamState.initial(llm, K=5)

    try:
        # warm up
        await state.greedy(b"I", steps=5)
        await state.greedy(b"Y", steps=5)

        start = time.time()
        concurrent_output = await asyncio.gather(
            state.greedy(b"I", steps=5),
            state.greedy(b"Y", steps=5),
        )
        concurrent_time = time.time() - start

        start = time.time()
        sequential_output_I = await state.greedy(b"I", steps=5)
        sequential_output_Y = await state.greedy(b"Y", steps=5)
        sequential_time = time.time() - start

        print(f"Concurrent requests time: {concurrent_time:.2f} seconds")
        print(f"Sequential requests time: {sequential_time:.2f} seconds")

        assert concurrent_output == [sequential_output_I, sequential_output_Y]
        assert concurrent_time < sequential_time
    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_weights(llm):
    state = await ByteBeamState.initial(llm, K=5)

    try:
        qs = b"An apple a day keeps the"
        for q in qs:
            state = await state.step(q)
            for candidate in state.states:
                context = candidate.lm_state.context
                llm = candidate.lm_state.model
                want = 0
                for i in range(1, len(context)):
                    logps = await llm.next_token_logprobs(context[:i])
                    want += logps[context[i]]
                want += candidate.mass[candidate.node]
                assert np.isclose(want, candidate.weight, rtol=0.01)
            state = state.prune()
    finally:
        await state.cleanup()
