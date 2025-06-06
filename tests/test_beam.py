import pytest
import numpy as np
from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams


@pytest.fixture(scope="module")
def llm():
    return load_model_by_name("gpt2-medium", backend="hf")


@pytest.mark.asyncio
async def test_basics(llm):
    state = await ByteBeamState.initial(
        llm, BeamParams(K=5), trie_opts={"max_batch_size": 100}
    )

    try:
        result = await state.greedy(b"An apple a day keeps ", steps=20)
        print(result)
        result = await state.sample(b"An apple a day keeps ", steps=20)
        print(result)
    finally:
        await state.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("prune_threshold", [0, 0.1])
async def test_generate(llm, prune_threshold):
    state = await ByteBeamState.initial(
        llm,
        BeamParams(
            K=5,
            prune_threshold=prune_threshold,
            verbose=True,
        ),
    )

    try:
        output = await state.greedy(b"An apple a day keeps the ", steps=12)
        print(repr(output))
        assert output == b"An apple a day keeps the doctor away."
    finally:
        await state.cleanup()


# @pytest.mark.parametrize("prune_threshold", [None, 0.1])
# @pytest.mark.asyncio
# async def test_async_batching(llm, prune_threshold):
#     state = await ByteBeamState.initial(
#         llm,
#         BeamParams(
#             K=5,
#             prune_threshold=prune_threshold,
#         ),
#     )

#     try:
#         # warm up
#         await state.greedy(b"I", steps=5)
#         await state.greedy(b"Y", steps=5)

#         start = time.time()
#         concurrent_output = await asyncio.gather(
#             state.greedy(b"I", steps=5),
#             state.greedy(b"Y", steps=5),
#         )
#         concurrent_time = time.time() - start

#         start = time.time()
#         sequential_output_I = await state.greedy(b"I", steps=5)
#         sequential_output_Y = await state.greedy(b"Y", steps=5)
#         sequential_time = time.time() - start

#         print(f"Concurrent requests time: {concurrent_time:.2f} seconds")
#         print(f"Sequential requests time: {sequential_time:.2f} seconds")

#         assert concurrent_output == [sequential_output_I, sequential_output_Y]
#         assert concurrent_time < sequential_time
#     finally:
#         await state.cleanup()


@pytest.mark.parametrize("prune_threshold", [0, 0.1])
@pytest.mark.asyncio
async def test_weights(llm, prune_threshold):
    state = await ByteBeamState.initial(
        llm,
        BeamParams(
            K=5,
            prune_threshold=prune_threshold,
        ),
    )

    try:
        qs = b"An apple a day keeps the"
        for q in qs:
            state = await (state << q)
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


def test_invalid_prune_threshold():
    with pytest.raises(ValueError):
        BeamParams(K=1, prune_threshold=-0.1)
