import pytest
from genlm.backend import load_model_by_name
from genlm.tokenization.byte_lm.stream import BeamByteStream


@pytest.fixture(scope="module")
def llm():
    return load_model_by_name("gpt2-medium")


@pytest.mark.asyncio
async def test_basics(llm):
    state = await BeamByteStream.initial(llm, K=5)

    try:
        result = await state.greedy(b"An apple a day keeps ", steps=20)
        print(result)
    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_generate(llm):
    state = await BeamByteStream.initial(llm, K=5)

    try:
        output = await state.greedy(
            b"An apple a day keeps the ", steps=12, verbose=True
        )
        print(repr(output))
        assert output == b"An apple a day keeps the doctor away."
    finally:
        await state.cleanup()
