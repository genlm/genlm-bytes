import pytest
from genlm.backend import load_model_by_name
from genlm.tokenization.byte_lm.beam import ByteBeam


@pytest.mark.asyncio
async def test_basics():
    llm = load_model_by_name("gpt2")
    K = 5
    qs = b"An apple a day keeps "
    C = ByteBeam(llm, K=K)

    try:
        beam = await C.beam(qs)
        print(beam)
        result = await C.greedy(qs, steps=20)
        print(result)
    finally:
        await C.cleanup()


@pytest.mark.asyncio
async def test_generate():
    llm = load_model_by_name("gpt2-medium")
    K = 5
    qs = b"An apple a day keeps the "
    M = ByteBeam(llm, K=K)

    try:
        output = await M.greedy(qs, steps=12, verbose=True)
        print(repr(output))
        assert output == b"An apple a day keeps the doctor away."
    finally:
        await M.cleanup()
