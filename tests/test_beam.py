import pytest
from genlm.backend import load_model_by_name
from genlm.tokenization.byte_lm.beam import ByteBeam


@pytest.mark.asyncio
async def test_basics():
    llm = load_model_by_name("gpt2")
    C = ByteBeam(llm, K=5)
    context = b"An apple a day keeps "
    beam = await C.beam(context)
    print(beam)
    result = await C.greedy(context, steps=20)
    print(result)


@pytest.mark.asyncio
async def test_generate():
    llm = load_model_by_name("gpt2")
    K = 5
    qs = b"An apple a day keeps the "
    M = ByteBeam(llm, K=K)
    output = await M.greedy(qs, steps=12, verbose=True)
    print(repr(output))
    assert output == b"An apple a day keeps the doctor away."
