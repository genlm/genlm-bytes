import pytest
import asyncio
from datasets import load_dataset

from genlm.backend import load_model_by_name
from genlm.tokenization import ByteBeamState

# Run with: python -m pytest prefill.py --benchmark-only


def load_wikitext():
    return "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1")["test"]["text"])


@pytest.mark.parametrize("K", [5, 10, 20])
@pytest.mark.parametrize("extend_threshold", [None, 0.1, 10])
def test_prefill(benchmark, K, extend_threshold):
    llm = load_model_by_name("gpt2-medium")
    state = asyncio.run(
        ByteBeamState.initial(llm, K=K, extend_threshold=extend_threshold)
    )
    text = load_wikitext()

    loop = asyncio.new_event_loop()

    async def run():
        await state.consume(text[:500].encode("utf-8"))

    benchmark.pedantic(
        lambda: loop.run_until_complete(run()),
        iterations=1,
        rounds=5,
        warmup_rounds=1,
    )

    loop.close()
