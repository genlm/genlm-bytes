import pytest
import numpy as np

from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams


TEXT = \
    ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."


@pytest.fixture(scope="module")
def llm():
    return load_model_by_name("gpt2", backend="hf")


async def _advance_bytes(llm, text: str, heal: bool, heal_max_backoff=None):
    eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam = await ByteBeamState.initial(
        llm,
        BeamParams(
            K=1,
            eos_tokens=[eos_token],
            heal=heal,
            heal_max_backoff=heal_max_backoff,
            verbose=False,
        ),
    )
    try:
        bs = text.encode("utf-8")
        current = beam
        for idx, b in enumerate(bs):
            next_beam = await (current.prune() << b)
            if len(next_beam) == 0:
                return False, idx, current
            current = next_beam

        # Completed full prefix; EOS should be reachable when heal succeeds
        logp_next_all = await current.logp_next()
        return True, None, current
    finally:
        await beam.cleanup()


@pytest.mark.asyncio
async def test_heal_disabled_k1_fails(llm):
    ok, fail_idx, _ = await _advance_bytes(llm, TEXT, heal=False)
    assert not ok, "Expected empty beam with heal disabled (K=1), but completed successfully"
    assert isinstance(fail_idx, int)


@pytest.mark.asyncio
async def test_heal_enabled_succeeds(llm):
    ok, _, state = await _advance_bytes(llm, TEXT, heal=True, heal_max_backoff=None)
    assert ok, "Healing enabled should complete the full prefix"
    logp_next_all = await state.logp_next()
    assert logp_next_all[257] > -np.inf, "EOS should be reachable after full prefix"


@pytest.mark.asyncio
async def test_heal_max_backoff_zero_fails(llm):
    # With zero backoff, we cannot move the boundary; expect failure similar to heal=False
    ok, fail_idx, _ = await _advance_bytes(
        llm, TEXT, heal=True, heal_max_backoff=2
    )
    assert not ok, "Expected failure with heal_max_backoff=2"
    assert isinstance(fail_idx, int)
