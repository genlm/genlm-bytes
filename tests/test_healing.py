import pytest
import numpy as np

from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams
from genlm.bytes.trie import TokenByteTrie


TEXT = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."


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
            verbose=True,
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

        # Completed full prefix; tests will check EOS reachability on returned state
        return True, None, current
    finally:
        await beam.cleanup()


@pytest.mark.asyncio
async def test_heal_disabled_k1_fails(llm):
    ok, fail_idx, _ = await _advance_bytes(llm, TEXT, heal=False)
    assert not ok, (
        "Expected empty beam with heal disabled (K=1), but completed successfully"
    )
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
    ok, fail_idx, _ = await _advance_bytes(llm, TEXT, heal=True, heal_max_backoff=2)
    assert not ok, "Expected failure with heal_max_backoff=2"
    assert isinstance(fail_idx, int)


@pytest.mark.asyncio
async def test_heal_multisplit(llm):
    # This case previously required multi-split healing around "Valkyria".
    VALKYRIA = "= Valkyria Chronicles III =Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , li"
    ok, _, state = await _advance_bytes(llm, VALKYRIA, heal=True, heal_max_backoff=None)
    assert ok, "Healing enabled should complete the full Valkyria prefix"
    logp_next_all = await state.logp_next()
    assert logp_next_all[257] > -np.inf, (
        "EOS should be reachable after completing Valkyria prefix"
    )


# -------------------------
# Targeted coverage for beam.py
# -------------------------


def _beam_min():
    # Minimal beam just to access internal helpers
    return ByteBeamState(states=[], params=BeamParams(K=1))


def test_plan_commits_empty_suffix():
    trie = TokenByteTrie(decode=[b"a", b"b"])  # root children: 'a', 'b'
    beam = _beam_min()

    # S empty, next_byte reachable at root
    plan = beam._plan_commits(trie, b"", ord("a"), heal_max_splits=None)
    assert plan == []

    # S empty, next_byte unreachable at root
    plan = beam._plan_commits(trie, b"", ord("z"), heal_max_splits=None)
    assert plan is None


def test_plan_commits_no_last_eot():
    # No token ends at 'a' or 'ab' => no EOT inside the segment 'ab'
    trie = TokenByteTrie(decode=[b"abc", b"abx"])  # missing 'ab'
    beam = _beam_min()

    plan = beam._plan_commits(trie, b"abz", ord("z"), heal_max_splits=None)
    assert plan is None


def test_plan_commits_heal_max_splits():
    # 'ab' exists, so last_eot_in_seg=2, but splits are disallowed
    trie = TokenByteTrie(decode=[b"ab", b"abc"])  # EOT at 'ab'
    beam = _beam_min()

    plan = beam._plan_commits(trie, b"abz", ord("z"), heal_max_splits=0)
    assert plan is None


def test_plan_commits_until_reachable():
    # After consuming S='a', to produce next_byte 'b' we must commit at 'a'
    trie = TokenByteTrie(decode=[b"a", b"aa", b"b"])  # EOT at 'a'; 'b' only from root
    beam = _beam_min()

    plan = beam._plan_commits(trie, b"a", ord("b"), heal_max_splits=None)
    assert plan == [1]


@pytest.mark.asyncio
async def test_prefill_and_prune_real_llm(llm):
    # Real ByteBeamState interaction without fakes
    state = await ByteBeamState.initial(llm, BeamParams(K=3))
    try:
        prefilled = await state.prefill(b"Hello ")
        assert len(prefilled.states) > 0

        pruned = prefilled.prune()
        assert isinstance(pruned, ByteBeamState)
        assert len(pruned.states) <= 3
    finally:
        await state.cleanup()


def test_plan_commits_tail_no_eot():
    # Only token 'abc', S='ab' has no EOT inside tail, next byte unreachable => None
    trie = TokenByteTrie(decode=[b"abc"])  # Only full token at 'abc'
    beam = _beam_min()
    plan = beam._plan_commits(trie, b"ab", ord("z"), heal_max_splits=None)
    assert plan is None
