import pytest
import numpy as np
import torch
from types import SimpleNamespace

from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams
from genlm.bytes.trie import TokenByteTrie
import genlm.bytes.byte_lm.beam as beam_module


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


def _make_wrapped_trie(children, decode=None, leaf2token_id=None, root=0, eot_token=None):
    inner = SimpleNamespace(
        children=children,
        root=root,
        eot_token=eot_token,
        leaf2token_id=leaf2token_id or {},
        decode=decode or [],
    )
    return SimpleNamespace(trie=inner)


class DummyLMState:
    def __init__(self, history=None):
        self.history = tuple(history or ())

    def __lshift__(self, token_id):
        return DummyLMState(self.history + (token_id,))


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


def test_format_byte_handles_type_error():
    beam = _beam_min()

    class Weird:
        def __str__(self):
            return "weird-value"

    assert beam._format_byte(Weird()) == "weird-value"


def test_traverse_bytes_failure_returns_none():
    beam = _beam_min()
    children = [{1: 1}, {}]
    assert beam._traverse_bytes(children, eot_token=None, start_node=0, bytes_seq=[1, 2]) == (
        None,
        None,
    )


def test_build_path_nodes_invalid_returns_none():
    beam = _beam_min()
    children = [{1: 1}, {}]
    assert beam._build_path_nodes(children, root=0, partial_bytes=[1, 2]) is None


@pytest.mark.asyncio
async def test_adaptive_heal_skips_state_when_partial_path_invalid():
    class DummyHealingState:
        def __init__(self):
            children = [{1: 1}, {}]
            self.trie = _make_wrapped_trie(
                children,
                decode=[b"a"],
                leaf2token_id={},
                eot_token=None,
            )
            self.weight = 0.0
            self.node = 0
            self.mass = torch.zeros(len(children), dtype=torch.float32)
            self.partial = [2]  # Missing path so _build_path_nodes -> None
            self.mode = "with_eos"
            self.lm_state = DummyLMState()

        async def materialize(self):
            return self

    beam = ByteBeamState(states=[DummyHealingState()], params=BeamParams(K=1, heal_max_backoff=1))
    healed = await beam._adaptive_heal(next_byte=1)
    assert healed is None


def test_find_heal_plan_no_plan_verbose():
    trie = TokenByteTrie(decode=[b"a"], device="cpu")
    children = trie.children
    partial_bytes = [ord("a")]

    beam = _beam_min()
    path_nodes = beam._build_path_nodes(children, trie.root, partial_bytes)
    chosen_k, plan = beam._find_heal_plan(
        trie=trie,
        children=children,
        path_nodes=path_nodes,
        partial_bytes=partial_bytes,
        next_byte=ord("z"),
        min_k=0,
        verbose=True,
    )
    assert chosen_k is None
    assert plan is None


def test_plan_commits_tail_split_cap_in_phase_two():
    trie = TokenByteTrie(decode=[b"a", b"b"], device="cpu")
    beam = _beam_min()
    plan = beam._plan_commits(trie, b"ab", ord("c"), heal_max_splits=1)
    assert plan is None


def test_plan_commits_replay_failure_returns_none():
    trie = TokenByteTrie(decode=[b"ab", b"abcd"], device="cpu")
    beam = _beam_min()
    plan = beam._plan_commits(trie, b"abc", ord("z"), heal_max_splits=None)
    assert plan is None


@pytest.mark.asyncio
async def test_apply_commit_plan_invalid_eot_abort(monkeypatch):
    class FakeLazyTrieState:
        def __init__(self, lm_state, trie, node, weight, mass, mode, terminated):
            self.lm_state = lm_state
            self.trie = trie
            self.node = node
            self.weight = weight
            self.mode = mode
            self.terminated = terminated
            if mass is None:
                self._mass = None
            else:
                self._mass = torch.tensor(mass, dtype=torch.float32)

        @property
        def mass(self):
            return self._mass

        @mass.setter
        def mass(self, value):
            self._mass = value

        async def materialize(self):
            if self._mass is None:
                size = len(self.trie.trie.children)
                self._mass = torch.linspace(0.0, 1.0, size)
            return self

    monkeypatch.setattr(beam_module, "LazyTrieState", FakeLazyTrieState)

    children = [
        {None: 3, 7: 1},
        {8: 2},
        {},
        {},
    ]
    trie = _make_wrapped_trie(
        children,
        decode=[b"T0"],
        leaf2token_id={3: 0},
        eot_token=None,
    )
    state = FakeLazyTrieState(
        lm_state=DummyLMState(),
        trie=trie,
        node=2,
        weight=0.0,
        mass=np.linspace(0.0, 0.3, len(children)),
        mode="with_eos",
        terminated=False,
    )

    beam = _beam_min()
    result = await beam._apply_commit_plan(
        state=state,
        trie=trie.trie,
        children=children,
        partial_bytes=[7, 8],
        chosen_k=0,
        plan_positions=[1],  # Invalid because there is no EOT at this node
        next_byte=9,
        base_weight=0.0,
        path_nodes=[0, 1, 2],
        verbose=True,
    )
    assert result is None
