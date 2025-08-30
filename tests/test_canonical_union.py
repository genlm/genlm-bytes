import pytest

from genlm.bytes.byte_lm.beam import ByteBeamState, BeamParams


class DummyLMState:
    def __init__(self, context):
        self.context = context


class DummyState:
    def __init__(self, weight, context, node, is_canonical=False):
        self.weight = weight
        self.lm_state = DummyLMState(context)
        self.node = node
        # Minimal attributes to satisfy ByteBeamState internals
        self.trie = type("T", (), {"trie": type("TT", (), {"decode": []})()})()
        self.is_canonical = is_canonical


@pytest.fixture
def fake_is_canonical(monkeypatch):
    from genlm.bytes.byte_lm.beam import ByteBeamState as B

    def _fake(self, s):
        return getattr(s, "is_canonical", False)

    monkeypatch.setattr(B, "_is_canonical_state", _fake)


def test_prune_unions_canonicals(fake_is_canonical):
    # Three states with decreasing weight; only the last two are canonical.
    s1 = DummyState(weight=0.0, context=[1], node=1, is_canonical=False)
    s2 = DummyState(weight=-0.1, context=[2], node=2, is_canonical=True)
    s3 = DummyState(weight=-0.2, context=[3], node=3, is_canonical=True)

    params = BeamParams(K=1, keep_all_canonical=True)
    beam = ByteBeamState([s1, s2, s3], params)

    pruned = beam.prune()

    # Normally K=1 would keep only the best state (s1). With the canonical-union
    # behavior enabled, s2 and s3 are kept as well, so size > K.
    assert len(pruned) == 3


def test_prune_respects_k_without_flag(fake_is_canonical):
    s1 = DummyState(weight=0.0, context=[1], node=1, is_canonical=False)
    s2 = DummyState(weight=-0.1, context=[2], node=2, is_canonical=True)
    s3 = DummyState(weight=-0.2, context=[3], node=3, is_canonical=True)

    params = BeamParams(K=1, keep_all_canonical=False)
    beam = ByteBeamState([s1, s2, s3], params)

    pruned = beam.prune()

    # Without the flag, we should obey K strictly.
    assert len(pruned) == 1

