import pytest
import numpy as np

from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams


TEXT = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."
VALKYRIA = "= Valkyria Chronicles III =Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , li"
AMAZING = "Wait... what?! That's amazing-truly incredible!"


@pytest.fixture(scope="module")
def llm():
    return load_model_by_name("gpt2", backend="hf")


async def _advance_bytes(
    llm, text: str, heal: bool, heal_max_backoff=None, heal_max_splits=None
):
    """Helper to advance through text bytes and check if healing works."""
    eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam = await ByteBeamState.initial(
        llm,
        BeamParams(
            K=1,
            eos_tokens=[eos_token],
            heal=heal,
            heal_max_backoff=heal_max_backoff,
            heal_max_splits=heal_max_splits,
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

        return True, None, current
    finally:
        await beam.cleanup()


# -------------------------
# Core healing tests
# -------------------------


@pytest.mark.asyncio
async def test_heal_disabled_fails(llm):
    """Without healing, K=1 beam fails on this text."""
    ok, fail_idx, _ = await _advance_bytes(llm, TEXT, heal=False)
    assert not ok, "Expected failure with heal disabled"
    assert isinstance(fail_idx, int)


@pytest.mark.asyncio
async def test_heal_enabled_succeeds(llm):
    """With healing, K=1 beam completes the text."""
    ok, _, state = await _advance_bytes(llm, TEXT, heal=True)
    assert ok, "Healing should complete the text"
    logp_next = await state.logp_next()
    assert logp_next[257] > -np.inf, "EOS should be reachable"


@pytest.mark.asyncio
async def test_heal_max_backoff_limited_fails(llm):
    """With limited backoff, healing fails on difficult text."""
    ok, fail_idx, _ = await _advance_bytes(llm, TEXT, heal=True, heal_max_backoff=2)
    assert not ok, "Expected failure with heal_max_backoff=2"
    assert isinstance(fail_idx, int)


@pytest.mark.asyncio
async def test_heal_multisplit(llm):
    """Valkyria text requires multi-split healing."""
    ok, _, state = await _advance_bytes(llm, VALKYRIA, heal=True)
    assert ok, "Multi-split healing should complete Valkyria text"
    logp_next = await state.logp_next()
    assert logp_next[257] > -np.inf, "EOS should be reachable"


@pytest.mark.asyncio
async def test_heal_max_splits_zero(llm):
    """With max_splits=0, multi-split is disabled so VALKYRIA text fails."""
    ok, fail_idx, _ = await _advance_bytes(llm, VALKYRIA, heal=True, heal_max_splits=0)
    assert not ok, "Expected failure with max_splits=0 on VALKYRIA text"
    assert isinstance(fail_idx, int)


@pytest.mark.asyncio
async def test_heal_amazing_text(llm):
    """Test healing on text ending with '!' that requires extend() with EOT node 0.

    This text exposed a bug where extend() failed when the EOT node was 0,
    because `if eot_node := get_EOT():` treated 0 as falsy.
    """
    ok, _, state = await _advance_bytes(llm, AMAZING, heal=True)
    assert ok, "Healing should complete the AMAZING text"
    logp_next = await state.logp_next()
    assert logp_next[257] > -np.inf, "EOS should be reachable after extend"


# -------------------------
# ByteBeamState API tests
# -------------------------


@pytest.mark.asyncio
async def test_prefill_and_prune(llm):
    """Test prefill and prune with real LLM."""
    state = await ByteBeamState.initial(llm, BeamParams(K=3))
    try:
        prefilled = await state.prefill(b"Hello ")
        assert len(prefilled.states) > 0

        pruned = prefilled.prune()
        assert isinstance(pruned, ByteBeamState)
        assert len(pruned.states) <= 3
    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_logp_next(llm):
    """Test logp_next returns valid probabilities."""
    state = await ByteBeamState.initial(llm, BeamParams(K=1))
    try:
        prefilled = await state.prefill(b"The ")
        logp = await prefilled.logp_next()

        # logp_next returns LazyByteProbs, access via indexing
        assert logp[ord("a")] <= 0
        assert logp[ord(" ")] <= 0
        assert logp[257] <= 0 or logp[257] == -np.inf
    finally:
        await state.cleanup()


# -------------------------
# TokenHealer helper tests
# -------------------------


def test_format_byte():
    """Test _format_byte helper."""
    from genlm.bytes.byte_lm.heal import TokenHealer

    healer = TokenHealer()

    # Normal bytes
    assert healer._format_byte(65) == "b'A'"
    assert healer._format_byte(0) == "b'\\x00'"

    # Edge cases handled gracefully
    assert isinstance(healer._format_byte(-1), str)
    assert isinstance(healer._format_byte(300), str)


def test_format_byte_exception_path():
    """Test _format_byte exception handling (covers lines 42-43)."""
    from genlm.bytes.byte_lm.heal import TokenHealer

    healer = TokenHealer()

    # Pass something that will cause bytes() to fail
    class BadInt:
        def __index__(self):
            raise ValueError("bad")

        def __str__(self):
            return "bad-value"

    result = healer._format_byte(BadInt())
    assert result == "bad-value"


# -------------------------
# Custom trie tests (no LM needed)
# -------------------------


class MinimalLMState:
    """Minimal LM state for testing - no real LLM needed."""

    def __init__(self, vocab_size=10):
        self.vocab_size = vocab_size

    def __lshift__(self, token_id):
        return MinimalLMState(self.vocab_size)

    async def logp_next(self):
        import torch

        # Return uniform log probabilities
        return torch.log(torch.ones(self.vocab_size) / self.vocab_size)


@pytest.mark.asyncio
async def test_healer_with_custom_trie_path_not_found():
    """Test healing when partial path doesn't exist (covers line 91)."""
    from genlm.bytes.trie import AsyncTokenByteTrie
    from genlm.bytes.byte_lm.heal import TokenHealer
    from genlm.bytes.byte_lm.trie_state import LazyTrieState

    # Simple vocab
    vocab = [b"a", b"ab", b"x"]
    async_trie = AsyncTokenByteTrie.from_vocab(vocab, device="cpu")

    lm_state = MinimalLMState(vocab_size=len(vocab))
    state = LazyTrieState(
        lm_state=lm_state,
        trie=async_trie,
        node=async_trie.trie.root,
        weight=0.0,
        mass=None,
        mode="without_eos",
        terminated=False,
    )
    state = await state.materialize()

    # Wrapper with invalid partial bytes (path doesn't exist)
    class StateWithBadPartial:
        def __init__(self, real_state):
            self.trie = real_state.trie
            self.weight = real_state.weight
            self.node = real_state.node
            self.mass = real_state.mass
            self.mode = real_state.mode
            self.lm_state = real_state.lm_state
            self.partial = [ord("z"), ord("z")]  # 'z' doesn't exist in trie

    bad_state = StateWithBadPartial(state)
    healer = TokenHealer(verbose=True)

    result = await healer._try_at_k(bad_state, k=1, next_byte=ord("x"))
    assert result is None  # Path doesn't exist - covers line 91


@pytest.mark.asyncio
async def test_healer_with_custom_trie_cant_extend():
    """Test when extend fails - no EOT at current position (covers lines 136-137)."""
    from genlm.bytes.trie import AsyncTokenByteTrie
    from genlm.bytes.byte_lm.heal import TokenHealer
    from genlm.bytes.byte_lm.trie_state import LazyTrieState

    # Vocab where "ab" exists but NOT "a" - so after consuming 'a' there's no EOT
    vocab = [b"ab", b"x", b"y"]
    async_trie = AsyncTokenByteTrie.from_vocab(vocab, device="cpu")

    lm_state = MinimalLMState(vocab_size=len(vocab))
    state = LazyTrieState(
        lm_state=lm_state,
        trie=async_trie,
        node=async_trie.trie.root,
        weight=0.0,
        mass=None,
        mode="without_eos",
        terminated=False,
    )
    state = await state.materialize()

    class StateWithPartial:
        def __init__(self, real_state, partial_bytes):
            self.trie = real_state.trie
            self.weight = real_state.weight
            self.node = real_state.node
            self.mass = real_state.mass
            self.mode = real_state.mode
            self.lm_state = real_state.lm_state
            self.partial = partial_bytes

    # partial = "abz" where 'z' doesn't exist after 'ab'
    # At k=2: commit "ab", suffix = "z"
    # After commit at root, try 'z' - not in trie, fail
    # This doesn't hit lines 136-137 because 'z' fails immediately from root

    # To hit lines 136-137, we need:
    # 1. After commit, replay some suffix bytes successfully
    # 2. Then a byte fails
    # 3. extend() returns None (no EOT at current position)

    # partial = "aba" where after committing "ab", we replay "a"
    # 'a' is at root, goes to node-after-a
    # At node-after-a, there's NO EOT (only "ab" is a token, not "a")
    # If next byte in suffix fails, we try extend() -> None

    # partial = "abaz" where:
    # k=2: commit "ab", suffix = "az"
    # Replay 'a' -> at node-after-a
    # Replay 'z' -> fails at node-after-a, try extend() -> NO EOT -> return None

    test_state = StateWithPartial(state, [ord("a"), ord("b"), ord("a"), ord("z")])
    healer = TokenHealer(max_splits=None, verbose=True)

    result = await healer.try_heal(test_state, next_byte=ord("x"))
    assert result is None  # Should hit lines 136-137


@pytest.mark.asyncio
async def test_healer_with_custom_trie_cant_consume_after_extend():
    """Test when byte can't be consumed even after extend (covers line 144)."""
    from genlm.bytes.trie import AsyncTokenByteTrie
    from genlm.bytes.byte_lm.heal import TokenHealer
    from genlm.bytes.byte_lm.trie_state import LazyTrieState

    # Vocab: "a", "ab" - 'a' exists so we CAN extend after consuming 'a'
    # But 'z' isn't in trie at all
    vocab = [b"a", b"ab", b"x"]
    async_trie = AsyncTokenByteTrie.from_vocab(vocab, device="cpu")

    lm_state = MinimalLMState(vocab_size=len(vocab))
    state = LazyTrieState(
        lm_state=lm_state,
        trie=async_trie,
        node=async_trie.trie.root,
        weight=0.0,
        mass=None,
        mode="without_eos",
        terminated=False,
    )
    state = await state.materialize()

    class StateWithPartial:
        def __init__(self, real_state, partial_bytes):
            self.trie = real_state.trie
            self.weight = real_state.weight
            self.node = real_state.node
            self.mass = real_state.mass
            self.mode = real_state.mode
            self.lm_state = real_state.lm_state
            self.partial = partial_bytes

    # partial = "aaz" where:
    # k=1: commit "a", suffix = "az"
    # Replay 'a' -> at node-after-a (has EOT for "a")
    # Replay 'z' -> fails, try extend() -> succeeds (commit "a") -> at root
    # Retry 'z' -> fails at root too! -> return None (line 144)

    test_state = StateWithPartial(state, [ord("a"), ord("a"), ord("z")])
    healer = TokenHealer(max_splits=None, verbose=True)

    result = await healer.try_heal(test_state, next_byte=ord("x"))
    assert result is None  # Should hit line 144


@pytest.mark.asyncio
async def test_healer_with_custom_trie_final_extend():
    """Test final extend path (covers lines 155-163)."""
    from genlm.bytes.trie import AsyncTokenByteTrie
    from genlm.bytes.byte_lm.heal import TokenHealer
    from genlm.bytes.byte_lm.trie_state import LazyTrieState

    # Vocab: "a", "ab", "x"
    # After consuming "ab", there's an EOT
    # next_byte 'x' is at root but NOT after "ab"
    vocab = [b"a", b"ab", b"x"]
    async_trie = AsyncTokenByteTrie.from_vocab(vocab, device="cpu")

    lm_state = MinimalLMState(vocab_size=len(vocab))
    state = LazyTrieState(
        lm_state=lm_state,
        trie=async_trie,
        node=async_trie.trie.root,
        weight=0.0,
        mass=None,
        mode="without_eos",
        terminated=False,
    )
    state = await state.materialize()

    class StateWithPartial:
        def __init__(self, real_state, partial_bytes):
            self.trie = real_state.trie
            self.weight = real_state.weight
            self.node = real_state.node
            self.mass = real_state.mass
            self.mode = real_state.mode
            self.lm_state = real_state.lm_state
            self.partial = partial_bytes

    # partial = "abab" where:
    # k=2: commit "ab", suffix = "ab"
    # Replay 'a' -> node-after-a
    # Replay 'b' -> node-after-ab (has EOT for "ab")
    # Try next_byte 'x' -> not at node-after-ab
    # Try final extend() -> succeeds (commit "ab") -> at root
    # Retry 'x' -> succeeds at root! -> lines 155-163

    test_state = StateWithPartial(state, [ord("a"), ord("b"), ord("a"), ord("b")])
    healer = TokenHealer(max_splits=None, verbose=True)

    result = await healer.try_heal(test_state, next_byte=ord("x"))
    assert result is not None  # Should succeed via final extend path
