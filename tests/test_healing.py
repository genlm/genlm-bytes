import pytest
import numpy as np

from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams


TEXT = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."
VALKYRIA = "= Valkyria Chronicles III =Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , li"
AMAZING = "Wait... what?! That's amazing-truly incredible!"

HEAL_TEST_CASES = [
    ("Boulter", TEXT),
    ("Valkyria", VALKYRIA),
    ("Amazing", AMAZING),
]


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
@pytest.mark.parametrize("name,text", HEAL_TEST_CASES)
async def test_heal_succeeds(llm, name, text):
    """With healing, K=1 beam completes various texts."""
    ok, _, state = await _advance_bytes(llm, text, heal=True)
    assert ok, f"Healing should complete: {name}"
    logp_next = await state.logp_next()
    assert logp_next[257] > -np.inf, f"EOS should be reachable: {name}"


@pytest.mark.asyncio
async def test_heal_max_backoff_limited_fails(llm):
    """With limited backoff, healing fails on difficult text."""
    ok, fail_idx, _ = await _advance_bytes(llm, TEXT, heal=True, heal_max_backoff=2)
    assert not ok, "Expected failure with heal_max_backoff=2"
    assert isinstance(fail_idx, int)


@pytest.mark.asyncio
async def test_heal_max_splits_zero(llm):
    """With max_splits=0, multi-split is disabled so VALKYRIA text fails."""
    ok, fail_idx, _ = await _advance_bytes(llm, VALKYRIA, heal=True, heal_max_splits=0)
    assert not ok, "Expected failure with max_splits=0 on VALKYRIA text"
    assert isinstance(fail_idx, int)


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
    """Test healing when partial path doesn't exist"""
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

    trie = bad_state.trie.trie
    base_weight = bad_state.weight - (
        bad_state.mass[bad_state.node] - bad_state.mass[trie.root]
    )

    result = await healer._try_at_k(
        bad_state, trie, base_weight, k=1, next_byte=ord("x")
    )
    assert result is None  # Path doesn't exist


@pytest.mark.asyncio
async def test_healer_with_custom_trie_cant_extend():
    """Test when extend fails - no EOT at current position"""
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
    assert result is None


@pytest.mark.asyncio
async def test_healer_with_custom_trie_cant_consume_after_extend():
    """Test when byte can't be consumed even after extend"""
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
    # Retry 'z' -> fails at root too -> return None

    test_state = StateWithPartial(state, [ord("a"), ord("a"), ord("z")])
    healer = TokenHealer(max_splits=None, verbose=True)

    result = await healer.try_heal(test_state, next_byte=ord("x"))
    assert result is None


@pytest.mark.asyncio
async def test_healer_with_custom_trie_final_extend():
    """Test final extend path"""
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
    # Retry 'x' -> succeeds at root

    test_state = StateWithPartial(state, [ord("a"), ord("b"), ord("a"), ord("b")])
    healer = TokenHealer(max_splits=None)

    result = await healer.try_heal(test_state, next_byte=ord("x"))
    assert result is not None


@pytest.mark.asyncio
async def test_healer_weight_calculation():
    """Test that healing computes weights correctly.

    Verifies the healed state weight matches manually computed expected value.
    """
    from genlm.bytes.trie import AsyncTokenByteTrie
    from genlm.bytes.byte_lm.heal import TokenHealer
    from genlm.bytes.byte_lm.trie_state import LazyTrieState

    # Vocab: token 0 = "a", token 1 = "ab", token 2 = "x"
    vocab = [b"a", b"ab", b"x"]
    async_trie = AsyncTokenByteTrie.from_vocab(vocab, device="cpu")

    lm_state = MinimalLMState(vocab_size=len(vocab))

    # Create initial state at root
    initial_state = LazyTrieState(
        lm_state=lm_state,
        trie=async_trie,
        node=async_trie.trie.root,
        weight=0.0,
        mass=None,
        mode="without_eos",
        terminated=False,
    )
    initial_state = await initial_state.materialize()

    # Consume 'a' to get to a real state (not mocked partial)
    state_after_a = initial_state << ord("a")
    assert state_after_a is not None

    # Now we're at partial="a". If we try to consume 'x', it should fail
    # because 'x' is not a continuation of 'a' in this trie
    cant_continue = state_after_a << ord("x")
    assert cant_continue is None, "Should not be able to consume 'x' after 'a'"

    # Heal to consume 'x'
    healer = TokenHealer()
    healed = await healer.try_heal(state_after_a, next_byte=ord("x"))
    assert healed is not None, "Healing should succeed"

    # Manually compute expected weight:
    # 1. Healing commits token "a" (token_id=0) at k=1
    # 2. Then consumes 'x' from root
    #
    # After committing "a":
    #   - LM state advances by token 0
    #   - weight = base_weight + (mass[eot_node_for_a] - mass[root])
    #   - where base_weight = state.weight - (mass[state.node] - mass[root])
    #
    # After consuming 'x' from root:
    #   - weight += (mass[node_after_x] - mass[root])

    trie = async_trie.trie
    mass = state_after_a.mass

    # Find the EOT node for "a"
    node_after_a = trie.children[trie.root].get(ord("a"))
    eot_node_for_a = trie.children[node_after_a].get(trie.eot_token)
    assert eot_node_for_a is not None

    # base_weight undoes the path from root to node_after_a
    base_weight = state_after_a.weight - (mass[state_after_a.node] - mass[trie.root])

    # weight after committing "a"
    weight_after_commit = base_weight + (mass[eot_node_for_a] - mass[trie.root])

    # After committing, we need the mass from the new LM state (after token 0)
    # The healed state has already materialized with new mass
    # So we verify against the healed state directly

    # The healed state should be at node_after_x with partial = [ord('x')]
    assert healed.partial == [ord("x")], f"Expected partial [120], got {healed.partial}"

    # Verify weight is reasonable (not -inf, not 0 since we've consumed tokens)
    assert healed.weight < 0, "Weight should be negative (log prob)"
    assert healed.weight > -np.inf, "Weight should not be -inf"

    # Compute expected weight:
    # With uniform probs over 3 tokens: log(1/3) ≈ -1.099
    # mass[root] = logsumexp([log(1/3), log(1/3), log(1/3)]) = log(1) = 0
    # mass[eot_node] = log(1/3) for any single token
    # mass[node_after_x] = log(1/3) since only "x" is reachable
    #
    # After healing:
    # 1. Commit "a": weight = base_weight + (mass[eot_a] - mass[root]) = 0 + log(1/3) - 0 = log(1/3)
    # 2. Materialize gets new mass from new LM state (still uniform)
    # 3. Consume 'x': weight += (mass[node_x] - mass[root]) = log(1/3) + log(1/3) - 0 = 2*log(1/3)

    log_one_third = np.log(1 / 3)
    expected_weight = 2 * log_one_third  # ≈ -2.197

    assert np.isclose(healed.weight, expected_weight, rtol=0.01), (
        f"Expected weight ≈ {expected_weight:.4f}, got {healed.weight:.4f}"
    )
