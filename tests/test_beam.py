import torch
import pytest
import numpy as np
from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams
from genlm.bytes.trie import EOS


def find_token_id_by_bytes(byte_vocab, target_bytes):
    """Find the token ID for a given byte string in a list of Token objects."""
    for token in byte_vocab:
        if token.byte_string == target_bytes:
            return token.token_id
    raise ValueError(f"{target_bytes} is not in byte_vocab")


@pytest.fixture(scope="module")
def llm():
    return load_model_by_name("gpt2-medium", backend="hf")


@pytest.mark.asyncio
async def test_basics(llm):
    # No EOS tokens for basic test
    state = await ByteBeamState.initial(
        llm, BeamParams(K=5), trie_opts={"max_batch_size": 100}
    )

    try:
        result = await state.greedy(b"An apple a day keeps ", steps=20)
        print(result)
        result = await state.sample(b"An apple a day keeps ", steps=20)
        print(result)
    finally:
        await state.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("prune_threshold", [0, 0.1])
async def test_generate(llm, prune_threshold):
    # No EOS tokens - basic generation test
    state = await ByteBeamState.initial(
        llm,
        BeamParams(
            K=5,
            prune_threshold=prune_threshold,
            verbose=True,
        ),
    )

    try:
        output = await state.greedy(b"An apple a day keeps the ", steps=12)
        print(repr(output))
        assert output == b"An apple a day keeps the doctor away."
    finally:
        await state.cleanup()


@pytest.mark.parametrize("prune_threshold", [0, 0.1])
@pytest.mark.asyncio
async def test_weights(llm, prune_threshold):
    state = await ByteBeamState.initial(
        llm,
        BeamParams(
            K=5,
            prune_threshold=prune_threshold,
        ),
    )

    try:
        qs = b"An apple a day keeps the"
        for q in qs:
            state = await (state << q)
            for candidate in state.states:
                context = candidate.lm_state.context
                llm = candidate.lm_state.model
                want = 0
                for i in range(1, len(context)):
                    logps = await llm.next_token_logprobs(context[:i])
                    want += logps[context[i]]
                want += candidate.mass[candidate.node]
                assert np.isclose(want, candidate.weight, rtol=0.01)
            state = state.prune()
    finally:
        await state.cleanup()


def test_invalid_prune_threshold():
    with pytest.raises(ValueError):
        BeamParams(K=1, prune_threshold=-0.1)


def test_beam_params_eos_tokens_deprecation():
    """Test that the deprecated eos_tokens kwarg works and warns."""
    with pytest.warns(DeprecationWarning, match="eos_tokens.*deprecated"):
        params = BeamParams(K=3, eos_tokens=[b".", b"!"])
    assert params.eos_byte_strings == {b".", b"!"}


def test_beam_params_eos_tokens_and_byte_strings_conflict():
    """Test that specifying both eos_tokens and eos_byte_strings raises."""
    with pytest.raises(TypeError, match="Cannot specify both"):
        BeamParams(K=3, eos_byte_strings=[b"."], eos_tokens=[b"!"])


# EOS-specific tests
@pytest.mark.asyncio
async def test_eos_manual_configuration(llm):
    """Test manual EOS token configuration."""
    manual_eos = [b".", b"!", b"?"]
    params = BeamParams(K=3, eos_byte_strings=manual_eos)
    state = await ByteBeamState.initial(llm, params)

    try:
        for state in state.states:
            assert state.trie.trie.eos_byte_strings == set(manual_eos)
            assert state.trie.trie.eos_node is not None

    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_eos_disabled(llm):
    """Test EOS functionality disabled."""
    params = BeamParams(K=3, eos_byte_strings=set())  # Empty set = no EOS
    state = await ByteBeamState.initial(llm, params)

    try:
        # Check that no EOS tokens were configured
        assert not any(state.trie.trie.eos_byte_strings for state in state.states)

        # check that EOS isn't available
        logp_next = await state.logp_next()
        probs = logp_next.materialize()
        assert 257 in probs
        assert probs[257] == -np.inf

    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_eos_termination(llm):
    """Test that EOS byte terminates sequences properly."""
    params = BeamParams(K=3, eos_byte_strings=[b"!"])
    state = await ByteBeamState.initial(llm, params)

    try:
        new_state = await (state << EOS)
        assert all(state.terminated for state in new_state.states)

        eos_token_id = find_token_id_by_bytes(llm.byte_vocab, b"!")
        lm_context = [llm.tokenizer.eos_token_id]
        target_weight = (await llm.next_token_logprobs(lm_context))[eos_token_id]

        assert all(
            np.isclose(state.weight, target_weight, rtol=1e-5)
            for state in new_state.states
        )
    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_can_generate_with_eos_in_prompt(llm):
    params = BeamParams(K=10, eos_byte_strings=[b"\n", b"\n\n"])
    state = await ByteBeamState.initial(llm, params)

    try:
        for trie_state in state.states:
            trie = trie_state.trie.trie
            assert b"\n" in trie.eos_byte_strings
            assert b"\n\n" in trie.eos_byte_strings

        # Test prefill with model EOS token (conditioning mode)
        context_with_eos = b"Hello world" + b"\n" + b" This continues."
        prefilled_state = await state.prefill(context_with_eos)
        assert len(prefilled_state.states) > 0

        # Test greedy generation for 10 steps after prefill
        generated_context = await prefilled_state.greedy(context_with_eos, 10)
        # Should have generated more content
        assert len(generated_context) > len(context_with_eos)
        print(f"Generated context: {generated_context}")

        # Get the state after generation
        post_generation_state = await prefilled_state.prefill(generated_context)
        assert len(post_generation_state.states) > 0

        # Test EOS byte (257) termination after generation
        eos_terminated_state = await (post_generation_state << EOS)
        assert all(state.terminated for state in eos_terminated_state.states)

        # Verify mass distribution behavior after generation
        post_gen_trie = post_generation_state.states[0].trie.trie
        masses_gen = post_generation_state.states[0].mass
        assert not np.isnan(masses_gen[post_gen_trie.eos_node])

        # Verify EOS probability is accessible from logp_next
        logp_next = await post_generation_state.logp_next()
        eos_logp = logp_next[257]
        assert not np.isnan(eos_logp)

    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_eos_logp_next_probability_sum(llm):
    """Test that EOS probability in logp_next equals sum of specified EOS token probabilities."""

    eos_byte_strings = [b".", b"\n", b"\n\n"]
    params = BeamParams(K=5, eos_byte_strings=eos_byte_strings)
    beam = await ByteBeamState.initial(llm, params)

    try:
        first_state = beam.states[0]
        logps = await first_state.lm_state.logp_next()
        eos_token_ids = [find_token_id_by_bytes(llm.byte_vocab, t) for t in eos_byte_strings]
        logps_eos = torch.logsumexp(logps[eos_token_ids], dim=0)

        logp_next = await beam.logp_next()
        eos_logp = logp_next[EOS]

        np.testing.assert_allclose(eos_logp, logps_eos, rtol=1e-5)
    finally:
        await beam.cleanup()


@pytest.mark.asyncio
async def test_trie_state_mass_not_materialized(llm):
    """Test that accessing mass before materializing raises an error."""
    from genlm.bytes.byte_lm.trie_state import LazyTrieState
    from genlm.bytes.trie import AsyncTokenByteTrie

    eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id].byte_string
    trie = AsyncTokenByteTrie.from_vocab(llm.byte_vocab, eos_byte_strings={eos_token})

    try:
        # Create a state without materializing
        state = LazyTrieState.initial(llm, trie)

        # Accessing mass before materialize should raise
        with pytest.raises(ValueError, match="not yet materialized"):
            _ = state.mass
    finally:
        await trie.cleanup()


@pytest.mark.asyncio
async def test_trie_state_lshift_terminated(llm):
    """Test that lshift on terminated state returns None."""
    eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id].byte_string
    params = BeamParams(K=3, eos_byte_strings=[eos_token])
    beam = await ByteBeamState.initial(llm, params)

    try:
        # Prefill and get a state
        beam = await beam.prefill(b"Hello")
        state = beam.states[0]

        # Manually set terminated to True to test the branch
        state.terminated = True

        # lshift on terminated state should return None
        result = state << ord("a")
        assert result is None
    finally:
        await beam.cleanup()


def test_lm_state_max_context_length(llm):
    """Test that StatefulTokenizedLM truncates context when max_context_length is reached."""
    from genlm.bytes.byte_lm.lm_state import StatefulTokenizedLM

    # Create a state with max_context_length=3 and context already at limit
    # This tests the truncation branch
    state = StatefulTokenizedLM.initial(llm, initial_context=[1, 2, 3], max_context_length=3)
    assert len(state.context) == 3

    # Adding a token should trigger truncation: [1, 2, 3] -> [2, 3] -> [2, 3, 4]
    new_state = state << 4
    # The truncation happens on the original state before creating new one
    # New state context = truncated_context + [new_token] = [2, 3] + [4] = [2, 3, 4]
    assert len(new_state.context) == 3
    assert new_state.context == [2, 3, 4]


@pytest.mark.asyncio
async def test_logp_next_with_duplicate_eot_edges():
    """Test that logp_next correctly aggregates probabilities for duplicate EOT edges."""
    import numpy as np
    from genlm.backend.tokenization import Token
    from genlm.bytes.trie import AsyncTokenByteTrie
    from genlm.bytes.byte_lm.trie_state import LazyTrieState

    # Create vocab with duplicate byte strings (same prefix leads to multiple EOT edges)
    vocab = [
        Token(token_id=0, byte_string=b"a"),
        Token(token_id=1, byte_string=b"a"),  # Duplicate - same byte string as token 0
        Token(token_id=2, byte_string=b"b"),
    ]

    trie = AsyncTokenByteTrie.from_vocab(vocab)
    try:
        # Create mock lm_state and mass
        class MockLMState:
            async def logp_next(self):
                # Return log probs for 3 tokens as tensor
                return torch.log(torch.tensor([0.3, 0.4, 0.3]))

        lm_state = MockLMState()

        # Create LazyTrieState at root
        state = LazyTrieState(
            lm_state=lm_state,
            trie=trie,
            node=trie.trie.root,
            weight=0.0,
            mode=None,
        )

        # Materialize to get masses
        state = await state.materialize()

        # Advance to "a" node where both tokens 0 and 1 have EOT edges
        advanced_state = state << ord("a")
        assert advanced_state is not None

        # Materialize the advanced state to have masses
        advanced_state = await advanced_state.materialize()

        # Access logp_next - this should trigger the logaddexp branch
        # because both token 0 and 1 are EOT edges at this position
        logps = advanced_state.logp_next

        # The EOT probability (index 256) should be valid (not -inf)
        # indicating that both duplicate EOT edges contributed via logaddexp
        eot_logp = logps[256]
        assert eot_logp > -np.inf, "EOT logp should be valid when duplicate EOT edges exist"
        
        # Verify we're at a position with multiple EOT edges (the duplicate case)
        eot_edges = advanced_state.get_all_EOT()
        assert len(eot_edges) == 2, f"Expected 2 EOT edges for duplicates, got {len(eot_edges)}"
    finally:
        await trie.cleanup()
