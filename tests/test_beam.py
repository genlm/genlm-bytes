import torch
import pytest
import numpy as np
from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams
from genlm.bytes.trie import EOS


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


# EOS-specific tests
@pytest.mark.asyncio
async def test_eos_manual_configuration(llm):
    """Test manual EOS token configuration."""
    manual_eos = [b".", b"!", b"?"]
    params = BeamParams(K=3, eos_tokens=manual_eos)
    state = await ByteBeamState.initial(llm, params)

    try:
        for state in state.states:
            assert state.trie.trie.eos_tokens == set(manual_eos)
            assert state.trie.trie.eos_node is not None

    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_eos_disabled(llm):
    """Test EOS functionality disabled."""
    params = BeamParams(K=3, eos_tokens=set())  # Empty set = no EOS
    state = await ByteBeamState.initial(llm, params)

    try:
        # Check that no EOS tokens were configured
        assert not any(state.trie.trie.eos_tokens for state in state.states)

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
    params = BeamParams(K=3, eos_tokens=[b"!"])
    state = await ByteBeamState.initial(llm, params)

    try:
        new_state = await (state << EOS)
        assert all(state.terminated for state in new_state.states)

        eos_token_id = llm.byte_vocab.index(b"!")
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
    params = BeamParams(K=10, eos_tokens=[b"\n", b"\n\n"])
    state = await ByteBeamState.initial(llm, params)

    try:
        for trie_state in state.states:
            trie = trie_state.trie.trie
            assert b"\n" in trie.eos_tokens
            assert b"\n\n" in trie.eos_tokens

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

    eos_tokens = [b".", b"\n", b"\n\n"]
    params = BeamParams(K=5, eos_tokens=eos_tokens)
    beam = await ByteBeamState.initial(llm, params)

    try:
        first_state = beam.states[0]
        logps = await first_state.lm_state.logp_next()
        eos_token_ids = [llm.byte_vocab.index(t) for t in eos_tokens]
        logps_eos = torch.logsumexp(logps[eos_token_ids], dim=0)

        logp_next = await beam.logp_next()
        eos_logp = logp_next[EOS]

        np.testing.assert_allclose(eos_logp, logps_eos, rtol=1e-5)
    finally:
        await beam.cleanup()
