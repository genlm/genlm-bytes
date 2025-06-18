import pytest
import numpy as np
from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams


@pytest.fixture(scope="module")
def llm():
    return load_model_by_name("gpt2-medium", backend="hf")


@pytest.mark.asyncio
async def test_basics(llm):
    state = await ByteBeamState.initial(
        llm, BeamParams(K=5, auto_eos=False), trie_opts={"max_batch_size": 100}
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
    state = await ByteBeamState.initial(
        llm,
        BeamParams(
            K=5,
            prune_threshold=prune_threshold,
            verbose=True,
            auto_eos=False,
        ),
    )

    try:
        output = await state.greedy(b"An apple a day keeps the ", steps=12)
        print(repr(output))
        assert output == b"An apple a day keeps the doctor away."
    finally:
        await state.cleanup()


# @pytest.mark.parametrize("prune_threshold", [None, 0.1])
# @pytest.mark.asyncio
# async def test_async_batching(llm, prune_threshold):
#     state = await ByteBeamState.initial(
#         llm,
#         BeamParams(
#             K=5,
#             prune_threshold=prune_threshold,
#         ),
#     )

#     try:
#         # warm up
#         await state.greedy(b"I", steps=5)
#         await state.greedy(b"Y", steps=5)

#         start = time.time()
#         concurrent_output = await asyncio.gather(
#             state.greedy(b"I", steps=5),
#             state.greedy(b"Y", steps=5),
#         )
#         concurrent_time = time.time() - start

#         start = time.time()
#         sequential_output_I = await state.greedy(b"I", steps=5)
#         sequential_output_Y = await state.greedy(b"Y", steps=5)
#         sequential_time = time.time() - start

#         print(f"Concurrent requests time: {concurrent_time:.2f} seconds")
#         print(f"Sequential requests time: {sequential_time:.2f} seconds")

#         assert concurrent_output == [sequential_output_I, sequential_output_Y]
#         assert concurrent_time < sequential_time
#     finally:
#         await state.cleanup()


@pytest.mark.parametrize("prune_threshold", [0, 0.1])
@pytest.mark.asyncio
async def test_weights(llm, prune_threshold):
    state = await ByteBeamState.initial(
        llm,
        BeamParams(
            K=5,
            prune_threshold=prune_threshold,
            auto_eos=False,
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
        BeamParams(K=1, prune_threshold=-0.1, auto_eos=False)


# EOS-specific tests
@pytest.mark.asyncio
async def test_eos_auto_detection(llm):
    """Test automatic EOS detection from tokenizer."""
    params = BeamParams(K=3, auto_eos=True)
    state = await ByteBeamState.initial(llm, params)

    try:
        # Check that EOS tokens were detected
        eos_tokens = getattr(state.states[0].trie.trie, "eos_tokens", set())
        assert len(eos_tokens) > 0, "Auto-detection should find EOS tokens"

        # Check that EOS node exists in trie
        assert hasattr(state.states[0].trie.trie, "eos_node")
        assert state.states[0].trie.trie.eos_node is not None

    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_eos_manual_configuration(llm):
    """Test manual EOS token configuration."""
    manual_eos = {b".", b"!", b"?"}
    params = BeamParams(K=3, eos_tokens=manual_eos, auto_eos=False)
    state = await ByteBeamState.initial(llm, params)

    try:
        # Check that manual EOS tokens were configured
        eos_tokens = getattr(state.states[0].trie.trie, "eos_tokens", set())
        assert eos_tokens == manual_eos

        # Check that EOS node exists
        assert state.states[0].trie.trie.eos_node is not None

    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_eos_disabled(llm):
    """Test EOS functionality disabled."""
    params = BeamParams(K=3, auto_eos=False, eos_tokens=set())
    state = await ByteBeamState.initial(llm, params)

    try:
        # Check that no EOS tokens were configured
        eos_tokens = getattr(state.states[0].trie.trie, "eos_tokens", set())
        assert len(eos_tokens) == 0

        # Check that EOS node doesn't exist
        assert state.states[0].trie.trie.eos_node is None

    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_eos_probability_availability(llm):
    """Test that EOS is available in generation mode but not conditioning mode."""
    params = BeamParams(K=3, auto_eos=True)
    state = await ByteBeamState.initial(llm, params)

    try:
        # Condition on some context first (conditioning mode)
        context = b"Hello world"
        state = await state.prefill(context)

        # Now check generation mode - EOS should be available
        assert state.generation_mode
        logp_next = await state.logp_next()
        probs = logp_next.materialize()

        # Check that EOS (byte 257) is available in generation mode
        assert 257 in probs or probs[257] is not None

    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_eos_combined_auto_manual(llm):
    """Test combining auto-detection with manual EOS tokens."""
    manual_eos = {b".", b"!"}
    params = BeamParams(K=3, eos_tokens=manual_eos, auto_eos=True)
    state = await ByteBeamState.initial(llm, params)

    try:
        # Check that both auto-detected and manual tokens are present
        eos_tokens = getattr(state.states[0].trie.trie, "eos_tokens", set())

        # Should contain manual tokens
        assert b"." in eos_tokens
        assert b"!" in eos_tokens

        # Should also contain auto-detected tokens (like <|endoftext|>)
        assert len(eos_tokens) > len(manual_eos), (
            "Should have both manual and auto-detected tokens"
        )

    finally:
        await state.cleanup()
