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
    manual_eos = {b".", b"!", b"?"}
    params = BeamParams(K=3, eos_tokens=manual_eos)
    state = await ByteBeamState.initial(llm, params)

    try:
        # Check that manual EOS tokens were configured
        eos_tokens = getattr(state.states[0].trie.trie, "eos_tokens", set())
        assert eos_tokens == manual_eos

        # Check that EOS node exists
        assert hasattr(state.states[0].trie.trie, "eos_node")
        assert state.states[0].trie.trie.eos_node is not None

    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_eos_disabled(llm):
    """Test EOS functionality disabled."""
    params = BeamParams(K=3, eos_tokens=set())  # Empty set = no EOS
    state = await ByteBeamState.initial(llm, params)

    try:
        # Check that no EOS tokens were configured
        eos_tokens = getattr(state.states[0].trie.trie, "eos_tokens", set())
        assert len(eos_tokens) == 0

        # check that EOS isn't available
        logp_next = await state.logp_next()
        probs = logp_next.materialize()
        # EOS (257) should not be available or have -inf probability
        if 257 in probs:
            assert probs[257] == -np.inf or np.isneginf(probs[257])

    finally:
        await state.cleanup()



@pytest.mark.asyncio
async def test_eos_termination(llm):
    """Test that EOS byte terminates sequences properly."""
    params = BeamParams(K=3, eos_tokens={b"!"})
    state = await ByteBeamState.initial(llm, params)

    try:
        # Test EOS termination
        initial_count = len(state.states)

        # Try to advance with EOS byte (257) - should terminate sequences
        from genlm.bytes.trie import EOS

        new_state = await (state << EOS)

        # States that consumed EOS should be terminated (removed)
        # Remaining states should be fewer or equal
        assert len(new_state.states) <= initial_count

        print(f"Initial states: {initial_count}, After EOS: {len(new_state.states)}")

    finally:
        await state.cleanup()




@pytest.mark.asyncio
async def test_eos_token(llm):
    """Test using the model's actual EOS token from tokenizer."""

    # Get the actual EOS token from the model's tokenizer and encode as bytes
    model_eos_str = llm.tokenizer.decode([llm.tokenizer.eos_token_id])
    model_eos_token = model_eos_str.encode("utf-8")  # Convert to bytes

    params = BeamParams(K=10, eos_tokens={model_eos_token})
    state = await ByteBeamState.initial(llm, params)

    try:
        # Test 1: Verify EOS token configuration
        trie = state.states[0].trie.trie
        assert model_eos_token in trie.eos_tokens
        assert hasattr(trie, "eos_node") and trie.eos_node is not None

        # Test 2: Test prefill with model EOS token (conditioning mode)
        context_with_eos = b"Hello world" + model_eos_token + b" This continues."
        prefilled_state = await state.prefill(context_with_eos)
        assert prefilled_state.generation_mode
        assert len(prefilled_state.states) > 0

        # Test 3: Test greedy generation for 10 steps after prefill
        generated_context = await prefilled_state.greedy(context_with_eos, 10)
        assert len(generated_context) > len(
            context_with_eos
        )  # Should have generated more content
        print(f"Generated context: {generated_context}")

        # Get the state after generation
        post_generation_state = await state.prefill(generated_context)
        assert len(post_generation_state.states) > 0

        # Test 4: Test EOS byte (257) termination after generation
        initial_count = len(post_generation_state.states)
        eos_terminated_state = await (post_generation_state << EOS)
        final_count = len(eos_terminated_state.states)
        assert (
            final_count <= initial_count
        )  # EOS should terminate or maintain state count

        # Test 5: Find model EOS token in vocabulary
        vocab = trie.decode
        model_eos_token_id = None
        for i, token in enumerate(vocab):
            if token == model_eos_token:
                model_eos_token_id = i
                break
        assert model_eos_token_id is not None, (
            f"Model EOS token {model_eos_token} not found in vocabulary"
        )

        # Test 6: Verify mass distribution behavior after generation
        post_gen_trie = post_generation_state.states[0].trie.trie
        masses_gen = post_generation_state.states[0].mass
        assert hasattr(post_gen_trie, "eos_node")
        assert not np.isnan(
            masses_gen[post_gen_trie.eos_node]
        )  # Mass should be a valid number

        # Test 7: Verify EOS probability is accessible from logp_next
        logp_next = await post_generation_state.logp_next()
        eos_logp = logp_next[257]
        assert not np.isnan(eos_logp)  # EOS log probability should be valid

    finally:
        await state.cleanup()

@pytest.mark.asyncio 
async def test_eos_logp_next_probability_sum(llm):
    """Test that EOS probability in logp_next equals sum of specified EOS token probabilities."""
    
    eos_tokens = {b".", b"!", b"?"}
    params = BeamParams(K=5, eos_tokens=eos_tokens)
    
    beam = await ByteBeamState.initial(llm, params)
    
    try:
        # check we're in generation mode and at root
        assert beam.generation_mode
        assert len(beam.states) > 0
        first_state = beam.states[0]
        assert first_state.node == first_state.trie.trie.root, f"Beam should start at root {first_state.trie.trie.root}, got {first_state.node}"
        
        # Get the underlying token probabilities from the language model
        token_logprobs = await first_state.lm_state.logp_next()
        token_probs = np.exp(token_logprobs.cpu().numpy()).astype(np.float64)
        
        # Find the token IDs for our EOS tokens
        decode_vocab = first_state.trie.trie.decode
        eos_token_ids = []
        expected_eos_prob_sum = 0.0
        
        for token_id, token_bytes in enumerate(decode_vocab):
            if token_bytes in eos_tokens:
                eos_token_ids.append(token_id)
                expected_eos_prob_sum += float(token_probs[token_id])
        
        print(f"EOS tokens: {eos_tokens}")
        print(f"EOS token IDs found: {eos_token_ids}")
        print(f"Expected EOS probability sum: {expected_eos_prob_sum}")
        
        logp_next = await beam.logp_next()
        eos_logp = logp_next[EOS]
        eos_prob = float(np.exp(eos_logp))  # Ensure float64
        
        print(f"Actual EOS probability from logp_next: {eos_prob}")
        print(f"Actual EOS log probability: {eos_logp}")
        
        assert not np.isnan(eos_prob), "EOS probability should not be NaN"
        assert not np.isinf(eos_logp), "EOS log probability should not be -inf"
        
        np.testing.assert_allclose(
            eos_prob, 
            expected_eos_prob_sum, 
            rtol=1e-4,
            err_msg=f"EOS probability {eos_prob} should equal sum of EOS token probabilities {expected_eos_prob_sum}"
        )
                
    finally:
        await beam.cleanup()
