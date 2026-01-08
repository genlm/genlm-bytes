"""Test generation with Gemma 2 2B model.

This model has duplicate tokens (multiple token IDs that decode to the same byte string),
which previously caused errors. This test verifies that the fix works correctly.

Note: These tests are skipped in CI because Gemma is a gated model requiring authentication.
Run locally with `huggingface-cli login` after accepting the license at:
https://huggingface.co/google/gemma-2-2b
"""

import pytest
import numpy as np

# Try to import and check if model is accessible
try:
    from genlm.backend import load_model_by_name
    from genlm.bytes import ByteBeamState, BeamParams
    
    # Check if we can access the gated model (will fail without auth)
    from huggingface_hub import model_info
    model_info("google/gemma-2-2b")
    GEMMA_AVAILABLE = True
except Exception:
    GEMMA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GEMMA_AVAILABLE,
    reason="Gemma model not accessible (gated model requires authentication)"
)


@pytest.fixture(scope="module")
def gemma_llm():
    """Load Gemma 2 2B model once for all tests in this module."""
    return load_model_by_name("google/gemma-2-2b", backend="hf")


@pytest.mark.asyncio
async def test_gemma_duplicate_tokens_load(gemma_llm):
    """Test that Gemma model loads successfully with duplicate tokens."""
    # Verify byte_vocab contains Token objects
    assert len(gemma_llm.byte_vocab) > 0
    
    # Check that Token objects have the expected attributes
    token = gemma_llm.byte_vocab[0]
    assert hasattr(token, "token_id")
    assert hasattr(token, "byte_string")
    assert token.token_id == 0


@pytest.mark.asyncio
async def test_gemma_beam_state_initialization(gemma_llm):
    """Test that ByteBeamState can be initialized with Gemma model."""
    eos_token = gemma_llm.byte_vocab[gemma_llm.tokenizer.eos_token_id].byte_string
    
    beam = await ByteBeamState.initial(
        gemma_llm,
        BeamParams(
            K=3,
            eos_tokens=[eos_token],
            heal=True,
        ),
    )
    
    assert beam is not None
    assert len(beam) > 0
    
    await beam.cleanup()


@pytest.mark.asyncio
async def test_gemma_prefill(gemma_llm):
    """Test prefilling with prompt works correctly."""
    eos_token = gemma_llm.byte_vocab[gemma_llm.tokenizer.eos_token_id].byte_string
    
    beam = await ByteBeamState.initial(
        gemma_llm,
        BeamParams(
            K=3,
            eos_tokens=[eos_token],
            heal=True,
        ),
    )
    
    prompt = b"Hello"
    beam = await beam.prefill(prompt)
    
    assert beam is not None
    assert len(beam) > 0
    
    await beam.cleanup()


@pytest.mark.asyncio
async def test_gemma_generation(gemma_llm):
    """Test that generation works with Gemma model (duplicate tokens)."""
    eos_token = gemma_llm.byte_vocab[gemma_llm.tokenizer.eos_token_id].byte_string
    
    beam = await ByteBeamState.initial(
        gemma_llm,
        BeamParams(
            K=3,
            eos_tokens=[eos_token],
            heal=True,
        ),
    )
    
    # Prefill with prompt
    prompt = b"The capital of France is"
    beam = await beam.prefill(prompt)
    
    # Generate a few bytes
    generated = bytearray()
    for _ in range(10):
        beam = beam.prune()
        
        if len(beam) == 0:
            break
        if all(s.terminated for s in beam.states):
            break
        
        # Get next byte distribution
        logp = await beam.logp_next()
        probs = np.exp(logp.ps)
        probs = probs / probs.sum()
        
        # Take argmax for deterministic output
        next_byte = int(np.argmax(probs))
        
        # Check for EOS
        if next_byte == 257:
            break
        
        generated.append(next_byte)
        beam = await (beam << next_byte)
    
    await beam.cleanup()
    
    # Should have generated something
    assert len(generated) > 0
    # Should be valid UTF-8 (or at least decodable)
    text = generated.decode("utf-8", errors="replace")
    assert isinstance(text, str)
