"""Test generation with Gemma 2 2B model.

This model has duplicate tokens (multiple token IDs that decode to the same byte string),
which previously caused errors. This test verifies that the fix works correctly.
"""

import asyncio
import numpy as np
from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams


async def generate(llm, prompt: str, max_bytes: int = 100, beam_width: int = 5):
    """Generate text byte-by-byte using beam search.
    
    Args:
        llm: Language model from genlm-backend
        prompt: Text prompt to condition on
        max_bytes: Maximum bytes to generate
        beam_width: Beam width for search
        
    Returns:
        Generated text as string
    """
    # Get EOS token bytes
    eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id].byte_string
    
    # Initialize beam state
    beam = await ByteBeamState.initial(
        llm,
        BeamParams(
            K=beam_width,
            eos_tokens=[eos_token],
            heal=True,
            verbose=True,
        ),
    )
    
    # Prefill with prompt
    prompt_bytes = prompt.encode("utf-8")
    beam = await beam.prefill(prompt_bytes)
    
    # Generate bytes
    generated = bytearray()
    for _ in range(max_bytes):
        beam = beam.prune()
        
        # Check if beam is empty or all states terminated
        if len(beam) == 0:
            break
        if all(s.terminated for s in beam.states):
            break
            
        # Sample next byte from distribution
        logp = await beam.logp_next()
        probs = np.exp(logp.ps)
        probs = probs / probs.sum()  # Normalize
        
        # Sample (or take argmax for deterministic output)
        next_byte = int(np.argmax(probs))
        
        # Check for EOS (index 257)
        if next_byte == 257:
            break
            
        generated.append(next_byte)
        beam = await (beam << next_byte)
    
    await beam.cleanup()
    return generated.decode("utf-8", errors="replace")


async def main():
    print("Loading Gemma 2 2B model...")
    llm = load_model_by_name("google/gemma-2-2b", backend="hf")
    
    prompt = "The capital of France is"
    print(f"\nPrompt: {prompt!r}")
    print("Generating...")
    
    output = await generate(llm, prompt, max_bytes=50, beam_width=3)
    print(f"\nGenerated: {output!r}")
    print(f"\nFull text: {prompt}{output}")


if __name__ == "__main__":
    asyncio.run(main())

