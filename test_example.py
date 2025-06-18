#!/usr/bin/env python3
"""
Simple test case for genlm-bytes with beam search.
This demonstrates byte-level language modeling with:
- Beam size (K) = 5
- Prune threshold = 0.05
"""

import asyncio
from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams


async def main():
    print("🚀 Loading language model...")
    # Load a small language model for faster inference
    llm = load_model_by_name("gpt2", backend="hf")  # Using smaller GPT-2 model
    
    print("🔧 Setting up beam search parameters...")
    # Define EOS tokens (tokens that should terminate generation)
    eos_tokens = {b"\n\n", b"<|endoftext|>"}  # Double newline and end-of-text
    
    # Create beam parameters with K=5 (beam width) and prune_threshold=0.05
    params = BeamParams(
        K=5,                    # Keep top 5 candidates
        prune_threshold=0.05,   # Remove candidates below 5% probability
        verbose=True,           # Print beam state at each step
        eos_tokens=eos_tokens,  # Tokens that cause termination
        terminate_on_eos=True   # Terminate when EOS is sampled
    )

    beam = await ByteBeamState.initial(llm, params)
    
    try:
        print("\n📝 Starting generation examples...")
        print("=" * 60)
        
        # Example 1: Greedy generation
        print("\n🎯 Example 1: Greedy generation")
        print("Starting context: 'The quick brown '")
        result1 = await beam.greedy(b"The quick brown ", steps=15)
        print(f"Result: {result1!r}")
        
        # Example 2: EOS demonstration
        print("\n🛑 Example 2: EOS demonstration")
        print("Context with EOS token: 'Hello world\\n\\nThis continues'")
        # This should work during conditioning but terminate during generation
        beam_eos = await ByteBeamState.initial(llm, params)
        beam_eos = await beam_eos.prefill(b"Hello world\n\nThis continues")
        print("✅ Prefill with EOS token succeeded (conditioning mode)")
        
        # Now try to generate - if we sample EOS (byte 257), it should terminate
        print("🎲 Generating from EOS context...")
        result2 = await beam_eos.sample(b"", steps=10)
        print(f"Result: {result2!r}")
        
    finally:
        print("\n🧹 Cleaning up...")
        await beam.cleanup()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    import numpy as np
    asyncio.run(main()) 