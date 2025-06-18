import asyncio
from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams


async def main():
    print("Loading language model...")
    llm = load_model_by_name("gpt2", backend="hf")
    
    print("Setting up beam search parameters...")
    # Define EOS tokens (tokens that should terminate generation)
    eos_tokens = {b"\n\n", b"<|endoftext|>"}
    
    params = BeamParams(
        K=5,                    
        prune_threshold=0.05,   
        verbose=True,           
        eos_tokens=eos_tokens,  
        terminate_on_eos=True 
    )

    beam = await ByteBeamState.initial(llm, params)
    
    try:
        print("\n Starting generation examples...")
        print("=" * 60)
        
        # Example 1: Greedy generation
        print("\n Example 1: Greedy generation")
        print("Starting context: 'The quick brown '")
        result1 = await beam.greedy(b"The quick brown ", steps=15)
        print(f"Result: {result1!r}")
        
        # Example 2: EOS demonstration
        print("\n Example 2: EOS demonstration")
        print("Context with EOS token: 'Hello world\\n\\nThis continues'")
        # This should work during conditioning but terminate during generation
        beam_eos = await ByteBeamState.initial(llm, params)
        beam_eos = await beam_eos.prefill(b"Hello world\n\nThis continues")
        print("Prefill with EOS token succeeded (conditioning mode)")
        
        # Now try to generate, if we sample EOS (byte 257), it should terminate
        print(" Generating from EOS context...")
        result2 = await beam_eos.sample(b"", steps=10)
        print(f"Result: {result2!r}")
        
    finally:
        await beam.cleanup()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    import numpy as np
    asyncio.run(main()) 
