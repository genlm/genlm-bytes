import asyncio

import numpy as np

from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams


TEXT = \
    ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."


async def run_case(keep_all_canonical: bool):
    print("\n==== canonical forcing:", keep_all_canonical, "====")
    llm = load_model_by_name("gpt2", backend="hf")

    eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    params = BeamParams(
        K=1,
        eos_tokens=[eos_token],
        heal=False,
        keep_all_canonical=keep_all_canonical,
        verbose=True,
    )

    beam = await ByteBeamState.initial(llm, params)

    try:
        bs = TEXT.encode("utf-8")
        current = beam
        for idx, b in enumerate(bs):
            print(f"\n[step {idx}] consume byte {b} ({repr(bytes([b]))})")
            next_beam = await (current.prune() << b)
            if len(next_beam) == 0:
                print(f"Beam became empty at index {idx}; canonical={keep_all_canonical}")
                return
            current = next_beam

        print("\nCompleted full prefix. Checking EOS reachability...")
        logp_next_all = await current.logp_next()
        print("EOS logp:", logp_next_all[257])
        print("EOS reachable:", logp_next_all[257] > -np.inf)
    finally:
        await beam.cleanup()


async def main():
    await run_case(False)
    await run_case(True)


if __name__ == "__main__":
    asyncio.run(main())

