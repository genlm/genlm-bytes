import asyncio
import argparse

from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams


TEXT = \
    ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."


async def run_once(text: str, heal: bool, verbose: bool, max_backoff: int | None):
    print("=" * 80)
    print(f"RUN heal={heal}")
    print("=" * 80)

    llm = load_model_by_name("gpt2", backend="hf")
    eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]

    beam = await ByteBeamState.initial(
        llm,
        BeamParams(
            K=1,
            verbose=verbose,
            eos_tokens=[eos_token],
            heal=heal,
            heal_max_backoff=max_backoff,
        ),
    )

    try:
        bs = text.encode("utf-8")
        current = beam
        for t, b in enumerate(bs):
            next_beam = await (current.prune() << b)
            if len(next_beam) == 0:
                print(f"[result] Empty beam at byte index {t}, b={repr(bytes([b]))}")
                return
            current = next_beam

        # Completed full prefix; check EOS reachability
        logp_next_all = await current.logp_next()
        eos_logp = logp_next_all[257]
        print(f"[result] Completed all bytes; EOS logp: {eos_logp}")
    finally:
        await beam.cleanup()


async def run(text: str, verbose: bool, max_backoff: int | None):
    # First run with healing disabled
    await run_once(text, heal=False, verbose=verbose, max_backoff=max_backoff)
    # Then run with healing enabled
    await run_once(text, heal=True, verbose=verbose, max_backoff=max_backoff)


def main():
    parser = argparse.ArgumentParser(description="Adaptive token healing demo (K=1)")
    parser.add_argument("--text", type=str, default=TEXT, help="Input text to prefill")
    parser.add_argument("--verbose", action="store_true", help="Verbose beam/heal logs")
    parser.add_argument(
        "--max_backoff",
        type=int,
        default=None,
        help="Max healing backoff within current token (None = unlimited)",
    )
    args = parser.parse_args()

    asyncio.run(run(text=args.text, verbose=args.verbose, max_backoff=args.max_backoff))


if __name__ == "__main__":
    main()

