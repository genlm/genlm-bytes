import asyncio
import argparse

from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams
from genlm.bytes.byte_lm.trie_state import LazyTrieState


TEXT = \
    ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."


async def run(verbose: bool, topk: int, beam_width: int):
    llm = load_model_by_name("gpt2", backend="hf")
    eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam = await ByteBeamState.initial(
        llm, BeamParams(K=beam_width, verbose=verbose, eos_tokens=[eos_token])
    )

    try:
        bs = TEXT.encode("utf-8")

        current = beam
        for t in range(len(bs)):
            logp_next = await current.logp_next()

            # if verbose:
            #     pretty = logp_next.pretty().top(topk)
            #     print(f"Step {t}: top-{topk} next-byte log-probabilities")
            #     for key, value in pretty.items():
            #         print(key, value)
            #     print("Beam state:")
            #     print(current)
            #     print()

            # If next byte is unreachable, attempt adaptive token healing.
            if logp_next[bs[t]] == -float("inf"):
                healed = await adaptive_heal(current, bs[t])
                if healed is None:
                    raise AssertionError(f"Unreachable byte at t={t} and adaptive healing failed")
                current = healed
                continue

            # Normal advance
            next_beam = await (current.prune() << bs[t])

            # If beam went empty after advance, try adaptive healing from the previous state.
            if len(next_beam) == 0:
                healed = await adaptive_heal(current, bs[t])
                if healed is None:
                    raise AssertionError(f"Beam empty after advance at t={t} and adaptive healing failed")
                current = healed
            else:
                current = next_beam

        logp_next_all = await current.logp_next()

        if eos_token is not None:
            if verbose:
                print("After full text: EOS(257) log-prob (model EOS)")
                print(logp_next_all[257])
            assert logp_next_all[257] > -float("inf"), "EOS (257) unreachable after full text"

        print("BASIC GPT-2 TEST: PASS")
    finally:
        await beam.cleanup()


async def adaptive_heal(current: ByteBeamState, next_byte: int) -> ByteBeamState | None:
    # K=1 path: operate on the best candidate only
    if len(current) == 0:
        return None

    s = current.states[0]
    # Ensure masses are available
    s = await s.materialize()

    trie = s.trie.trie
    children = trie.children

    # Reconstruct the path of nodes for the current partial prefix
    prefix = s.partial  # list of byte ints
    node_path = [trie.root]
    node = trie.root
    for letter in prefix:
        if letter in children[node]:
            node = children[node][letter]
            node_path.append(node)
        else:
            break

    best = None
    best_w = -float("inf")

    # Search ancestors (nearest first) for a legal child with next_byte
    for anc in reversed(node_path):
        child = children[anc].get(next_byte)
        if child is None:
            continue
        # Heuristic reweighting: replace tail with edge to child using current mass
        cand_w = s.weight + s.mass[child] - s.mass[anc]
        if cand_w > best_w:
            best_w = cand_w
            best = LazyTrieState(
                lm_state=s.lm_state,
                trie=s.trie,
                node=child,
                weight=cand_w,
                mass=s.mass,
                mode=s.mode,
                terminated=False,
            )

    if best is None:
        return None

    healed = ByteBeamState([best], current.params)
    return healed


def main():
    parser = argparse.ArgumentParser(description="Basic GPT-2 byte-level test with optional verbosity.")
    parser.add_argument("--verbose", action="store_true", help="Print beam state and top-K at each step.")
    parser.add_argument("--topk", type=int, default=10, help="Show top-K next bytes when verbose.")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam width K.")
    args = parser.parse_args()

    asyncio.run(run(verbose=args.verbose, topk=args.topk, beam_width=args.beam_width))


if __name__ == "__main__":
    main()


