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
            target = bs[t]
            logp_next = await current.logp_next()

            # If next byte is unreachable, attempt adaptive token healing, then consume the same byte.
            if logp_next[target] == -float("inf"):
                healed = await adaptive_heal(current, target)
                if healed is None:
                    raise AssertionError(
                        f"Unreachable byte at t={t} and adaptive healing failed"
                    )
                next_beam = await (healed.prune() << target)
                if len(next_beam) == 0:
                    raise AssertionError(
                        f"Healed state still cannot consume target byte at t={t}"
                    )
                current = next_beam
                continue

            # Normal advance
            next_beam = await (current.prune() << target)

            # If beam went empty after advance, try adaptive healing and then consume the same byte.
            if len(next_beam) == 0:
                healed = await adaptive_heal(current, target)
                if healed is None:
                    raise AssertionError(
                        f"Beam empty after advance at t={t} and adaptive healing failed"
                    )
                next_beam2 = await (healed.prune() << target)
                if len(next_beam2) == 0:
                    raise AssertionError(
                        f"Healed state still cannot consume target byte at t={t}"
                    )
                current = next_beam2
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


async def adaptive_heal(current: ByteBeamState, next_byte: int, *, max_backoff: int | None = None) -> ByteBeamState | None:
    """Byte-preserving adaptive token healing (K=1).

    Retokenize the tail of the current token to a valid earlier boundary, commit that
    token, re-materialize, and replay the already consumed suffix so that the next
    byte becomes reachable — without changing any previously consumed bytes.
    """
    if len(current) == 0:
        return None

    # Work on the top state only (K=1 scenario)
    s = current.states[0]
    s = await s.materialize()  # ensure we have masses for current token state

    trie = s.trie.trie
    children = trie.children

    P = s.partial  # bytes since last token boundary (list[int])
    verbose = bool(getattr(current.params, "verbose", False))
    if verbose:
        try:
            nb_disp = repr(bytes([next_byte])) if 0 <= next_byte <= 255 else str(next_byte)
        except Exception:
            nb_disp = str(next_byte)
        print(
            f"[heal] Start: next_byte={nb_disp}, P={repr(bytes(P))}, max_backoff={max_backoff}"
        )

    # Base weight at the start of the current token (before consuming P)
    # s.mass is log-mass; contribution of P is (mass[node] - mass[root])
    base_weight = s.weight - (s.mass[s.node] - s.mass[trie.root])

    # Precompute the chain of nodes for P (from root)
    path_nodes = [trie.root]
    node = trie.root
    ok = True
    for b in P:
        nxt = children[node].get(b)
        if nxt is None:
            ok = False
            break
        path_nodes.append(nxt)
        node = nxt
    if not ok:
        # Current partial shouldn't be unreachable; fail safe.
        return None

    # Determine how far we allow backing off
    L = len(P)
    min_k = max(0, L - (max_backoff or L))

    # Try backing off within P: k = L, L-1, ..., min_k
    for k in range(L, min_k - 1, -1):
        anc_node = path_nodes[k]

        # We can only commit a token if there is an EOT from this ancestor
        eot_node = children[anc_node].get(trie.eot_token)
        if eot_node is None:
            if verbose:
                print(f"[heal] k={k}: no EOT at prefix {repr(bytes(P[:k]))}")
            continue

        # Commit token at P[:k] under the old mass
        token_id = int(trie.leaf2token_id[eot_node])
        w_after_eot = base_weight + (s.mass[eot_node] - s.mass[anc_node])

        # New state after committing token boundary
        committed = LazyTrieState(
            lm_state=(s.lm_state << token_id),
            trie=s.trie,
            node=trie.root,
            weight=w_after_eot,
            mass=None,
            mode=s.mode,
            terminated=False,
        )

        # Materialize new masses for the next token
        committed = await committed.materialize()

        if verbose:
            tok_bytes = trie.decode[token_id]
            print(
                f"[heal] k={k}: commit token={repr(tok_bytes)}, base→w={w_after_eot:.2f}; replay suffix={repr(bytes(P[k:]))}"
            )

        # Replay the suffix R = P[k:] under the new mass to reconstruct the same bytes
        node2 = trie.root
        weight2 = committed.weight
        for b in P[k:]:
            nxt = children[node2].get(b)
            if nxt is None:
                # This backoff doesn't permit reconstructing the same bytes under new tokenization
                node2 = None
                break
            weight2 = weight2 + (committed.mass[nxt] - committed.mass[node2])
            node2 = nxt

        if node2 is None:
            if verbose:
                print(f"[heal] k={k}: replay failed (unreachable suffix)")
            continue

        # Check if next_byte is reachable now
        if children[node2].get(next_byte) is None:
            # Not yet extendable; try a deeper backoff
            if verbose:
                print(f"[heal] k={k}: next_byte still unreachable; continue")
            continue

        healed_state = LazyTrieState(
            lm_state=committed.lm_state,
            trie=s.trie,
            node=node2,
            weight=weight2,
            mass=committed.mass,
            mode=s.mode,
            terminated=False,
        )

        if verbose:
            print(f"[heal] SUCCESS at k={k}: will consume {nb_disp} next; new_weight={weight2:.2f}")
        return ByteBeamState([healed_state], current.params)

    if verbose:
        print("[heal] FAILED: no valid backoff found")
    return None


def main():
    parser = argparse.ArgumentParser(description="Basic GPT-2 byte-level test with optional verbosity.")
    parser.add_argument("--verbose", action="store_true", help="Print beam state and top-K at each step.")
    parser.add_argument("--topk", type=int, default=10, help="Show top-K next bytes when verbose.")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam width K.")
    args = parser.parse_args()

    asyncio.run(run(verbose=args.verbose, topk=args.topk, beam_width=args.beam_width))


if __name__ == "__main__":
    main()
