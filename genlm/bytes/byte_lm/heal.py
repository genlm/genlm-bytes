from .trie_state import LazyTrieState
from ..util import format_byte


def _find_all_eot_edges(children, eot_sentinel):
    """Find all EOT edges in children dict. Returns list of (node, token_id) tuples.

    EOT edges are stored as tuple keys: (eot_sentinel, token_id).
    With duplicate tokens, multiple token IDs can map to the same byte string.
    """
    results = []
    for key, node in children.items():
        if isinstance(key, tuple) and key[0] == eot_sentinel:
            results.append((node, key[1]))
    return results


class TokenHealer:
    """Handles adaptive token healing for ByteBeamState.
    Token healing finds alternative tokenizations when the current tokenization
    cannot consume the next byte. It works by:
    1. Trying different "backoff" positions k (commit partial[:k] as a token)
    2. Replaying the remaining bytes (partial[k:]) from fresh root
    3. Using extend_all() when stuck to commit intermediate tokens
    4. Finally consuming the target next_byte

    Args:
        max_backoff: Maximum bytes to back off (None = unlimited)
        max_splits: Maximum intra-suffix commits allowed (None = unlimited)
        verbose: Whether to print debug information
    """

    def __init__(
        self,
        max_backoff: int | None = None,
        max_splits: int | None = None,
        verbose: bool = False,
    ):
        self.max_backoff = max_backoff
        self.max_splits = max_splits
        self.verbose = verbose

    async def try_heal(self, state, next_byte: int):
        """Try to heal a state so it can consume next_byte.

        Args:
            state: A materialized LazyTrieState that cannot consume next_byte
            next_byte: The byte we want to consume

        Returns:
            LazyTrieState if healing succeeds, None otherwise
        """
        partial = state.partial
        partial_len = len(partial)

        if self.verbose:
            print(
                f"[heal] Start: next_byte={format_byte(next_byte)}, partial={bytes(partial)!r}, max_backoff={self.max_backoff}"
            )

        # Extract invariants computed once for all k values
        trie = state.trie.trie
        # base_weight undoes current path contribution: weight + mass[root] - mass[node]
        # NOTE: mass[root] terms cancel, written this way to show undo current path contribution, add commit path
        base_weight = state.weight - (state.mass[state.node] - state.mass[trie.root])

        # Calculate how far back we're allowed to go
        min_k = (
            0 if self.max_backoff is None else max(0, partial_len - self.max_backoff)
        )

        # Try each backoff position k (from longest prefix to shortest)
        for k in range(partial_len, min_k - 1, -1):
            result = await self._try_at_k(state, trie, base_weight, k, next_byte)
            if result is not None:
                return result

        if self.verbose:
            print("[heal] FAILED: no valid healing found")
        return None

    async def _try_at_k(self, state, trie, base_weight: float, k: int, next_byte: int):
        """Try healing by committing partial[:k], replaying partial[k:], then consuming next_byte.

        With duplicate tokens, there can be multiple EOT edges at position k.
        This method tries all of them until one succeeds.

        Args:
            state: The original state to heal from
            trie: The trie structure (state.trie.trie)
            base_weight: Precomputed weight after undoing current path
            k: Backoff position to try
            next_byte: The byte we want to consume

        Returns:
            LazyTrieState if successful, None otherwise
        """
        children = trie.children
        partial = state.partial

        # Navigate to position k to check if we can commit there
        node_at_k = trie.root
        for b in partial[:k]:
            node_at_k = children[node_at_k].get(b)
            if node_at_k is None:
                return None  # Path doesn't exist

        # Find all EOT edges at position k
        # With duplicate tokens, multiple token IDs can map to the same byte string
        eot_edges = _find_all_eot_edges(children[node_at_k], trie.eot_sentinel)
        if not eot_edges:
            if self.verbose:
                print(f"[heal] k={k}: no EOT at {bytes(partial[:k])!r}")
            return None

        # Try each possible EOT edge
        for eot_node, eot_token_id in eot_edges:
            result = await self._try_eot_at_k(
                state, trie, base_weight, k, next_byte, eot_node, eot_token_id
            )
            if result is not None:
                return result

        return None

    async def _try_eot_at_k(
        self, state, trie, base_weight: float, k: int, next_byte: int,
        eot_node: int, eot_token_id: int
    ):
        """Try healing with a specific EOT edge at position k.

        Args:
            state: The original state to heal from
            trie: The trie structure (state.trie.trie)
            base_weight: Precomputed weight after undoing current path
            k: Backoff position
            next_byte: The byte we want to consume
            eot_node: The EOT node to commit
            eot_token_id: The token ID for this EOT

        Returns:
            LazyTrieState if successful, None otherwise
        """
        partial = state.partial

        # Commit at position k with this specific token
        weight_after_commit = base_weight + (
            state.mass[eot_node] - state.mass[trie.root]
        )
        token_id = int(eot_token_id)

        current = LazyTrieState(
            lm_state=(state.lm_state << token_id),
            trie=state.trie,
            node=trie.root,
            weight=weight_after_commit,
            mass=None,
            mode=state.mode,
            terminated=False,
        )
        current = await current.materialize()

        if self.verbose:
            # trie.decode contains Token objects, get byte_string for display
            token_bytes = trie.decode[token_id].byte_string
            print(
                f"[heal] k={k}: commit {token_bytes!r} (token_id={token_id}), w={weight_after_commit:.2f}"
            )

        # Replay suffix bytes then consume next_byte
        all_bytes = list(partial[k:]) + [next_byte]
        splits_used = 0

        for b in all_bytes:
            next_state = current << b
            if next_state is not None:
                current = next_state
                continue

            # Can't consume this byte - try extend (commit current partial) first
            if self.max_splits is not None and splits_used >= self.max_splits:
                if self.verbose:
                    print(f"[heal] k={k}: hit max_splits={self.max_splits}")
                return None

            # extend_all() returns list of all possible extensions
            extensions = current.extend_all()
            if not extensions:
                if self.verbose:
                    print(f"[heal] k={k}: can't extend at {bytes(current.partial)!r}")
                return None

            # Try each possible extension (duplicates = same split, different token_id)
            splits_used += 1
            for extended in extensions:
                materialized = await extended.materialize()
                if self.verbose:
                    print(f"[heal] k={k}: split #{splits_used}, w={materialized.weight:.2f}")

                # Retry consuming the byte after extend
                next_state = materialized << b
                if next_state is not None:
                    current = next_state
                    break  # Found a valid extension
            else:
                # None of the extensions worked
                if self.verbose:
                    print(
                        f"[heal] k={k}: couldn't consume {format_byte(b)} even after extend"
                    )
                return None

        if self.verbose:
            print(f"[heal] SUCCESS at k={k}: w={current.weight:.2f}")
        return current
