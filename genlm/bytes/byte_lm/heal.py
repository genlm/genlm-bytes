from .trie_state import LazyTrieState


class TokenHealer:
    """Handles adaptive token healing for ByteBeamState.
    Token healing finds alternative tokenizations when the current tokenization
    cannot consume the next byte. It works by:
    1. Trying different "backoff" positions k (commit partial[:k] as a token)
    2. Replaying the remaining bytes (partial[k:]) from fresh root
    3. Using extend() when stuck to commit intermediate tokens
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

    def _log(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[heal] {msg}")

    def _format_byte(self, byte_val: int) -> str:
        """Format byte value for logging."""
        try:
            return repr(bytes([byte_val])) if 0 <= byte_val <= 255 else str(byte_val)
        except Exception:
            return str(byte_val)

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

        self._log(
            f"Start: next_byte={self._format_byte(next_byte)}, "
            f"partial={repr(bytes(partial))}, max_backoff={self.max_backoff}"
        )

        # Calculate how far back we're allowed to go
        min_k = (
            0 if self.max_backoff is None else max(0, partial_len - self.max_backoff)
        )

        # Try each backoff position k (from longest prefix to shortest)
        for k in range(partial_len, min_k - 1, -1):
            result = await self._try_at_k(state, k, next_byte)
            if result is not None:
                return result

        self._log("FAILED: no valid healing found")
        return None

    async def _try_at_k(self, state, k: int, next_byte: int):
        """Try healing by committing partial[:k], replaying partial[k:], then consuming next_byte.

        Returns LazyTrieState if successful, None otherwise.
        """
        trie = state.trie.trie
        children = trie.children
        partial = state.partial

        # Navigate to position k to check if we can commit there
        node_at_k = trie.root
        for b in partial[:k]:
            node_at_k = children[node_at_k].get(b)
            if node_at_k is None:
                return None  # Path doesn't exist

        # Check if there's an EOT at position k
        eot_node = children[node_at_k].get(trie.eot_token)
        if eot_node is None:
            self._log(f"k={k}: no EOT at {repr(bytes(partial[:k]))}")
            return None

        # Commit at position k
        # NOTE: mass[root] terms cancel; equivalent to: weight + mass[eot] - mass[node]
        # Written this way to show: undo current path contribution, add commit path
        base_weight = state.weight - (state.mass[state.node] - state.mass[trie.root])
        weight_after_commit = base_weight + (
            state.mass[eot_node] - state.mass[trie.root]
        )
        token_id = int(trie.leaf2token_id[eot_node])

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

        self._log(
            f"k={k}: commit {repr(trie.decode[token_id])}, w={weight_after_commit:.2f}"
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
                self._log(f"k={k}: hit max_splits={self.max_splits}")
                return None

            extended = current.extend()
            if extended is None:
                self._log(f"k={k}: can't extend at {repr(bytes(current.partial))}")
                return None

            current = await extended.materialize()
            splits_used += 1
            self._log(f"k={k}: split #{splits_used}, w={current.weight:.2f}")

            # Retry consuming the byte after extend
            next_state = current << b
            if next_state is None:
                self._log(
                    f"k={k}: couldn't consume {self._format_byte(b)} even after extend"
                )
                return None
            current = next_state

        self._log(f"SUCCESS at k={k}: w={current.weight:.2f}")
        return current
