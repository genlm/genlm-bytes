import asyncio
from types import SimpleNamespace

import pytest

from genlm.bytes import StatefulByteLM, advance_byte_states


class DummyState(StatefulByteLM):
    """Minimal concrete implementation for batching tests."""

    def __init__(self, history):
        self.history = list(history)

    def prune(self):
        return DummyState(self.history + ["pruned"])

    async def __lshift__(self, b: int):
        await asyncio.sleep(0)
        return DummyState(self.history + [f"byte:{b}"])

    async def logp_next(self):  # pragma: no cover - not used in tests
        return SimpleNamespace(materialize=lambda: None)


@pytest.mark.asyncio
async def test_advance_byte_states_batches_prune_and_shift():
    states = [DummyState(["s0"]), DummyState(["s1"])]
    next_bytes = [97, 98]

    advanced = await advance_byte_states(states, next_bytes)

    assert [s.history for s in advanced] == [
        ["s0", "pruned", "byte:97"],
        ["s1", "pruned", "byte:98"],
    ]

    # Original states are unchanged
    assert [s.history for s in states] == [["s0"], ["s1"]]


@pytest.mark.asyncio
async def test_advance_byte_states_length_mismatch():
    with pytest.raises(ValueError):
        await advance_byte_states([DummyState(["s0"])], [])


@pytest.mark.asyncio
async def test_advance_byte_states_empty_inputs():
    assert await advance_byte_states([], []) == []

