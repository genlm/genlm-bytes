import torch
import pytest
import asyncio
import numpy as np
from transformers import AutoTokenizer

from genlm.backend.llm import MockAsyncLM
from genlm.bytes import TokenByteTrie, AsyncTokenByteTrie
from genlm.bytes.byte_lm.trie_state import TrieMode

from hypothesis import given, strategies as st


@pytest.fixture()
def decode():
    return [b"a", b"b", b"ab", b"<eos>"]


@pytest.fixture(scope="module")
def mock_llm():
    return MockAsyncLM(AutoTokenizer.from_pretrained("gpt2"))


@st.composite
def tokens_and_weights(draw, n_weights):
    vocab = draw(
        st.lists(
            st.binary(min_size=1, max_size=5), min_size=1, max_size=10, unique=True
        )
    )

    # Ensure we have at least two tokens with a shared prefix.
    for token in vocab:
        if len(token) > 1:
            new_token = token[:-1]
            if new_token not in vocab:
                vocab.append(new_token)
                break

    weights = []
    for _ in range(n_weights):
        weights.append(
            draw(
                st.lists(
                    st.floats(min_value=0.0, max_value=10.0),
                    min_size=len(vocab),
                    max_size=len(vocab),
                )
            )
        )

    return vocab, weights


def make_wants(trie, weights, op, f):
    assert len(weights) == len(trie.decode)

    leaf_wants = {}
    for token, weight in zip(trie.decode, weights):
        assert token in trie.word2leaf
        leaf_wants[token] = weight

    internal_wants = {}
    for token, weight in zip(trie.decode, weights):
        for i in range(len(token) + 1):
            prefix = token[:i]
            if prefix not in internal_wants:
                internal_wants[f(prefix)] = weight
            else:
                internal_wants[f(prefix)] = op(internal_wants[f(prefix)], weight)

    return leaf_wants, internal_wants


def assert_weights_close(trie, leaf_wants, internal_wants, haves, f):
    assert len(haves) == len(trie.children)

    haves = haves.cpu().numpy()

    for node, prefix in trie.node2prefix.items():
        if node in trie.leaf2word:
            continue
        have = haves[node]
        if prefix == [257]:  # EOS node
            continue
        want = internal_wants[f(prefix)]
        assert np.isclose(have, want, rtol=1e-5, atol=1e-8), [have, want, prefix]

    for word in trie.decode:
        assert word in trie.word2leaf
        node = trie.word2leaf[word]
        have = haves[node]
        want = leaf_wants[word]
        assert np.isclose(have, want, rtol=1e-5, atol=1e-8), [have, want, word]


def test_weight_sum_single(decode):
    trie = TokenByteTrie(decode=decode)
    haves = trie.weight_sum(torch.tensor([0.1, 0.2, 0.2, 0.5]))

    leaf_wants = {
        b"a": 0.1,
        b"b": 0.2,
        b"ab": 0.2,
        b"<eos>": 0.5,
    }
    internal_wants = {
        b"": 1,
        b"a": 0.3,
        b"b": 0.2,
        b"ab": 0.2,
        b"<": 0.5,
        b"<e": 0.5,
        b"<eo": 0.5,
        b"<eos": 0.5,
        b"<eos>": 0.5,
    }

    assert_weights_close(trie, leaf_wants, internal_wants, haves, bytes)


def test_weight_sum_single_atomic(decode):
    trie = TokenByteTrie(decode=decode, atomic_tokens=[b"ab"])
    haves = trie.weight_sum(torch.tensor([0.1, 0.2, 0.2, 0.5]))

    leaf_wants = {
        b"a": 0.1,
        b"b": 0.2,
        b"ab": 0.2,
        b"<eos>": 0.5,
    }
    internal_wants = {
        b"": 1,
        b"a": 0.1,
        b"b": 0.2,
        b"ab": 0.2,
        b"<": 0.5,
        b"<e": 0.5,
        b"<eo": 0.5,
        b"<eos": 0.5,
        b"<eos>": 0.5,
    }

    assert_weights_close(trie, leaf_wants, internal_wants, haves, bytes)


@given(tokens_and_weights(n_weights=1))
def test_weight_sum(tokens_and_weights):
    vocab, weights = tokens_and_weights
    trie = TokenByteTrie(decode=vocab)
    haves = trie.weight_sum(weights[0])
    leaf_wants, internal_wants = make_wants(trie, weights[0], np.add, bytes)
    assert_weights_close(trie, leaf_wants, internal_wants, haves, bytes)


@given(tokens_and_weights(n_weights=1))
def test_weight_max(tokens_and_weights):
    vocab, weights = tokens_and_weights
    trie = TokenByteTrie(decode=vocab)
    haves = trie.weight_max(weights[0])
    leaf_wants, internal_wants = make_wants(trie, weights[0], np.maximum, bytes)
    assert_weights_close(trie, leaf_wants, internal_wants, haves, bytes)


@given(tokens_and_weights(n_weights=3))
def test_batch_weight_sum(tokens_and_weights):
    vocab, weights = tokens_and_weights
    trie = TokenByteTrie(decode=vocab)
    haves = trie.batch_weight_sum(weights)
    for i in range(len(weights)):
        leaf_wants, internal_wants = make_wants(trie, weights[i], np.add, bytes)
        assert_weights_close(trie, leaf_wants, internal_wants, haves[i], bytes)


@given(tokens_and_weights(n_weights=3))
def test_batch_weight_max(tokens_and_weights):
    vocab, weights = tokens_and_weights
    trie = TokenByteTrie(decode=vocab)
    haves = trie.batch_weight_max(weights)
    for i in range(len(weights)):
        leaf_wants, internal_wants = make_wants(trie, weights[i], np.maximum, bytes)
        assert_weights_close(trie, leaf_wants, internal_wants, haves[i], bytes)


@pytest.mark.asyncio
async def test_async_trie(mock_llm):
    async_trie = AsyncTokenByteTrie.from_vocab(mock_llm.byte_vocab)
    all_token_ids = [[0, 1, 3], [10, 20, 30], [8, 100]]
    all_weights = torch.exp(await mock_llm.batch_next_token_logprobs(all_token_ids))

    haves = await asyncio.gather(*[async_trie.weight_sum(ws) for ws in all_weights])
    haves = [h.cpu().numpy() for h in haves]
    wants = async_trie.trie.batch_weight_sum(all_weights).cpu().numpy()

    assert len(haves) == len(wants)
    for have, want in zip(haves, wants):
        np.testing.assert_allclose(have, want, rtol=1e-5, atol=1e-8)

    haves = await asyncio.gather(*[async_trie.weight_max(ws) for ws in all_weights])
    haves = [h.cpu().numpy() for h in haves]
    wants = async_trie.trie.batch_weight_max(all_weights).cpu().numpy()

    assert len(haves) == len(wants)
    for have, want in zip(haves, wants):
        np.testing.assert_allclose(have, want, rtol=1e-5, atol=1e-8)


@pytest.mark.asyncio
async def test_async_trie_cleanup(mock_llm):
    async_trie = AsyncTokenByteTrie.from_vocab(mock_llm.byte_vocab)
    async_trie.start()
    await async_trie.cleanup()
    assert async_trie._task is None


@pytest.mark.asyncio
async def test_async_error_handling(decode):
    async_trie = AsyncTokenByteTrie.from_vocab(decode)
    async_trie.start()
    with pytest.raises(ValueError):
        future = async_trie._queue_request(
            torch.tensor([0.1, 0.2, 0.2, 0.5]), TrieMode.WITHOUT_EOS, "invalid-op"
        )
        await future


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu"),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_preprocessing(decode, device):
    trie = TokenByteTrie(decode=decode, device=device)

    # Test numpy array input
    np_weights = np.array([[0.5, 0.5, 0.5, 0.5], [0.1, 0.5, 0.5, 0.5]])
    processed = trie._preprocess_ws(np_weights)
    assert isinstance(processed, torch.Tensor)
    assert processed.device.type == trie.device
    assert processed.dtype == torch.float32

    # Test list input
    list_weights = [[0.5, 0.5, 0.5, 0.5], [0.1, 0.5, 0.5, 0.5]]
    processed = trie._preprocess_ws(list_weights)
    assert isinstance(processed, torch.Tensor)
    assert processed.device.type == trie.device
    assert processed.dtype == torch.float32

    # Test tensor with wrong device
    if torch.cuda.is_available():
        wrong_device = "cuda" if trie.device == "cpu" else "cpu"
        tensor_weights = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5], [0.1, 0.5, 0.5, 0.5]], device=wrong_device
        )
        processed = trie._preprocess_ws(tensor_weights)
        assert processed.device.type == trie.device
        assert processed.dtype == torch.float32


def test_visualize(decode):
    trie = TokenByteTrie(decode=decode)

    trie.visualize()

    ws = torch.tensor([0.1] * len(trie.children))
    trie.visualize(ws)

    ws = torch.tensor([0] * len(trie.children))
    trie.visualize(ws)

    with pytest.raises(ValueError):
        trie.visualize(torch.tensor([0.1] * (len(trie.children) + 1)))


@pytest.mark.asyncio
async def test_eos_token_configuration():
    """Test EOS token configuration in trie."""
    vocab = [b"hello", b"world", b"<eos>"]
    eos_tokens = [b"<eos>"]

    # Test trie with EOS tokens
    trie = TokenByteTrie(decode=vocab, eos_tokens=eos_tokens)

    # EOS token should be in the eos_tokens set
    assert b"<eos>" in trie.eos_tokens
    assert len(trie.eos_tokens) == 1

    # EOS token IDs should be populated
    assert len(trie.eos_token_ids) == 1
    assert trie.eos_token_ids[0] == 2  # "<eos>" is at index 2

    # EOS node should exist
    assert hasattr(trie, "eos_node")
    assert trie.eos_node is not None

    # EOS node should be connected to root
    assert trie.children[trie.root].get(257) == trie.eos_node


@pytest.mark.asyncio
async def test_eos_dual_matrix_behavior():
    """Test dual matrix behavior for propagate_eos vs no_eos modes."""
    vocab = [b"hello", b"world", b"<eos>"]
    eos_tokens = [b"<eos>"]

    trie = TokenByteTrie(decode=vocab, eos_tokens=eos_tokens)
    weights = torch.tensor([0.3, 0.4, 0.3])  # hello, world, <eos>

    # Test no_eos mode (no EOS node mass)
    masses_without_eos = trie.weight_sum(weights, mode=TrieMode.WITHOUT_EOS)

    # Test propagate_eos mode (excludes EOS tokens' mass from ancestors and moves it to the EOS node)
    masses_with_eos = trie.weight_sum(weights, mode=TrieMode.WITH_EOS)

    # Both should be valid arrays
    assert len(masses_without_eos) == len(trie.children)
    assert len(masses_with_eos) == len(trie.children)

    root_mass_without_eos = masses_without_eos[trie.root]
    root_mass_with_eos = masses_with_eos[trie.root]

    assert np.isclose(root_mass_without_eos.item(), 1.0, rtol=1e-5)
    assert np.isclose(root_mass_with_eos.item(), 1.0, rtol=1e-5)

    # The masses should be different between modes
    assert not np.allclose(
        masses_without_eos.cpu().numpy(), masses_with_eos.cpu().numpy()
    )


@pytest.mark.asyncio
async def test_eos_weight_sum_with_eos():
    """Test weight_sum_with_eos method."""
    vocab = [b"hello", b"world", b"<eos>"]
    eos_tokens = [b"<eos>"]

    trie = TokenByteTrie(decode=vocab, eos_tokens=eos_tokens)
    weights = torch.tensor([0.3, 0.4, 0.1])  # hello, world, <eos>

    # Test with_eos mode
    masses_with_eos = trie.weight_sum(weights, mode=TrieMode.WITH_EOS)

    # Should return array with EOS node included
    assert len(masses_with_eos) == len(trie.children)
    assert not np.isnan(masses_with_eos.cpu().numpy()).any()

    # EOS node should have the EOS token probability
    eos_mass = masses_with_eos[trie.eos_node]
    assert np.isclose(eos_mass.item(), 0.1)  # EOS token weight


@pytest.mark.asyncio
async def test_eos_multiple_tokens():
    """Test with multiple EOS tokens."""
    vocab = [b"hello", b"world", b"dog", b"dogs"]
    eos_tokens = [b"dog", b"dogs"]

    trie = TokenByteTrie(decode=vocab, eos_tokens=eos_tokens)

    # Should have both EOS tokens
    assert len(trie.eos_tokens) == 2
    assert b"dog" in trie.eos_tokens
    assert b"dogs" in trie.eos_tokens

    # Should have both EOS token IDs
    assert len(trie.eos_token_ids) == 2
    assert 2 in trie.eos_token_ids  # dog at index 2
    assert 3 in trie.eos_token_ids  # dogs at index 3

    # Test weight sum with multiple EOS tokens
    weights = torch.tensor([0.2, 0.3, 0.1, 0.8])  # hello, world, dog, dogs
    masses = trie.weight_sum(weights, mode=TrieMode.WITH_EOS)

    # EOS node should collect both EOS token masses
    eos_mass = masses[trie.eos_node]
    expected_eos_mass = 0.1 + 0.8  # dog + dogs
    assert np.isclose(eos_mass.item(), expected_eos_mass, rtol=1e-5)


def test_invalid_device():
    with pytest.raises(ValueError):
        TokenByteTrie(decode=["a", "b", "c"], device="invalid")


def test_invalid_eos_tokens():
    with pytest.raises(ValueError):
        TokenByteTrie(decode=["a", "b", "c"], eos_tokens=["d"])


def test_invalid_atomic_tokens():
    with pytest.raises(ValueError):
        TokenByteTrie(decode=["a", "b", "c"], atomic_tokens=["d"])
