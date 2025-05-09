import torch
import pytest
import asyncio
import numpy as np
from transformers import AutoTokenizer

from genlm.backend.llm import MockAsyncLM
from genlm.bytes import TokenByteTrie, AsyncTokenByteTrie

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
        future = await async_trie._queue_request(
            torch.tensor([0.1, 0.2, 0.2, 0.5]), "invalid-op"
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


def test_invalid_device():
    with pytest.raises(ValueError):
        TokenByteTrie(decode=["a", "b", "c"], device="invalid")
