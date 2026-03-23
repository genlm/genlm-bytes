import pytest
import torch
import numpy as np

from genlm.bytes.trie import TokenByteTrie, EOS
from genlm.bytes.byte_lm.trie_state import TrieMode
from genlm.backend.tokenization import Token


def find_token_id_by_bytes(decode, target_bytes):
    """Find the first token ID for a given byte string in a list of Token objects.
    
    Note: Returns only the first match if multiple tokens share the same byte string.
    This is fine for these tests since the test vocabularies have unique byte strings.
    """
    for token in decode:
        if token.byte_string == target_bytes:
            return token.token_id
    raise ValueError(f"{target_bytes} is not in decode")


@pytest.fixture(scope="module")
def eos_trie():
    """Provides a pre-configured trie with multiple EOS tokens."""
    vocab = [
        Token(token_id=0, byte_string=b"hello"),
        Token(token_id=1, byte_string=b"world"),
        Token(token_id=2, byte_string=b"!"),  # EOS
        Token(token_id=3, byte_string=b"!!"),
        Token(token_id=4, byte_string=b"."),  # EOS
        Token(token_id=5, byte_string=b"normal"),
        Token(token_id=6, byte_string=b"end"),
        Token(token_id=7, byte_string=b"</s>"),  # EOS
    ]
    eos_byte_strings = [b"!", b".", b"</s>"]
    return TokenByteTrie(decode=vocab, eos_byte_strings=eos_byte_strings)


def test_trie_structure(eos_trie: TokenByteTrie):
    """1. Validates the fundamental structure of the trie with EOS configured."""
    # An EOS node should be created
    assert hasattr(eos_trie, "eos_node")
    assert eos_trie.eos_node is not None

    # The EOS node should be a direct child of the root via the special EOS byte
    assert eos_trie.children[eos_trie.root].get(EOS) == eos_trie.eos_node

    # The original EOS tokens should still exist as leaf nodes in the trie for conditioning
    # Now word2leaf uses (bytes, token_id) as keys
    for eos_token_bytes in eos_trie.eos_byte_strings:
        token_id = find_token_id_by_bytes(eos_trie.decode, eos_token_bytes)
        assert (eos_token_bytes, token_id) in eos_trie.word2leaf


def test_without_eos_mode_mass_distribution(eos_trie: TokenByteTrie):
    """2. Validates that in conditioning mode, EOS tokens contribute to ancestor mass like normal tokens."""
    # Probabilities: "!" = 0.4, "!!" = 0.1. Total starting with "!" is 0.5
    weights = torch.tensor([0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.0])

    # In no_eos mode, EOS tokens are treated normally
    masses = eos_trie.weight_sum(weights, mode=TrieMode.WITHOUT_EOS)

    # Find the node for the prefix "!"
    node_for_exclamation = eos_trie.children[eos_trie.root][ord("!")]

    # The mass at this node should be the sum of probabilities of all tokens starting with "!", including "!" itself.
    idx_excl = find_token_id_by_bytes(eos_trie.decode, b"!")
    idx_excl2 = find_token_id_by_bytes(eos_trie.decode, b"!!")
    expected_mass = weights[idx_excl] + weights[idx_excl2]  # P("!") + P("!!")
    assert np.isclose(masses[node_for_exclamation].item(), expected_mass.item())

    # The EOS node should have zero mass in no_eos mode
    assert masses[eos_trie.eos_node] == 0.0


def test_with_eos_mode_mass_distribution(eos_trie: TokenByteTrie):
    """Validates that in propagate_eos mode, EOS tokens DO NOT contribute to ancestor mass."""
    # Probabilities: "!" = 0.4, "!!" = 0.1.
    weights = torch.tensor([0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.0])

    # In propagate_eos mode, EOS tokens should be excluded from ancestor paths
    masses = eos_trie.weight_sum(weights, mode=TrieMode.WITH_EOS)

    # Find the node for the prefix "!"
    node_for_exclamation = eos_trie.children[eos_trie.root][ord("!")]

    # The mass at this node should ONLY be the sum of non-EOS tokens starting with "!"
    idx_excl2 = find_token_id_by_bytes(eos_trie.decode, b"!!")
    expected_mass = weights[idx_excl2]  # Only P("!!")
    assert np.isclose(masses[node_for_exclamation].item(), expected_mass.item())


def test_with_eos_mode_eos_node_aggregation(eos_trie: TokenByteTrie):
    """Validates that the virtual EOS node correctly aggregates all EOS token probabilities in propagate_eos mode."""
    weights = torch.tensor(
        [0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1]
    )  # "!", ".", "</s>" are EOS

    masses = eos_trie.weight_sum(weights, mode=TrieMode.WITH_EOS)

    # The mass of the EOS node should be the sum of all defined EOS tokens
    idx_excl = find_token_id_by_bytes(eos_trie.decode, b"!")
    idx_dot = find_token_id_by_bytes(eos_trie.decode, b".")
    idx_eos = find_token_id_by_bytes(eos_trie.decode, b"</s>")
    expected_eos_mass = weights[idx_excl] + weights[idx_dot] + weights[idx_eos]
    actual_eos_mass = masses[eos_trie.eos_node]

    assert np.isclose(actual_eos_mass.item(), expected_eos_mass.item())


def test_root_mass_conservation(eos_trie: TokenByteTrie):
    """Validates that the root node's mass is correct in both modes, conserving total probability."""
    weights = torch.tensor([0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1])
    total_prob = torch.sum(weights).item()

    # No_eos mode: root mass should be the sum of all token probabilities
    masses_no_eos = eos_trie.weight_sum(weights, mode=TrieMode.WITHOUT_EOS)
    assert np.isclose(masses_no_eos[eos_trie.root].item(), total_prob)

    # With_eos mode: root mass should also be the sum of all token probabilities,
    # because the mass of the EOS node is added back to the root.
    masses_with_eos = eos_trie.weight_sum(weights, mode=TrieMode.WITH_EOS)
    assert np.isclose(masses_with_eos[eos_trie.root].item(), total_prob)
