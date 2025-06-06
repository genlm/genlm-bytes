![Logo](assets/logo.png)

<p align="center">
<a href="https://genlm.github.io/genlm-bytes/"><img alt="Docs" src="https://github.com/genlm/genlm-bytes/actions/workflows/docs.yml/badge.svg"/></a>
<a href="https://genlm.github.io/genlm-bytes/"><img alt="Tests" src="https://github.com/genlm/genlm-bytes/actions/workflows/pytest.yml/badge.svg"/></a>
<a href="https://codecov.io/github/genlm/genlm-bytes" >  <img src="https://codecov.io/github/genlm/genlm-bytes/graph/badge.svg?token=4atmwhxEeb"/></a>
</p>

GenLM Bytes is a Python library for byte-level language modeling. It contains algorithms for turning token-level language models into byte-level language models.

See the [docs](https://genlm.github.io/genlm-bytes/) for details and [basic usage](https://genlm.github.io/genlm-bytes/usage).

*Note: This project is under active development — expect bugs, missing features, and breaking changes. Please report any issues or suggestions in the issue tracker.*

## Quick Start

This library requires python>=3.11 and can be installed using pip:

```bash
pip install genlm-bytes
```

For faster and less error-prone installs, consider using [`uv`](https://github.com/astral-sh/uv):

```bash
uv pip install genlm-bytes
```

See [DEVELOPING.md](https://github.com/genlm/genlm-bytes/blob/main/DEVELOPING.md) for details on how to install the project for development.

## Usage

```python
from genlm.bytes import ByteBeamState, BeamParams
from genlm.backend import load_model_by_name

# Load a token-level language model from a huggingface model name
# (Note: for fast GPU inference, specify `backend="vllm"`)
llm = load_model_by_name("gpt2-medium")

# Initialize a beam state with a maximum beam width of 5 and a prune threshold of 0.05 (higher threshold values lead to more aggressive pruning).
beam = await ByteBeamState.initial(llm, BeamParams(K=5, prune_threshold=0.05))

# Populate the beam state with byte context.
beam = await beam.prefill(b"An apple a day keeps the ")

# Get the log probability distribution over the next byte.
logp_next = await beam.logp_next()
logp_next.pretty().top(5)
# Example output:
# b'd' -0.5766762743944795
# b'b' -2.8732729803080233
# b's' -2.9816068063730867
# b'w' -3.3758250127787264
# b'm' -3.528177345847574

# Prune the beam and extend it with a new byte
new_beam = await (beam.prune() << 100) # 100 is the byte value of 'd'
```

See [basic usage](https://genlm.github.io/genlm-bytes/usage) for a more detailed example.
