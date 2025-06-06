import html
import numpy as np
import pandas as pd
from IPython.display import HTML, SVG
from typing import Union
import warnings

from arsenal import colors


class LazyByteProbs:
    """Represents a lazy (log) probability distribution over bytes.

    Handles probability distributions over bytes plus an EOT (End of Token) symbol.

    Args:
        ps (list): List of probabilities
        log_space (bool, optional): Whether probabilities are in log space. Defaults to True
    """

    def __init__(self, ps, encode, decode, log_space=True):
        self.ps = ps
        self.encode = encode
        self.decode = decode
        self.log_space = log_space

    def __getitem__(self, b):
        """Get probability for a byte or EOT.

        Args:
            b (int|None): Byte value or None for EOT

        Returns:
            (float): Probability (or log probability) for the byte/EOT
        """
        return self.ps[self.encode[b]]

    def materialize(self):
        """Materializes the probability distribution into a Chart.

        Returns:
            (Chart): Chart with probabilities for each byte/EOT
        """
        Q = Chart(-np.inf if self.log_space else 0)
        for b, p in zip(self.decode, self.ps):
            Q[b] = p
        return Q

    def pretty(self):
        """Returns a pretty string representation of the probability distribution.

        Returns:
            (str): Pretty string representation of the probability distribution
        """
        return self.materialize().map_keys(
            lambda x: bytes([x]) if x in range(256) else ("EOT" if x is None else x)
        )


def logsumexp(arr):
    """
    Compute `log(sum(exp(arr)))` without overflow.
    """
    arr = np.array(arr, dtype=np.float64)
    arr = arr[arr > -np.inf]
    if len(arr) == 0:
        return -np.inf
    vmax = arr.max()
    arr -= vmax
    np.exp(arr, out=arr)
    out = np.log(arr.sum())
    out += vmax
    return out


def format_table(rows, headings=None):
    def fmt(x):
        if isinstance(x, (SVG, HTML)):
            return x.data
        elif hasattr(x, "_repr_html_"):
            return x._repr_html_()
        elif hasattr(x, "_repr_svg_"):
            return x._repr_svg_()
        elif hasattr(x, "_repr_image_svg_xml"):
            return x._repr_image_svg_xml()
        else:
            return f"<pre>{html.escape(str(x))}</pre>"

    return (
        "<table>"
        + (
            '<tr style="font-weight: bold;">'
            + "".join(f"<td>{x}</td>" for x in headings)
            + "</tr>"
            if headings
            else ""
        )
        + "".join(
            "<tr>" + "".join(f"<td>{fmt(x)}</td>" for x in row) + " </tr>"
            for row in rows
        )
        + "</table>"
    )


class Chart(dict):
    """A specialized dictionary for managing probability distributions.

    Extends dict with operations useful for probability distributions and numeric computations,
    including arithmetic operations, normalization, and visualization.

    Args:
        zero (Any): Default value for missing keys
        vals (tuple, optional): Initial (key, value) pairs
    """

    def __init__(self, zero, vals=()):
        self.zero = zero
        super().__init__(vals)

    def __missing__(self, k):
        return self.zero

    def spawn(self):
        return Chart(self.zero)

    def __add__(self, other):
        new = self.spawn()
        for k, v in self.items():
            new[k] += v
        for k, v in other.items():
            new[k] += v
        return new

    def __mul__(self, other):
        new = self.spawn()
        for k in self:
            v = self[k] * other[k]
            if v == self.zero:
                continue
            new[k] += v
        return new

    def copy(self):
        return Chart(self.zero, self)

    def trim(self):
        return Chart(self.zero, {k: v for k, v in self.items() if v != self.zero})

    def metric(self, other):
        assert isinstance(other, Chart)
        err = 0
        for x in self.keys() | other.keys():
            err = max(err, abs(self[x] - other[x]))
        return err

    def _repr_html_(self):
        return (
            '<div style="font-family: Monospace;">'
            + format_table(self.trim().items(), headings=["key", "value"])
            + "</div>"
        )

    def __repr__(self):
        return repr({k: v for k, v in self.items() if v != self.zero})

    def __str__(self, style_value=lambda k, v: str(v)):
        def key(k):
            return -self[k]

        return (
            "Chart {\n"
            + "\n".join(
                f"  {k!r}: {style_value(k, self[k])},"
                for k in sorted(self, key=key)
                if self[k] != self.zero
            )
            + "\n}"
        )

    def assert_equal(self, want, *, domain=None, tol=1e-5, verbose=False, throw=True):
        if not isinstance(want, Chart):
            want = Chart(self.zero, want)
        if domain is None:
            domain = self.keys() | want.keys()
        assert verbose or throw
        errors = []
        for x in domain:
            if abs(self[x] - want[x]) <= tol:
                if verbose:
                    print(colors.mark(True), x, self[x])
            else:
                if verbose:
                    print(colors.mark(False), x, self[x], want[x])
                errors.append(x)
        if throw:
            for x in errors:
                raise AssertionError(f"{x}: {self[x]} {want[x]}")

    def argmax(self):
        return max(self, key=self.__getitem__)

    def argmin(self):
        return min(self, key=self.__getitem__)

    def top(self, k):
        return Chart(
            self.zero,
            {k: self[k] for k in sorted(self, key=self.__getitem__, reverse=True)[:k]},
        )

    def max(self):
        return max(self.values())

    def min(self):
        return min(self.values())

    def sum(self):
        return sum(self.values())

    def sort(self, **kwargs):
        return Chart(self.zero, [(k, self[k]) for k in sorted(self, **kwargs)])

    def sort_descending(self):
        return Chart(
            self.zero, [(k, self[k]) for k in sorted(self, key=lambda k: -self[k])]
        )

    def normalize(self):
        Z = self.sum()
        if Z == 0:
            return self
        return Chart(self.zero, [(k, v / Z) for k, v in self.items()])

    def filter(self, f):
        return Chart(self.zero, [(k, v) for k, v in self.items() if f(k)])

    def map_values(self, f):
        return Chart(f(self.zero), [(k, f(v)) for k, v in self.items()])

    def map_keys(self, f):
        return Chart(self.zero, [(f(k), v) for k, v in self.items()])

    def project(self, f):
        "Apply the function `f` to each key; summing when f-transformed keys overlap."
        out = self.spawn()
        for k, v in self.items():
            out[f(k)] += v
        return out

    # TODO: the more general version of this method is join
    def compare(self, other, *, domain=None):
        if not isinstance(other, Chart):
            other = Chart(self.zero, other)
        if domain is None:
            domain = self.keys() | other.keys()
        rows = []
        for x in domain:
            m = abs(self[x] - other[x])
            rows.append(dict(key=x, self=self[x], other=other[x], metric=m))
        return pd.DataFrame(rows)

    def to_dict(self):
        return {k: v for k, v in self.items()}


def escape(x):
    if isinstance(x, int):  # assume its a byte
        x = bytes([x])
    if isinstance(x, bytes):
        y = repr(x)[2:-1]
    else:
        y = repr(x)[1:-1]
    return y.replace(" ", "␣")


def split_with_atomic_tokens(data: bytes, atomic_tokens: list[bytes]) -> list[Union[int, bytes]]:
    """
    Splits a bytestring into a list of either individual bytes (as integers) or atomic tokens (as bytes),
    depending on whether the current position matches an atomic token.

    Args:
        data (bytes): The input byte string to split.
        atomic_tokens (list[bytes]): A list of byte substrings that are treated as indivisible atomic tokens.

    Returns:
        list[Union[int, bytes]]: A list where each element is either:
            - an atomic token (as bytes) if a match is found at that position,
            - or a single byte (as an int) if no atomic token matches.

    Notes:
        - Matching is greedy but only left-to-right: at each position, the function checks for atomic token matches
          starting from length 1 up to the maximum token length.
        - Only the first match (shortest prefix match) is used; longer overlapping tokens may be missed if a shorter
          prefix matches first.
        - If atomic tokens overlap (e.g., b"A" and b"AB"), a warning is raised and only the shortest prefix match
          will be used.

    Example:
        >>> split_with_atomic_tokens(b"ABC", [b"A", b"AB"])
        [b'A', 66, 67]  # b"AB" is not matched because b"A" matched first
    """
    # Detect overlapping atomic tokens
    for i, token1 in enumerate(atomic_tokens):
        for j, token2 in enumerate(atomic_tokens):
            if i != j and (token1.startswith(token2) or token2.startswith(token1)):
                warnings.warn(
                    f"Overlapping atomic tokens detected: {token1!r} and {token2!r}. "
                    "Only the shortest matching prefix will be used."
                )
                break  # One warning is enough

    result = []
    i = 0
    token_set = set(atomic_tokens)
    max_len = max(len(t) for t in atomic_tokens) if atomic_tokens else 0

    while i < len(data):
        matched = False
        for length in range(1, max_len + 1):
            fragment = data[i:i+length]
            if fragment in token_set:
                result.append(fragment)
                i += length
                matched = True
                break
        if not matched:
            result.append(data[i])
            i += 1

    return result