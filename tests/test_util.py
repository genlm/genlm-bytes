import pytest
import numpy as np
import pandas as pd

from genlm.bytes.util import LazyByteProbs, Chart, format_table, escape, logsumexp


class TestLazyByteProbs:
    def test_init_log_space(self):
        ps = [0.1] * 256 + [0.2] + [0.3]  # 256 bytes + 1 EOT + 1 EOS
        lazy_probs = LazyByteProbs(ps, log_space=True)
        assert lazy_probs.ps == ps
        assert lazy_probs.log_space is True

    def test_init_prob_space(self):
        ps = [0.1] * 256 + [0.2] + [0.3]  # 256 bytes + 1 EOT + 1 EOS
        lazy_probs = LazyByteProbs(ps, log_space=False)
        assert lazy_probs.ps == ps
        assert lazy_probs.log_space is False

    def test_init_default_log_space(self):
        ps = [0.1] * 256 + [0.2] + [0.3]  # 256 bytes + 1 EOT + 1 EOS
        lazy_probs = LazyByteProbs(ps)
        assert lazy_probs.log_space is True

    def test_init_wrong_length(self):
        ps = [0.1] * 255  # Wrong length
        with pytest.raises(AssertionError):
            LazyByteProbs(ps)

    def test_getitem_byte(self):
        ps = list(range(258))
        lazy_probs = LazyByteProbs(ps)
        assert lazy_probs[0] == 0
        assert lazy_probs[100] == 100
        assert lazy_probs[255] == 255

    def test_getitem_eot(self):
        ps = list(range(258))
        lazy_probs = LazyByteProbs(ps)
        assert lazy_probs[None] == 256

    def test_materialize_log_space(self):
        ps = [-1.0] * 256 + [-2.0] + [-3.0]
        lazy_probs = LazyByteProbs(ps, log_space=True)
        chart = lazy_probs.materialize()

        assert isinstance(chart, Chart)
        assert chart.zero == -np.inf
        assert chart[0] == -1.0
        assert chart[255] == -1.0
        assert chart[None] == -2.0

    def test_materialize_prob_space(self):
        ps = [0.1] * 256 + [0.2] + [0.3]
        lazy_probs = LazyByteProbs(ps, log_space=False)
        chart = lazy_probs.materialize()

        assert isinstance(chart, Chart)
        assert chart.zero == 0
        assert chart[0] == 0.1
        assert chart[255] == 0.1
        assert chart[None] == 0.2

    def test_pretty(self):
        ps = [0.1] * 256 + [0.2] + [0.3]
        lazy_probs = LazyByteProbs(ps, log_space=False)
        pretty_chart = lazy_probs.pretty()

        assert isinstance(pretty_chart, Chart)
        # Check that keys are transformed properly
        for key in pretty_chart.keys():
            if key == "EOT":
                assert pretty_chart[key] == 0.2
            elif key == "EOS":
                assert pretty_chart[key] == 0.3
            else:
                assert isinstance(key, bytes)
                assert len(key) == 1
                assert pretty_chart[key] == 0.1

    def test_invalid_index(self):
        ps = list(range(258))
        lazy_probs = LazyByteProbs(ps)
        with pytest.raises(ValueError):
            lazy_probs[258]


def test_format_table():
    rows = [["a", "b"], ["c", "d"]]
    headings = ["col1", "col2"]
    result = format_table(rows, headings)

    assert "<table>" in result


def test_logsumexp_empty():
    arr = []
    result = logsumexp(arr)
    assert result == -np.inf


class TestChart:
    def test_init(self):
        chart = Chart(0)
        assert chart.zero == 0
        assert len(chart) == 0

    def test_init_with_vals(self):
        vals = [("a", 1), ("b", 2)]
        chart = Chart(0, vals)
        assert chart["a"] == 1
        assert chart["b"] == 2

    def test_missing_returns_zero(self):
        chart = Chart(42)
        assert chart["nonexistent"] == 42

    def test_spawn(self):
        chart = Chart(5)
        chart["a"] = 10
        new_chart = chart.spawn()
        assert new_chart.zero == 5
        assert len(new_chart) == 0
        assert new_chart["a"] == 5

    def test_add(self):
        chart1 = Chart(0, [("a", 1), ("b", 2)])
        chart2 = Chart(0, [("a", 3), ("c", 4)])
        result = chart1 + chart2

        assert result["a"] == 4
        assert result["b"] == 2
        assert result["c"] == 4

    def test_mul(self):
        chart1 = Chart(0, [("a", 2), ("b", 3)])
        chart2 = Chart(1, [("a", 4), ("c", 5)])
        result = chart1 * chart2

        assert result["a"] == 8
        assert result["b"] == 3
        assert "c" not in result

    def test_mul_zero_elimination(self):
        chart1 = Chart(0, [("a", 2), ("b", 0)])
        chart2 = Chart(0, [("a", 3), ("b", 5)])
        result = chart1 * chart2

        assert result["a"] == 6
        assert "b" not in result  # 0 * 5 = 0, which equals zero, so eliminated

    def test_copy(self):
        chart = Chart(10, [("a", 1), ("b", 2)])
        copied = chart.copy()

        assert copied.zero == 10
        assert copied["a"] == 1
        assert copied["b"] == 2

        # Modify original
        chart["c"] = 3
        assert "c" not in copied

    def test_trim(self):
        chart = Chart(0, [("a", 1), ("b", 0), ("c", 2)])
        trimmed = chart.trim()

        assert "a" in trimmed
        assert "c" in trimmed
        assert "b" not in trimmed

    def test_metric(self):
        chart1 = Chart(0, [("a", 1), ("b", 2)])
        chart2 = Chart(0, [("a", 1.5), ("c", 3)])

        metric = chart1.metric(chart2)
        expected = max(abs(1 - 1.5), abs(2 - 0), abs(0 - 3))  # max difference
        assert metric == expected

    def test_repr_html(self):
        chart = Chart(0, [("a", 1), ("b", 2)])
        html = chart._repr_html_()

        assert '<div style="font-family: Monospace;">' in html
        assert "<table>" in html
        assert "a" in html
        assert "1" in html

    def test_repr(self):
        chart = Chart(0, [("a", 1), ("b", 2)])
        repr_str = repr(chart)

        assert "a" in repr_str
        assert "1" in repr_str

    def test_str(self):
        chart = Chart(0, [("a", 1), ("b", 2)])
        str_repr = str(chart)

        assert "Chart {" in str_repr
        assert "'a': 1" in str_repr or "'b': 2" in str_repr

    def test_str_with_custom_style(self):
        chart = Chart(0, [("a", 1), ("b", 2)])
        str_repr = chart.__str__(style_value=lambda k, v: f"value_{v}")

        assert "value_1" in str_repr or "value_2" in str_repr

    def test_assert_equal_success(self):
        chart1 = Chart(0, [("a", 1), ("b", 2)])
        chart2 = Chart(0, [("a", 1), ("b", 2)])

        chart1.assert_equal(chart2)

    def test_assert_equal_failure(self):
        chart1 = Chart(0, [("a", 1), ("b", 2)])
        chart2 = Chart(0, [("a", 1.1), ("b", 2)])

        with pytest.raises(AssertionError):
            chart1.assert_equal(chart2)

    def test_assert_equal_with_tolerance(self):
        chart1 = Chart(0, [("a", 1), ("b", 2)])
        chart2 = Chart(0, [("a", 1.001), ("b", 2)])

        chart1.assert_equal(chart2, tol=0.01)

        with pytest.raises(AssertionError):
            chart1.assert_equal(chart2, tol=0.0001)

    def test_assert_equal_with_domain(self):
        chart1 = Chart(0, [("a", 1), ("b", 2)])
        chart2 = Chart(0, [("a", 1), ("c", 3)])

        chart1.assert_equal(chart2, domain=["a"])

    def test_assert_equal_verbose_only(self):
        chart1 = Chart(0, [("a", 1), ("b", 2)])
        chart2 = Chart(0, [("a", 1.1), ("b", 2)])

        # Should not raise with verbose=True and throw=False
        chart1.assert_equal(chart2, throw=False, verbose=True)

    def test_assert_equal_with_dict(self):
        chart = Chart(0, [("a", 1), ("b", 2)])
        dict_data = {"a": 1, "b": 2}

        # Should not raise
        chart.assert_equal(dict_data)

    def test_argmax(self):
        chart = Chart(0, [("a", 1), ("b", 3), ("c", 2)])
        assert chart.argmax() == "b"

    def test_argmin(self):
        chart = Chart(0, [("a", 1), ("b", 3), ("c", 2)])
        assert chart.argmin() == "a"

    def test_top(self):
        chart = Chart(0, [("a", 1), ("b", 3), ("c", 2), ("d", 4)])
        top2 = chart.top(2)

        assert len(top2) == 2
        assert "d" in top2
        assert "b" in top2

    def test_max_min_sum(self):
        chart = Chart(0, [("a", 1), ("b", 3), ("c", 2)])

        assert chart.max() == 3
        assert chart.min() == 1
        assert chart.sum() == 6

    def test_sort(self):
        chart = Chart(0, [("c", 3), ("a", 1), ("b", 2)])
        sorted_chart = chart.sort()

        keys = list(sorted_chart.keys())
        assert keys == ["a", "b", "c"]

    def test_sort_descending(self):
        chart = Chart(0, [("a", 1), ("b", 3), ("c", 2)])
        sorted_chart = chart.sort_descending()

        keys = list(sorted_chart.keys())
        values = [sorted_chart[k] for k in keys]
        assert values == sorted(values, reverse=True)

    def test_normalize(self):
        chart = Chart(0, [("a", 2), ("b", 3), ("c", 5)])
        normalized = chart.normalize()

        assert abs(normalized.sum() - 1.0) < 1e-10
        assert normalized["a"] == 0.2
        assert normalized["b"] == 0.3
        assert normalized["c"] == 0.5

    def test_normalize_zero_sum(self):
        chart = Chart(0, [("a", 0), ("b", 0)])
        normalized = chart.normalize()
        assert normalized["a"] == 0
        assert normalized["b"] == 0

    def test_filter(self):
        chart = Chart(0, [("a", 1), ("ab", 2), ("b", 3)])
        filtered = chart.filter(lambda k: k.startswith("a"))

        assert "a" in filtered
        assert "ab" in filtered
        assert "b" not in filtered

    def test_map_values(self):
        chart = Chart(0, [("a", 1), ("b", 2)])
        mapped = chart.map_values(lambda x: x * 2)

        assert mapped.zero == 0
        assert mapped["a"] == 2
        assert mapped["b"] == 4

    def test_map_keys(self):
        chart = Chart(0, [("a", 1), ("b", 2)])
        mapped = chart.map_keys(lambda k: k.upper())

        assert mapped["A"] == 1
        assert mapped["B"] == 2
        assert "a" not in mapped

    def test_project(self):
        chart = Chart(0, [("aa", 1), ("ab", 2), ("ba", 3)])
        projected = chart.project(lambda k: k[0])

        assert projected["a"] == 3
        assert projected["b"] == 3

    def test_compare(self):
        chart1 = Chart(0, [("a", 1), ("b", 2)])
        chart2 = Chart(0, [("a", 1.1), ("c", 3)])

        df = chart1.compare(chart2)

        assert isinstance(df, pd.DataFrame)
        assert "key" in df.columns
        assert "self" in df.columns
        assert "other" in df.columns
        assert "metric" in df.columns

    def test_compare_with_dict(self):
        chart = Chart(0, [("a", 1), ("b", 2)])
        dict_data = {"a": 1.1, "c": 3}

        df = chart.compare(dict_data)
        assert isinstance(df, pd.DataFrame)

    def test_compare_with_domain(self):
        chart1 = Chart(0, [("a", 1), ("b", 2)])
        chart2 = Chart(0, [("a", 1.1), ("c", 3)])

        df = chart1.compare(chart2, domain=["a"])
        assert len(df) == 1
        assert df.iloc[0]["key"] == "a"

    def test_to_dict(self):
        chart = Chart(0, [("a", 1), ("b", 2)])
        result = chart.to_dict()

        assert result == {"a": 1, "b": 2}


class TestEscape:
    def test_escape_byte(self):
        result = escape(65)  # ASCII 'A'
        expected = "A"
        assert result == expected

    def test_escape_bytes(self):
        result = escape(b"hello")
        expected = "hello"
        assert result == expected

    def test_escape_bytes_with_space(self):
        result = escape(b"hello world")
        expected = "hello␣world"
        assert result == expected

    def test_escape_string(self):
        result = escape("hello world")
        expected = "hello␣world"
        assert result == expected

    def test_escape_special_characters(self):
        result = escape(b"\n\t\r")
        expected = "\\n\\t\\r"
        assert result == expected

    def test_escape_non_printable_byte(self):
        result = escape(255)
        expected = "\\xff"
        assert result == expected

    def test_escape_null_byte(self):
        result = escape(0)
        expected = "\\x00"
        assert result == expected
