"""Unit tests for the text normalization utilities in ``resolution.cleaner``.

Covers normalization, qualifier extraction, character n-gram generation,
and tokenization. All scenarios use real-world-style team name strings;
no synthetic JSON payloads are involved.
"""

import pytest

from resolution.cleaner import (
    extract_category_markers,
    extract_qualifiers,
    generate_char_ngrams,
    normalize_team_name,
    normalize_text,
    tokenize,
)


class TestNormalizeText:
    """Tests for :func:`normalize_text` and its alias."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Atlético de Madrid", "atletico de madrid"),
            ("São Paulo", "sao paulo"),
            ("Boca Juniors!", "boca juniors"),
            ("R.C. Celta de Vigo", "r c celta de vigo"),
            ("  Real   Madrid  ", "real madrid"),
        ],
    )
    def test_strips_diacritics_punctuation_and_whitespace(
        self, raw: str, expected: str
    ) -> None:
        assert normalize_text(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
    def test_defensive_coercion_returns_empty_string(self, raw: object) -> None:
        assert normalize_text(raw) == ""

    def test_normalize_team_name_alias_matches(self) -> None:
        samples = ["Atlético de Madrid", "São Paulo", None, ""]
        for sample in samples:
            assert normalize_team_name(sample) == normalize_text(sample)


class TestExtractQualifiers:
    """Tests for :func:`extract_qualifiers` and its alias."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Arsenal U21", {"youth:21"}),
            ("Juventus U-19", {"youth:19"}),
            ("Santos Sub-20", {"youth:20"}),
            ("Santos Sub20", {"youth:20"}),
            ("Flamengo Youth", {"youth"}),
        ],
    )
    def test_youth_markers(self, raw: str, expected: set) -> None:
        assert extract_qualifiers(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Real Madrid B", {"reserve"}),
            ("Barcelona II", {"reserve"}),
            ("Ajax Reserves", {"reserve"}),
            ("Porto Res", {"reserve"}),
        ],
    )
    def test_reserve_markers(self, raw: str, expected: set) -> None:
        assert extract_qualifiers(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Chelsea Women", {"gender:female"}),
            ("Tigres Femenil", {"gender:female"}),
            ("Lyon Fem", {"gender:female"}),
            ("Arsenal Ladies", {"gender:female"}),
        ],
    )
    def test_gender_markers(self, raw: str, expected: set) -> None:
        assert extract_qualifiers(raw) == expected

    @pytest.mark.parametrize("raw", ["Arsenal", "Liverpool", "Real Madrid"])
    def test_clean_names_yield_no_qualifiers(self, raw: str) -> None:
        assert extract_qualifiers(raw) == set()

    def test_empty_and_none_inputs_are_safe(self) -> None:
        assert extract_qualifiers(None) == set()
        assert extract_qualifiers("") == set()

    def test_extract_category_markers_alias_matches(self) -> None:
        samples = ["Arsenal U21", "Real Madrid B", "Chelsea Women", "Arsenal"]
        for sample in samples:
            assert extract_category_markers(sample) == extract_qualifiers(sample)


class TestGenerateCharNgrams:
    """Tests for :func:`generate_char_ngrams`."""

    def test_generates_trigrams_from_normalized_text(self) -> None:
        assert generate_char_ngrams("Atletico", n=3) == {
            "atl",
            "tle",
            "let",
            "eti",
            "tic",
            "ico",
        }

    def test_whitespace_is_removed_before_windowing(self) -> None:
        assert generate_char_ngrams("real madrid", n=3) == generate_char_ngrams(
            "realmadrid", n=3
        )

    def test_short_input_returns_whole_compacted_string(self) -> None:
        assert generate_char_ngrams("ab", n=3) == {"ab"}

    def test_empty_input_returns_empty_set(self) -> None:
        assert generate_char_ngrams("", n=3) == set()
        assert generate_char_ngrams(None, n=3) == set()

    def test_invalid_n_raises_value_error(self) -> None:
        for bad_n in (0, -1):
            with pytest.raises(ValueError):
                generate_char_ngrams("abc", n=bad_n)


class TestTokenize:
    """Tests for :func:`tokenize`."""

    def test_returns_clean_lowercase_tokens(self) -> None:
        assert tokenize("  Real   Madrid C.F. ") == [
            "real",
            "madrid",
            "c",
            "f",
        ]

    def test_strips_diacritics_before_splitting(self) -> None:
        assert tokenize("Atlético de Madrid") == ["atletico", "de", "madrid"]

    def test_empty_and_none_inputs_are_safe(self) -> None:
        assert tokenize(None) == []
        assert tokenize("") == []
        assert tokenize("   ") == []
