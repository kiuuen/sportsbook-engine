"""Route template deduction for observed network URLs.

This module isolates URL template inference so that classification and
replay tooling can share a single, dependency-light implementation without
importing the legacy monolith. Path segments and query parameters are
normalized into parameterized placeholders (e.g. ``{event_id}``).
"""

import re
import urllib.parse


class RouteTemplateDeducer:
    """Deduces a parameterized route template from a concrete URL."""

    UUID_PATTERN: re.Pattern = re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )
    EPOCH_TIMESTAMP_PATTERN: re.Pattern = re.compile(r"^\d{10}(\d{3})?$")
    LONG_NUMERIC_ID_PATTERN: re.Pattern = re.compile(r"^\d{5,}$")

    COUNT_LIMIT_TOKENS: tuple = ("count", "limit", "perpage", "max", "size", "page", "offset")
    SPORT_TOKENS: tuple = ("sport", "sports")
    CHAMPIONSHIP_TOKENS: tuple = ("champ", "championship", "league", "tournament", "competition", "comp")
    MARKET_TOKENS: tuple = ("market", "mkt")
    EVENT_TOKENS: tuple = ("event", "match", "fixture", "game")

    @staticmethod
    def _tokenize_key(key: str) -> list:
        return [t for t in re.split(r"[_\-\s\.]", key) if t]

    @classmethod
    def _deduce_query_placeholder(cls, key: str, value: str) -> str:
        """Deduces the placeholder for a single query parameter.

        Resolution order:
          1. Count/limit parameters keep their literal value.
          2. Sport keys -> ``{sport_id}``.
          3. Championship/league keys -> ``{champ_id}``.
          4. Market keys -> ``{market_id}``.
          5. Explicit event/match/fixture keys -> ``{event_id}``.
          6. Timestamp-like keys or values -> ``{timestamp}``.
          7. Long numeric values -> ``{<key>_id}``.
          8. Generic ``id`` keys -> ``{id}``.
          9. Otherwise the literal value is preserved.
        """
        key_lower = key.lower()
        key_tokens = {t.lower() for t in cls._tokenize_key(key_lower)}

        # 1. Count/limit parameters – keep their numeric value
        if any(token in key_tokens or token in key_lower for token in cls.COUNT_LIMIT_TOKENS):
            return f"{key}={value}"

        # 2-5. Semantic identifier resolution (token-based, no substring bugs)
        if any(token in key_tokens for token in cls.SPORT_TOKENS):
            return f"{key}={{sport_id}}"
        if any(token in key_tokens for token in cls.CHAMPIONSHIP_TOKENS):
            return f"{key}={{champ_id}}"
        if any(token in key_tokens for token in cls.MARKET_TOKENS):
            return f"{key}={{market_id}}"
        if any(token in key_tokens for token in cls.EVENT_TOKENS):
            return f"{key}={{event_id}}"

        # 6. Timestamp-like keys or values
        if "time" in key_tokens or "date" in key_tokens or cls.EPOCH_TIMESTAMP_PATTERN.match(value):
            return f"{key}={{timestamp}}"

        # 7. Long numeric values -> key-specific identifier
        if cls.LONG_NUMERIC_ID_PATTERN.match(value):
            return f"{key}={{{key}_id}}"

        # 8. Generic bare "id" key -> generic placeholder
        if key_tokens == {"id"} or key_lower == "id":
            return f"{key}={{id}}"

        # 9. Literal fallback
        return f"{key}={value}"

    @classmethod
    def deduce_template(cls, url: str) -> str:
        """Deduces a parameterized route template from a concrete URL.

        Raises:
            ValueError: If ``url`` is empty or not a string.
        """
        if not isinstance(url, str) or not url.strip():
            raise ValueError("A non-empty URL string is required to deduce a route template.")

        parsed = urllib.parse.urlparse(url)
        path_segments = parsed.path.split("/")
        deduced_segments = []

        for seg in path_segments:
            if not seg:
                deduced_segments.append("")
                continue
            if cls.UUID_PATTERN.match(seg):
                deduced_segments.append("{uuid}")
            elif cls.EPOCH_TIMESTAMP_PATTERN.match(seg):
                deduced_segments.append("{timestamp}")
            elif cls.LONG_NUMERIC_ID_PATTERN.match(seg):
                deduced_segments.append("{id}")
            else:
                deduced_segments.append(seg)

        deduced_path = "/".join(deduced_segments)

        query_params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        deduced_query = [
            cls._deduce_query_placeholder(k, vals[0] if vals else "")
            for k, vals in query_params.items()
        ]

        query_str = "&".join(deduced_query) if deduced_query else ""
        return urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc,
            deduced_path,
            parsed.params,
            query_str,
            parsed.fragment,
        ))
