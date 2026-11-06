"""Lexicon loader for the resolution package.

Serves the team-name vocabulary (cosmetic noise tokens and semantic
category discriminators) from the package's JSON lexicon file, keeping
the resolution logic data-driven and free of hardcoded token sets.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Optional

_DEFAULT_LEXICON_PATH = Path(__file__).with_name("lexicon.json")


class ResolutionLexicon:
    """Loads and serves the resolution vocabulary from its JSON source."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path) if path else _DEFAULT_LEXICON_PATH
        self._cosmetic: FrozenSet[str] = frozenset()
        self._categories: Dict[str, FrozenSet[str]] = {}
        self._token_to_category: Dict[str, str] = {}
        self._token_synonyms: Dict[str, str] = {}
        self._team_aliases: Dict[str, str] = {}
        self.load()

    def load(self) -> None:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._cosmetic = frozenset(
            token.lower() for token in data.get("cosmetic_noise_tokens", [])
        )
        self._categories = {
            category: frozenset(token.lower() for token in tokens)
            for category, tokens in data.get("category_discriminators", {}).items()
        }
        self._token_to_category = {
            token: category
            for category, tokens in self._categories.items()
            for token in tokens
        }
        self._token_synonyms = {
            key.lower(): value.lower()
            for key, value in data.get("token_synonyms", {}).items()
        }
        self._team_aliases = {
            alias.strip().lower(): canonical.strip().lower()
            for canonical, aliases in data.get("team_aliases", {}).items()
            for alias in aliases
        }

    @property
    def cosmetic_noise_tokens(self) -> FrozenSet[str]:
        return self._cosmetic

    @property
    def category_discriminators(self) -> Dict[str, FrozenSet[str]]:
        return self._categories

    @property
    def token_synonyms(self) -> Dict[str, str]:
        return self._token_synonyms

    def category_for(self, token: str) -> Optional[str]:
        """Return the semantic category a token belongs to, if any."""
        return self._token_to_category.get(token.lower())

    def canonical_for_alias(self, alias: str) -> Optional[str]:
        """Return the canonical team name for a known alias, if any."""
        return self._team_aliases.get(alias.strip().lower())


@lru_cache(maxsize=None)
def get_lexicon() -> ResolutionLexicon:
    """Process-wide cached lexicon instance."""
    return ResolutionLexicon()
