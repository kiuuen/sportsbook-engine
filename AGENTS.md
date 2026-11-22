
# AGENTS.md



## Directives

- **Language:** English strictly for all code, docstrings, comments, commits, and logs.

- **Zero Stubs:** No `TODO`, `pass`, dummy mocks, or skipped assertions. Implement all paths completely.

- **No Hardcoding:** Domain vocabulary belongs in `sportsbook_lexicon.json`. Target quirks belong in SQLite `field_mappings`. Never declare static keyword dicts/sets in Python.

- **Real Fixtures:** Tests must load raw captured JSON from `tests/fixtures/`. No synthetic JSON.

- **Scope & Typing:** Edit ONLY the requested module. Full type hints on all functions. Minimal diffs.



## Commands

- Run tests: `.venv/bin/pytest -q --tb=short`

- Target test: `.venv/bin/pytest tests/ -k <module> -q --tb=short`

