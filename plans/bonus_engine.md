# Module Execution Spec: Bonus Engine Ingestion, Discovery & Lifecycle Pipeline

## Target Scope & Architecture
The `bonus_engine` is a fully decoupled subsystem operating on its own dedicated SQLite database (`data/bonus_engine.db`).
It monitors sportsbook domains, discovers promotional URLs, ingests terms, extracts structured promotion objects via LLM with mandatory textual evidence citations, and validates all mathematical and temporal constraints deterministically using pure Python rules before committing promotions to operator lifecycle states.

### Architectural Principles & Boundaries
1. **Decoupled Persistence**: Uses `BonusDatabaseManager` (`data/bonus_engine.db`) in WAL mode. Zero dependency on `control_plane.db` or `heuristic_classifier`.
2. **Deterministic LLM Audit**: LLM acts strictly as a text-to-structured-data parser. All output MUST contain `EvidenceCitation` quotes. `InvariantValidator` executes Python rules over extracted dataclasses. Invalid citations or mathematical contradictions transition promotions to `MANUAL_REVIEW`.
3. **Hybrid Acquisition Strategy**:
   - **Level 0 (Diff Watcher)**: Hash-based change detection over clean, sanitized text (excluding dynamic scripts, timestamps, nonces). Skips LLM calls if clean text is unchanged.
   - **Level 1 (Sitemap & Link Discovery)**: Auto-discovers promo pages from `/sitemap.xml` and root homepage links using regex path matching (`promociones`, `bonos`, `offers`, `terminos`, `welcome`). Evaluates candidates through `PromotionScanner`.
   - **Level 3 (Human Fallback)**: If zero viable promo links are found or JavaScript SPA execution fails, targets transition to `TargetAcquisitionState.NEEDS_BONUS_URL` for manual URL pinning (`pinned=1`).

---

## Data Contracts & State Machine Updates

### 1. `bonus_engine/models.py` Adjustments
- **`TargetAcquisitionState` Enum**:
  Add `NEEDS_BONUS_URL = "NEEDS_BONUS_URL"` for targets requiring operator intervention.
- **`VerificationState` Transition Graph (`_ALLOWED_TRANSITIONS`)**:
  Allow explicit transition from `EXTRACTED` to `MANUAL_REVIEW` and from `MANUAL_REVIEW` to `OPERATOR_CONFIRMED` or `REJECTED`:
  ```python
  VerificationState.EXTRACTED: {
      VerificationState.INVARIANT_VALIDATED,
      VerificationState.MANUAL_REVIEW,
      VerificationState.REJECTED,
  },
  VerificationState.MANUAL_REVIEW: {
      VerificationState.OPERATOR_CONFIRMED,
      VerificationState.REJECTED,
  },
  ```

### 2. Candidate Window Handling in `extractor.py`
- Modify `extract_from_evaluations` to process ALL viable candidate windows returned by `ScanEvaluation` rather than choosing only a single primary window, ensuring multi-promotion landings (e.g. Welcome + Cashback + Acca Boost) are extracted in full.

---

## Existing vs Missing Code Analysis

### Existing Code (Validated via CodeGraph)
- `bonus_engine/models.py`: Core dataclasses (`Promotion`, `WageringRequirement`, `TemporalValidity`, `MonetaryConstraints`, `SourceProvenance`, `EvidenceCitation`) and state enums.
- `bonus_engine/database.py`: `BonusDatabaseManager` with schema for `bonus_targets`, `promotion_sources`, `promotions`, and `target_source_registry`. Implements registry management, raw content versioning, and operator state protection.
- `bonus_engine/scanner.py`: `PromotionScanner` using `promo_lexicon.json` to score text relevance, candidate window extraction, and noise filtering.
- `bonus_engine/extractor.py`: `PromotionExtractor` & `ExtractionConfig` executing LLM extraction and initial validation routing.
- `bonus_engine/validator.py`: `InvariantValidator` auditing rollover multipliers (1x-50x), calculation base ambiguity, odds bounds (1.01-100.0), percentage bounds, temporal limits, and quote verification against source text.

### Missing Code & Extensions Required
- `bonus_engine/fetcher.py` **[NEW]**: HTTP fetcher with `curl_cffi`, HTML cleaner (DOM elements removal, clean text extraction), text normalization, and clean-text SHA-256 hash calculation.
- `bonus_engine/discovery.py` **[NEW]**: Level 1 link and sitemap scraper for domain root discovery.
- `bonus_engine/acquisition.py` **[NEW]**: Acquisition pipeline orchestrating Level 0 Diff Watching, Level 1 Discovery, Scanner filtering, Extractor execution, Invariant Validation, and SQLite updates.
- `bonus_engine/cli.py` or CLI subcommands in `main.py` **[EXTENDED]**: CLI subcommands for managing bonus targets, running pipeline polls, listing audit metrics, and executing interactive operator reviews.

---

## Task Breakdown & Implementation Steps

### Task 1: Data Model & Validator Routing Fixes (`bonus_engine/models.py` & `bonus_engine/extractor.py`)
- Update `TargetAcquisitionState` enum to include `NEEDS_BONUS_URL`.
- Update `_ALLOWED_TRANSITIONS` graph to permit `EXTRACTED -> MANUAL_REVIEW` and `MANUAL_REVIEW -> OPERATOR_CONFIRMED | REJECTED`.
- Update `PromotionExtractor._apply_lifecycle` in `extractor.py`:
  - When `InvariantValidator.validate` returns discrepancies, explicitly invoke `promotion.transition_to(VerificationState.MANUAL_REVIEW, notes=...)`.
- Update `extract_from_evaluations` in `extractor.py` to iterate through all `evaluation.candidate_windows` and collect promotions across all viable windows.

### Task 2: Robust Fetcher & HTML Sanitizer (`bonus_engine/fetcher.py`)
- Implement `BonusFetcher`:
  - Uses `curl_cffi.requests.AsyncSession` or `Session` with custom user-agents and timeout settings.
  - Returns raw HTML payload and HTTP status.
- Implement `HTMLSanitizer`:
  - Strips `<script>`, `<style>`, `<noscript>`, `<header>`, `<footer>`, `<nav>`, HTML comments.
  - Normalizes whitespace into clean plain text.
  - Computes deterministic SHA-256 hash on clean text: `hashlib.sha256(clean_text.encode('utf-8')).hexdigest()`.

### Task 3: Level 1 Discovery Scraper (`bonus_engine/discovery.py`)
- Implement `TargetDiscovery`:
  - `fetch_sitemap(domain: str) -> list[str]`: Fetches `/sitemap.xml` or parses `robots.txt` for sitemap URLs, extracting link entries matching promotional keyword patterns (`promociones`, `bonos`, `promos`, `offers`, `welcome`).
  - `extract_root_links(domain: str, html: str) -> list[str]`: Parses internal `<a>` tags matching promotional regexes.
  - Combines discovered URLs, deduplicates, and limits candidates per target domain (e.g. max 10 URLs per domain).

### Task 4: Acquisition Pipeline Orchestrator (`bonus_engine/acquisition.py`)
- Implement `BonusAcquisitionPipeline`:
  - Accepts `BonusDatabaseManager`, `BonusFetcher`, `PromotionScanner`, `PromotionExtractor`.
  - Method `process_target(domain: str)`:
    1. Fetches registered/pinned URLs from `target_source_registry`.
    2. If no pinned URLs exist, runs `TargetDiscovery`. If discovery finds zero viable candidate URLs, transitions target state to `NEEDS_BONUS_URL` and returns.
    3. For each candidate URL:
       - Fetches payload via `BonusFetcher`. If HTTP error (404/403/redirect failure), logs warning and marks source as unreachable.
       - Runs Diff Watcher: compares clean text SHA-256 hash with `last_content_hash` in `target_source_registry`.
       - If hash matches `last_content_hash`, skips downstream LLM calls (Level 0 optimization).
       - If hash differs: runs `PromotionScanner.evaluate_text()`.
       - If scan is viable (`evaluation.is_viable`): calls `PromotionExtractor.extract_from_evaluations()`.
       - For each extracted promotion: saves/upserts promotion in `BonusDatabaseManager.save_promotion()`.
       - Updates `record_source_check()` and `record_source()` with the new content hash and timestamp.

### Task 5: Subcommand Dispatcher & CLI Commands (`bonus_engine/cli.py` & `main.py`)
- Implement subcommands in CLI:
  - `bonus-target-add <domain> [--url <url>]`: Registers a new target domain and optional pinned source URL.
  - `bonus-poll [--domain <domain>]`: Executes `BonusAcquisitionPipeline` across active domains.
  - `bonus-audit`: Displays summary counts of targets by acquisition state and promotions by verification state (`INVARIANT_VALIDATED`, `MANUAL_REVIEW`, `OPERATOR_CONFIRMED`, `REJECTED`).
  - `bonus-review`: Interactive CLI loop allowing operators to view `MANUAL_REVIEW` promotions with discrepancy details and promote to `OPERATOR_CONFIRMED` or `REJECTED`.

---

## Invariants, Edge Cases & Safety Mechanisms

1. **Operator State Lock**: `BonusDatabaseManager.save_promotion` MUST preserve promotions in `OPERATOR_CONFIRMED` or `REJECTED` state unless the source content version has changed.
2. **Clean Text Hash Isolation**: Hash generation MUST operate on sanitized text only, ensuring dynamic HTML tokens (session IDs, dynamic scripts, timestamps) do not trigger spurious LLM calls.
3. **Verbatim Citation Matching**: `InvariantValidator` MUST check that every `EvidenceCitation.quote` appears verbatim in `source_text`. If a quote fails string lookup, the validator flags `DiscrepancyCode.CITATION_NOT_FOUND` and routes to `MANUAL_REVIEW`.
4. **Range & Mathematical Audits**:
   - `wagering_requirement.multiplier`: 1.0 to 50.0.
   - `monetary_constraints.bonus_percentage`: 10.0 to 1000.0.
   - `odds_constraints.min_odds_per_selection`: 1.01 to 100.0.
   - `temporal_validity.claim_window_days` / `completion_window_days`: <= 365 days.
5. **Handling Empty/JS Shell Pages**: If clean text length is below 100 characters after HTML stripping, `BonusFetcher` flags the fetch as insufficient content, preventing scanner errors and routing target to `NEEDS_BONUS_URL` for manual intervention if no alternative links succeed.

---

## Verification & Testing Command
Run unit and integration tests across the complete bonus engine module:
```bash
.venv/bin/pytest tests/ -k bonus_engine -q --tb=short
```
