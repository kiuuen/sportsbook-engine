## 1. Production-Ready Code & Anti-Hardcoding Directives
- **Zero Placeholders & Stubs:** Never use `TODO`, `pass`, `// implement later`, dummy mocks, or empty function bodies.
- **No Lazy Hardcoding:** Do not substitute real dynamic logic with hardcoded fixtures, fake responses, or magic constants just to make scripts pass superficially. Parameters, configurations, and dynamic states must be explicitly handled.
- **Large File Integrity & Targeted Changes:**
  - When modifying large files, only target the relevant code blocks, but ensure the resulting file remains 100% syntactically complete and functional.
  - Never replace surrounding working code with elisions like `# ... existing code remains here ...` or truncate existing logic to save output length.
- Every function, method, and class must include explicit error handling (graceful failure instead of silent crashes) and complete static type hints.
## Language Constraint
- All source code, docstrings, inline comments, variable/function identifiers, exception messages, and console/CLI print outputs MUST be strictly written in Englis/.