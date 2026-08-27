# Agent Docs

Working documents produced by the 2026-08-27 code analysis & optimization pass (commit `efb8c0a`).

- [code_analysis.md](code_analysis.md) — architecture walkthrough, per-module notes, and the
  13 correctness issues that were fixed (with file:line references to the pre-optimization state).
- [optimization_report.md](optimization_report.md) — what was changed, benchmark results
  (32.7 → 28.6 ms/frame on RTX 4090 / 3.94M particles), the negative device-side-grid-sizing
  result, and how to reproduce.
- [known_issues.md](known_issues.md) — remaining correctness landmines (all in dead code paths),
  deferred performance ideas, and cleanup debt.
