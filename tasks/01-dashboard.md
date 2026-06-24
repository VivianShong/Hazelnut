# Task 01 — Dashboard (inquiry DAG + verdicts + pass@1 curves)

**Owner:** _____   **GPU:** none   **Depends on:** `ledger.json` only (read-only)

## Goal
A shareable visual of the science engine — the demo's money shot. One command renders:
1. **Inquiry DAG**: each experiment as a node showing its question + verdict, edges = `motivated_by`.
   Color by verdict (confirmed / refuted / inconclusive / pending) and flag underpowered ones.
2. **pass@1 curves**: Exp 2's intra-run curve (`runs/exp2_long/eval_curve.jsonl`) and the Exp 3
   chains (per-node `eval_passrate` along `c0..c3` / `d0..d3`), with the base anchor line.
3. **Checkpoint tree**: the `parent` structure of `ledger.json:checkpoints`.

## Why
"An agent that does RL science" only lands if you can *see* the inquiry tree with hypotheses and
verdicts. This is the artifact the demo is built around.

## Where / how
- New `dashboard.py` (or a small `dashboard/` static-HTML build). **Read** `ledger.json` and
  `runs/<id>/eval_curve.jsonl` / `runs/<id>/metrics.json`. **Do not** modify `ledger.py`'s schema.
- Reuse `Ledger` from `ledger.py` (`.checkpoints`, `.experiments`, `.inquiry_dag()`, `._trajectory(...)`).
- A teammate left a prototype `visualize_tree.py` + `tree.png` in the `worktrees/grpo` worktree
  (untracked) — check with them and build on it rather than starting cold.
- Output should be a committable static artifact (PNG/HTML), not a server the demo depends on.

## Definition of done
- `python dashboard.py` produces an HTML/PNG showing the inquiry DAG (with per-experiment verdict +
  power flag), the pass@1 curves, and the checkpoint tree — from the current `ledger.json`.
- Re-running after `scripts/backfill_ledger.py` (once Exp 3 syncs) updates `exp3` automatically.
- Looks good enough to put on a slide.
