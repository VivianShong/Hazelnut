# Task 02 — Unify the driver into the ledger (one system of record)

**Owner:** _____   **GPU:** dev with `--mock`; final smoke needs the card (coordinate)
**Depends on:** `ledger.py`, `ledger_driver.py`, `driver.py`

## Goal
Make the real GRPO driver write **directly into `ledger.json`**, retiring the parallel
`tree.json` + `scripts/backfill_ledger.py` bridge. After this, a queued checkpoint in the ledger
runs, its metrics land in `ledger.checkpoints`, and any experiment it completes auto-evaluates —
no back-fill step.

## Why
Right now there are two systems: experiments run on the old `driver.py`/`tree.json` and get
*back-filled* into the ledger. That's a demo smell ("which file is real?") and a footgun (two
drivers could double-run a node). One system of record fixes both.

## Where / how
- `ledger_driver.py` already has the forward loop + `_execute_real` (runs `grpo.py` with
  `--init-from <parent>`). Port the **robustness** from `driver.py` into it:
  - fcntl lock + atomic write around ledger mutations (see `driver.py:_locked/_write_tree`),
  - subprocess timeout, crash / nonzero-exit / missing-`metrics.json` → `failed` + reason,
  - `config_hash` dedupe, and **keep the `external`-status guard** (never claim nodes another
    driver owns — see `ledger.py` status semantics in `LEDGER.md`).
- Add the status transitions to `ledger.py` if cleaner (queued→running→done/failed).
- Leave `driver.py`/`tree.json` in place until Exp 3 finishes; this is the replacement for the *next* runs.

## Definition of done
- `uv run python ledger_driver.py` (no `--mock`) takes a `queued` checkpoint in `ledger.json`,
  runs `grpo.py`, records `done`+metrics (or `failed`+reason), and auto-evaluates the experiment —
  with locking + timeout + crash isolation.
- `--mock` path still works; a mock multi-node experiment runs end-to-end and verdicts compute.
- `scripts/backfill_ledger.py` is no longer needed to record *new* experiments (only for the
  one-time Exp 3 sync).
