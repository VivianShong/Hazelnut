# Task 04 — Demo narrative + replication feature

**Owner:** _____   **GPU:** narrative none; replication can run mock or one small coordinated run
**Depends on:** `results/`, `ledger.json`, `ledger.py`

## Goal
Two things that turn the system into a *story*:
1. **`DEMO.md` storyboard** — the 3–5 beat arc, each beat anchored to a real ledger artifact:
   - Exp 1: agent runs a KL×depth experiment → **the power gate flags it underpowered**
     (n=24, MDE 0.20) and the verdict is INCONCLUSIVE — *the system catches its own bad experiment.*
   - Exp 2: agent diagnoses the cause (dose) → extended-training run → CONFIRMED, pass@1
     0.42→0.62, plateau@40.
   - Exp 3: agent re-tests KL×depth at the working dose → verdict (sync pending).
   - Punchline: the inquiry DAG `exp1 → exp2 → exp3` *is* the scientific method, automated.
2. **Replication feature** — re-run an experiment with a new seed and check the verdict holds
   (the "conclusions survive replication" half of the goal). A real differentiator: most "agent
   runs experiments" demos never replicate.

## Why
The headline is "an agent that does *science*." Catching its own underpowering and replicating its
conclusions are exactly what separate science from hill-climbing. This is the slide that wins.

## Where / how
- `DEMO.md`: pull numbers from `results/exp1_kl_depth.md`, `results/exp2_extended_training.md`,
  and `ledger.py show`. Coordinate with Task 01 for the inquiry-DAG screenshots.
- Replication: add a `ledger.py` helper that clones an experiment with `seed` varied in the arms'
  configs, runs it (via `ledger_driver --mock` for the demo, or a small real run), and reports
  whether `verdict.result` reproduces. Keep it small.

## Definition of done
- `DEMO.md` is a clean 3–5 beat storyboard with real numbers + DAG visual, runnable as the demo script.
- A `replicate(exp_id)` path spawns a seed-varied twin experiment and prints whether the verdict
  reproduces (confirmed-stays-confirmed, etc.).
