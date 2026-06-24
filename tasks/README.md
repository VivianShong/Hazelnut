# Work items — day 3/5 hackathon

**Headline goal:** an autonomous research agent that does *RL science* on a GRPO substrate
(see `PROGRESS.md` "Goal" + `LEDGER.md`). Success = good science (correct, *powered*
conclusions; first-class negative results; replication), **not** best pass@1.

## ⚠️ The one constraint that shapes everything
**One T4 GPU, serialized, ~occupied for ~9h by Exp 3 (the `c*`/`d*` chains).** Only one GRPO
run fits at a time. Therefore:
- **The day-5 demo must NOT depend on new large training runs.** It demos the *system* (agent +
  ledger + dashboard) over experiments we already have (Exp 1/2/3 in `ledger.json`) plus small/mock runs.
- **Develop GPU-free.** Almost every task below works against `ledger.json` + `ledger_driver.py --mock`.
  Only do real GRPO runs after coordinating (Exp 3 owns the card until it drains).

## State you're building on
- `grpo.py` — GRPO substrate: `train()`, `evaluate()` (held-out greedy pass@1), `--init-from`
  (resume parent adapter), `--eval-every` (pass@1 curve → `runs/<id>/eval_curve.jsonl`), `metrics.json`.
- `ledger.py` — `Ledger`: `checkpoints` (facts) ⊂ `experiments` (claims via **selectors**
  point/path/curve/contrast, cached `resolved_nodes`) ⊂ inquiry DAG (`motivated_by`). Verdict engine
  → confirmed/refuted/inconclusive/pending + `power_check`. `ledger.py show` prints the DAG.
- `ledger_driver.py` — forward loop (queued checkpoint → execute → auto-verdict); `--mock` = no GPU.
- `driver.py` + `tree.json` — the *old* checkpoint driver (Exp 3 runs on this); being unified into the ledger (Task 02).
- Findings: dose lesson (lr 5e-5, ~40 steps), pass@1 0.42→0.62 plateau@40, compile saturates.

## Tasks (4 parallel, all GPU-free to develop)
| # | Task | File | Touches |
|---|---|---|---|
| 01 | Dashboard (inquiry DAG + curves) | `01-dashboard.md` | new `dashboard*`; reads `ledger.json` |
| 02 | Driver → ledger unification | `02-driver-ledger-unification.md` | `ledger_driver.py`, `driver.py`, `ledger.py` |
| 03 | Eval rigor (pass@k, bigger n) | `03-eval-rigor.md` | `grpo.py:evaluate`, `data.py` |
| 04 | Demo narrative + replication | `04-demo-narrative-replication.md` | `results/`, new `DEMO.md`, `ledger.py` |

**Not in this list (owned by lead + Claude):** the autonomous agent loop itself — the LLM that
reads the ledger, proposes a pre-registered experiment (must pass the power gate), runs it, and
writes the verdict. That's the crown jewel; these four tasks make it demoable.
