# Task 03 — Eval rigor: pass@k + bigger n, drop saturated compile

**Owner:** _____   **GPU:** logic is CPU; validating a run needs the card (coordinate)
**Depends on:** `grpo.py:evaluate`, `data.py`, `rewards.py`

## Goal
Sharper, more honest verdicts. Two changes:
1. **pass@k** alongside greedy pass@1 — sample `k` completions per problem at temperature, count
   "any correct". Coding quality is better captured by pass@k; it also un-saturates the signal.
2. **Bigger held-out n** (≥100) so the verdict engine's σ (binomial) tightens — Exp 1 was ruled
   INCONCLUSIVE precisely because n=24 → σ≈0.10. Drop `compile` as a tracked metric (it pins at 1.0).

## Why
The verdict engine's power gate is only as good as the eval. Tighter n + pass@k = experiments that
can actually resolve the effects the agent predicts (the whole point of the "powered science" story).

## Where / how
- `grpo.py:evaluate()` currently does greedy (`do_sample=False`) pass@1 on a fixed slice. Add an
  `eval_k` config: for k>1 sample k per problem (temperature), report `eval_pass@k` (any-correct)
  plus mean. Keep greedy pass@1 for determinism/comparability. Return both in `metrics`.
- `rewards.py:score_completion` already returns correctness + `n_passed/n_tests` — reuse it.
- `data.py:load_dataset("eval", source="mbpp")` is the held-out MBPP-test slice; bump the count.
- Keep eval **deterministic where it matters** (fixed problem order; fixed seed for the k samples)
  so scores stay comparable across checkpoints/nodes.
- Update `ledger.py` metric handling if you add a new metric name (the verdict engine reads
  `metric["name"]` from `checkpoints[id].metrics`).

## Definition of done
- `evaluate()` returns `eval_pass@k` for a configurable `k` and a larger `n`; a quick run shows
  pass@k ≥ pass@1 and a tighter σ in `power_check`.
- The ledger can register an experiment whose `metric.name` is the new pass@k field and verdict it.
