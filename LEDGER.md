# Experiment ledger (v2 schema) — the science-engine system of record

Headline structure: **inquiry DAG over a checkpoint substrate**. Three nested ideas:

```
checkpoints (facts)  ⊂  experiments (claims over node-sets, via selectors)  ⊂  inquiry DAG
```

`ledger.json` has two sections:

- **`checkpoints`** — facts. `{id -> parent, config, status, metrics, checkpoint}`. A
  checkpoint is a *point* (weights from one GRPO run resumed from `parent`). Status
  `done` (eval recorded), `queued` (ledger_driver may run it), `external` (owned by
  another/live driver — never claimed here), `running`, `failed`.
- **`experiments`** — claims. Each has a `hypothesis`, a **selector** that *references*
  (never owns) a set of checkpoints, a pre-registered `metric` + `decision_rule` +
  `predicted` finding, a `power_check`, `motivated_by` (inquiry-DAG edge), and a
  machine-computed `verdict`. A checkpoint may belong to many experiments → selectors
  reference, they don't contain (many-to-many).

## Selectors (the structural shape the hypothesis is about)
| type | resolves to | use |
|---|---|---|
| `{"type":"point","node":X}` | one checkpoint | single run vs anchor |
| `{"type":"path","tip":X}` | chain root→X (parent-walk) | trend over cumulative steps |
| `{"type":"curve","node":X}` | one run's intra-node eval curve | trend within a single run |
| `{"type":"contrast","arms":{name:<selector>}}` | union of arms | compare KL / depth / lr |

`resolved_nodes` is **cached** on register and refreshed before each evaluate. The
segment-tree intuition lives in `path`/`curve`: a trend is a *segment over the step
axis*, and the verdict engine runs range queries on it (argmax checkpoint, **plateau
onset**, delta-vs-anchor).

## Decision rules → verdict
- `trend_vs_anchor` — best point on the trajectory vs anchor; `finding ∈ {improved, regressed, flat}`,
  reports `plateau_onset_step`. (Exp 2 → improved, Δ0.24, plateau@40.)
- `contrast_at_tip` — arm tip difference vs `k·σ` (pooled binomial); `finding ∈ {arm_effect, no_arm_effect}`.
  (Exp 1 → no_arm_effect, but INCONCLUSIVE: underpowered.)

`verdict.result` = **confirmed** (finding == predicted) / **refuted** (contradicts) /
**inconclusive** (effect within noise → underpowered) / **pending** (nodes unfinished).
`power_check` reports σ and the min detectable effect at 2σ — this is what auto-flagged
Exp 1's n=24 as too coarse (MDE 0.204).

## Run it
```bash
uv run python scripts/backfill_ledger.py   # ingest real runs -> ledger.json, compute verdicts
uv run python ledger.py show               # inquiry DAG + verdicts
uv run python ledger_driver.py --mock      # live loop on queued checkpoints, no GPU
uv run python ledger_driver.py             # real: runs grpo.py per queued checkpoint
```

## Syncing the live Exp 3
Exp 3 runs on the *old* `tree.json`/`driver.py`. Its nodes are ingested as `external`
(not claimed here). When it finishes, re-run `backfill_ledger.py`: nodes with a recorded
`eval_passrate` flip to `done` and the `exp3` verdict computes. (Migration target: the
old driver writes straight into the ledger; until then, back-fill is the bridge.)
