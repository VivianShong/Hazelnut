# Hazelnut — an autonomous RL-science agent

An LLM research agent that does **RL science** on a GRPO substrate. It post-trains
**Qwen3.5-2B-Base** to write better Python (MBPP), but the headline is the *scientific loop*,
not a single best checkpoint: the agent forms **hypotheses**, pre-registers **experiments**
(with a power check), runs them on a deterministic GRPO + reward substrate, and writes
**verdicts** that spawn the next hypothesis. Built on Karpathy's autoresearch framework.

**Success = good science**, not best pass@1: correct, *adequately-powered* conclusions;
first-class negative results; verdicts that survive replication — per unit of compute.
MBPP pass@1 is the *substrate*, not the objective.

> Status & findings live in `PROGRESS.md`. Exp 1–3 already ran this loop by hand:
> hypothesis → underpowered null → diagnose dose → confirm → re-test.

## Two-model system
- **Research agent** (large hosted model, runs as a Claude Code skill in `.github/skills/`):
  forms hypotheses, designs experiments (config dicts + arms), reads results, writes verdicts.
  **Never edits training code.** Tools: bash (hardware probe), the ledger.
- **Trainee** (Qwen3.5-2B-Base, local GPU): post-trained with GRPO/LoRA; emits code scored by a
  **deterministic** reward harness (not the LLM).

## The ledger — `outputs/ledger.json` (what the agent reads/writes)
`Goal → Hypotheses ⇄ Experiments`, linked both ways (an experiment's result feeds back into
*multiple* hypotheses).

```jsonc
{
  "goal": "Improve Qwen3.5-2B-Base pass@1 on MBPP",
  "hypotheses": [
    { "h_id": 3, "idea": "Once KL is active, its strength (0.04 vs 0.01) doesn't change the plateau",
      "status": "refuting", "derived_from": 2, "next_h": 4,
      "ref_e_ids": [3], "feedback": "e3 (partial): kl0.04 ≈0.56–0.62 vs kl0.01 ≈0.44" }
  ],
  "experiments": [
    { "e_id": 3, "ref_h_ids": [3, 4],
      "experiment_setting": "KL contrast 0.04 vs 0.01, depth-4 chains @ lr5e-5, 40 steps/node, eval n=50",
      "tree_ref": { "run": "run-001", "nodes": ["c0","c1","c2","c3","d0","d1","d2","d3"] },
      "status": "incomplete",
      "result": { "verdict": "pending", "partial": "kl0.04 > kl0.01 at depth≤2" } }
  ]
}
```

## Three layers (why / what / receipts)
- **`outputs/ledger.json`** — *why*: hypotheses + experiments. Global, committed.
- **`outputs/runs/<run>/tree.json`** — *what*: checkpoint nodes (configs, status, metrics).
  **Per-run** (each driver invocation owns its own → no cross-driver collisions). Committed.
- **`outputs/runs/<run>/<node>/logs/`** — *receipts*: `train.log`, `metrics.json`,
  `eval_curve.jsonl`, `metrics_steps.jsonl`. Small text, committed.
- **`outputs/runs/<run>/<node>/model_checkpoint/`** — LoRA adapters (~67 MB). **gitignored**.

The agent reads the structured layers (ledger + tree) by default and cracks open raw `logs/`
only for forensics.

## Experiment model: branch / continue / stop
Each node trains from its **parent's checkpoint** (root = base model).
- **continue** — same config, more steps, **carries optimizer state**.
- **branch** — new config from the parent checkpoint, **resets optimizer**.
- **stop** — prune a dead end.

Reference policy = **frozen base** (KL anchor), never updated. Configs deduped by content hash.
Single GPU, one experiment at a time.

## Reward (deterministic)
`compile` (ast) + `correctness` (sandboxed tests) + `style` (ruff) + `security` (bandit)
→ scalar reward + components logged separately for agent reasoning. (Note: `compile` saturates
early; `pass@1`/`correctness` is the live signal — see `PROGRESS.md`.)

## Repo structure
```text
.
├── main.py                 # entrypoint: runs the research loop
├── grpo.py                 # GRPO/LoRA training engine (CLI; agent never edits)
├── rewards.py              # deterministic reward harness
├── data.py                 # MBPP / APPS loaders
├── driver.py               # executes a run's tree.json nodes (subprocess → grpo.py)
├── ledger.py               # ledger lib: hypotheses/experiments + verdict/power engine
├── dashboard.py            # inquiry DAG + pass@1 curves (GUI)
├── .github/skills/repur/   # the research agent, as a Claude Code skill
│   ├── skill.md
│   └── tools/
├── relics/                 # Karpathy's original autoresearch (out of scope, untouched)
├── outputs/
│   ├── ledger.json         # global research log              (committed)
│   └── runs/
│       └── run-001/
│           ├── tree.json   # this session's checkpoint nodes  (committed)
│           └── node-a0/
│               ├── model_checkpoint/   # LoRA adapter          (gitignored)
│               └── logs/               # train.log, metrics.json, *.jsonl (committed)
└── README.md
```
Modules sit flat at root because `grpo.py` imports `rewards.py`/`data.py` (welded), and
`ledger.py` is shared infra (`driver.py` writes it, `dashboard.py` reads it) — so it can't live
inside the agent skill.
