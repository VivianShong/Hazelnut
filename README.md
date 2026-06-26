# Repurrr

## What we did

We gave a frontier agent one goal — make Qwen3.5-2B-Base write better Python — a single 16GB T4, and no further instructions. It ran its own research lab overnight: 5 experiments, zero human inputs after launch.

It pushed held-out MBPP pass@1 from 0.46 to 0.62. Along the way it hit a null result and diagnosed its own under-dosed learning rate, confirmed the fix, then when a deeper run regressed it backtracked to its best checkpoint and re-explored at a gentler setting. It even killed two of its own bad configs before spending GPU on them.

## How it operates

The agent forms a hypothesis, pre-registers the metric and a power check, and writes it to a ledger. A deterministic driver runs the GRPO training and scores it. The agent reads back a machine-computed verdict — confirmed, refuted, or inconclusive — and decides what to ask next.

It never touches the training code; it can only write to the ledger. Every run is a falsifiable claim in a growing inquiry graph, with dead ends, a frontier, and recovery.

## What it can scale to

**More compute:** the same loop runs deeper chains, parallel arms, and harder questions. Karpathy's agent ran 276 experiments across days on 8×H100; ours ran 5 on one T4. The ceiling is GPUs, not design.

**More capable models:** the substrate is swappable. Point it at any task with a deterministic evaluator and it researches that instead. The end of the line is the loop turned on the research agent itself — research that improves the thing doing the research.

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
