# Autoresearch — Project Guidelines

This repo is an **autonomous self-improvement loop**: an AI research agent chooses a
hyperparameter config, runs a fixed-budget training experiment via an exposed `train(...)`
function, scores the result, and keeps or rolls back — iterating like an ML research
engineer to drive the benchmark metric down.

Treat every session as a controlled experiment. Act with **ML-engineer discipline**, not
brute-force search.

## Architecture
- **Research model (orchestrator)** — the disciplined experimentalist that reads hardware,
  forms one hypothesis at a time, picks a hyperparameter config, and decides keep/rollback.
  Its persona lives in [.github/agents/research.md](agents/research.md).
- **Student model** — the small local model being trained/fine-tuned. Its outputs are
  scored by a deterministic harness; the score (and components) feed back to the research
  model.
- **Operational spec** — `program.md` defines the experiment loop, fixed time budget,
  metric, and `results.tsv` logging. Read it before running the loop.

## Golden rules
- **Don't edit the training code; supply a config.** The research agent never edits
  `train.py` or `prepare.py`. Each experiment is a hypothesis + a hyperparameter config
  passed to the exposed `train(param1, param2, ...)` function in `train.py`.
- **`prepare.py` and the evaluation harness are off-limits** (fixed constants, data prep,
  tokenizer, dataloader, evaluation) — they are the ground-truth metric.
- **No new dependencies.** Use only what is already in `pyproject.toml`.
- **Do not commit `results.tsv`** — leave it untracked.

## Research discipline (how to iterate)
- **Baseline first**: the first run uses the default config.
- **One change per experiment**: never combine multiple variable changes in one run.
- **State the hypothesis** before each run: the change, why it should help, and the
  predicted effect — then compare against the *observed* effect.
- **Version as a tree**: one node per config; keep a frontier of best nodes; roll back to
  the best parent only when a branch clearly regresses (sparingly). Every node trains from
  the **same fixed base model** — nodes branch on the *config*, not on a continued
  checkpoint — so runs stay directly comparable. Only the trained models at the
  **leaf/frontier nodes** are exported.
- **Dedupe**: don't retry configs already recorded in `results.tsv`. Prefer cheap heuristic
  search (coordinate descent, successive-halving, simple bandit) over grid/random.
- **Fit the hardware**: run `nvidia-smi` on setup and size the model/batch to available VRAM.
- **Reproducibility**: pin seeds; log hardware, git SHA, and full config with every run.

## Build and run
- Install deps: `uv sync`
- One-time data + tokenizer prep: `uv run prepare.py`
- Single experiment (~5 min): call the exposed `train(...)` with the chosen config; output
  goes to `run.log`.
- Read the metric: `val_bpb` / `peak_vram_mb` from `run.log`.

## Autonomy
Once the experiment loop has begun, run continuously until manually interrupted. Do not
pause to ask whether to continue (see the "NEVER STOP" section in `program.md`).
