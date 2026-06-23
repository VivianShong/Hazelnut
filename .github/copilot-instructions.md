# Autoresearch — Project Guidelines

This repo is an **autonomous self-improvement loop**: an AI research agent edits a small
LLM training script, trains for a fixed budget, scores the result, and keeps or rolls back
— iterating like an ML research engineer to drive the benchmark metric down.

Treat every session as a controlled experiment. Act with **ML-engineer discipline**, not
brute-force search.

## Architecture
- **Research model (orchestrator)** — the disciplined experimentalist that reads hardware,
  forms one hypothesis at a time, edits the config, and decides keep/rollback. Its persona
  lives in [.github/agents/research.md](agents/research.md).
- **Student model** — the small local model being trained/fine-tuned. Its outputs are
  scored by a deterministic harness; the score (and components) feed back to the research
  model.
- **Operational spec** — `program.md` defines the experiment loop, fixed time budget,
  metric, and `results.tsv` logging. Read it before running the loop.

## Golden rules
- **Only edit `train.py`.** It holds the model, optimizer, and training loop — all knobs
  are fair game here.
- **Never modify `prepare.py`** (fixed constants, data prep, tokenizer, dataloader,
  evaluation) or the evaluation harness — it is the ground-truth metric.
- **No new dependencies.** Use only what is already in `pyproject.toml`.
- **Do not commit `results.tsv`** — leave it untracked.

## Research discipline (how to iterate)
- **Baseline first**: the first run is always the unmodified script.
- **One change per experiment**: never combine multiple variable changes in one run.
- **State the hypothesis** before editing: the change, why it should help, and the
  predicted effect — then compare against the *observed* effect.
- **Version as a tree**: one branch/commit per hypothesis; keep a frontier of best nodes;
  roll back to the best parent only when a branch clearly regresses (sparingly).
- **Dedupe**: don't retry configs already recorded in `results.tsv`. Prefer cheap heuristic
  search (coordinate descent, successive-halving, simple bandit) over grid/random.
- **Fit the hardware**: run `nvidia-smi` on setup and size the model/batch to available VRAM.
- **Reproducibility**: pin seeds; log hardware, git SHA, and full config with every run.

## Build and run
- Install deps: `uv sync`
- One-time data + tokenizer prep: `uv run prepare.py`
- Single experiment (~5 min): `uv run train.py > run.log 2>&1`
- Read the metric: `grep "^val_bpb:\|^peak_vram_mb:" run.log`

## Autonomy
Once the experiment loop has begun, run continuously until manually interrupted. Do not
pause to ask whether to continue (see the "NEVER STOP" section in `program.md`).
