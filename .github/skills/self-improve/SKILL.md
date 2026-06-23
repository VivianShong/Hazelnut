---
name: self-improve
description: "Run the autoresearch self-improvement loop on this repo: read the hardware, form one hypothesis, change a single knob in train.py, train for the fixed 5-minute budget, score val_bpb, then keep or roll back — iterating like an ML research engineer. Use when asked to /self-improve, start the training loop, or autonomously optimize the model."
---

# /self-improve

Drive the autonomous training-research loop for this repository. Act with **ML-engineer
discipline**, not brute-force search.

## Before you start
1. Read [program.md](../../../program.md) — the operational spec (loop, fixed time budget,
   metric, `results.tsv` logging). It is authoritative for the loop mechanics.
2. Read [.github/copilot-instructions.md](../../copilot-instructions.md) for the golden
   rules (only edit `train.py`, never touch `prepare.py`, no new deps).
3. Prefer delegating the iteration to the **Research** agent
   ([.github/agents/research.md](../../agents/research.md)), which encodes this discipline.

## Setup (once, before the loop)
1. **Hardware**: `uv run python .github/skills/self-improve/tools/read_hardware.py` —
   note VRAM so the model and batch size fit the GPU.
2. **Dataset**: search the web for a dataset suited to the task + hardware, then fetch it:
   `uv run python .github/skills/self-improve/tools/download_dataset.py --repo <hf_dataset>`.
3. **Student model**: pick a small Qwen that fits VRAM and pull it:
   `uv run python .github/skills/self-improve/tools/pull_model.py --model Qwen/Qwen2.5-0.5B-Instruct`.

## Loop
1. **Baseline**: run the unmodified `train.py`, record it, commit.
2. **Hypothesis**: state the single change, why it should help, and the predicted effect.
3. **One change**: edit a single knob in `train.py`; commit on its own branch/commit.
4. **Train + score**: `uv run train.py > run.log 2>&1`, then read `val_bpb` /
   `peak_vram_mb` from `run.log`. No metric line = crash → read `tail -n 50 run.log`.
5. **Decide**: improved → keep (advance); equal/worse → `git reset` back. Log the row in
   `results.tsv` (untracked) with the observed effect. Don't retry configs already logged.
6. **Iterate**: prefer coordinate descent / successive-halving over grid/random; keep a
   frontier of best nodes.

## Helper tools (in `tools/`)
- `read_hardware.py` — GPU/VRAM/CPU/RAM summary for sizing the model.
- `download_dataset.py` — fetch the chosen dataset from the Hugging Face Hub.
- `pull_model.py` — pull the chosen small Qwen student model from the Hugging Face Hub.

## Autonomy
Once the loop has begun, keep going until interrupted — do not pause to ask whether to
continue (see "NEVER STOP" in program.md).

> Setup: `uv sync` then `uv run prepare.py` (one-time). To pre-approve shell commands for
> unattended runs, launch the CLI with `--allow-all-tools` (review the repo first).
