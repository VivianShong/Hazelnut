---
name: repur
description: "Run the autonomous GRPO post-training research loop on this repo end-to-end: read hardware, establish a baseline, form ONE hypothesis, write a config + experiment to ledger.json, let the driver run GRPO and score held-out pass@1, read the machine verdict, then branch/continue/stop — iterating like an ML research engineer over the inquiry DAG. Use when asked to /repur, start the training loop, or autonomously optimize the model on MBPP."
---

# /repur

Drive the autonomous **GRPO post-training** research loop for this repo. Act with
**ML-engineer discipline**, not brute-force search. You navigate hyperparameter space to
post-train Qwen3.5-2B-base to write better Python code; a deterministic driver runs the
training and a deterministic harness scores it — **you only emit config dicts and
experiment claims, never training code.**

## The system of record: `ledger.json`
You should only write to ledger.json.
Two layers:
- **checkpoints** — facts: `{id → parent, config, status, metrics, checkpoint}`. A
  checkpoint is one GRPO run resumed from its parent (root resumes from base).
- **experiments** — claims: a hypothesis + a **selector** over checkpoints + a
  pre-registered metric/decision-rule/`predicted` finding + a `power_check`, linked into an
  **inquiry DAG** by `motivated_by`, with a machine-computed `verdict`.

**Read and write the ledger only through its CLI — never hand-edit `ledger.json`:**
```bash
uv run python ledger.py show                       # inquiry DAG + verdicts (human view)
uv run python ledger.py read --section summary     # goal, states, frontier, verdicts (model view)
uv run python ledger.py read --section checkpoints --id a3   # one node's full record
uv run python ledger.py goal                       # print the standing goal
uv run python ledger.py goal --set "<objective>"   # set it (once, at init)
uv run python ledger.py frontier --top 5           # best done checkpoints by eval_passrate

uv run python ledger.py add-checkpoint --id <id> --parent <pid> \
    --config '{"train_steps":40,"lr":5e-5,"kl_coeff":0.04,"eval_n":50}'   # queue a run
uv run python ledger.py register --id <eid> --question "..." --hypothesis "..." \
    --predicted improved --selector '{"type":"path","tip":"<id>"}' \
    --metric '{"name":"eval_passrate","n":50}' \
    --decision-rule '{"type":"trend_vs_anchor","k_sigma":2}' \
    --anchor <anchor_id> --motivated-by <parent_eid>     # pre-register a claim
uv run python ledger.py evaluate --id <eid>        # (or --all) recompute verdict(s)
```
**Write the hypothesis to the ledger BEFORE you train, not after.** Every run must
correspond to a `register`ed experiment (its hypothesis) and the `add-checkpoint` node(s) it
selects. A run that exists only under `outputs/runs/` with no ledger entry is a lost experiment.

Selectors: `point` (one node vs anchor), `path` (root→tip chain, trend over cumulative
steps), `curve` (one run's intra-node eval curve), `contrast` (arms, e.g. KL 0.04 vs 0.01).
Decision rules: `trend_vs_anchor` → `{improved,regressed,flat}`; `contrast_at_tip` →
`{arm_effect,no_arm_effect}`. The `power_check` reports the min detectable effect at 2σ —
**heed it**: if your `eval_n` can't resolve the effect you predict, raise it before running.

## Setup (once, before the loop)
1. **Hardware**: `uv run python .github/skills/repur/tools/read_hardware.py` — note
   VRAM and fit `num_prompts` / `group_size` / `max_new_tokens` / `eval_batch` to the GPU.
2. **Deps + data** (one-time): `uv sync`. MBPP is the default training/eval source and
   `data.py` caches it locally on first load — no separate prep step. For a new dataset use
   `.github/skills/repur/tools/download_dataset.py --repo <hf_dataset>`.
3. **Goal**: `uv run python ledger.py goal --set "improve Qwen3.5-2B-base's pass rate on MBPP dataset"`.

## Loop (the README Workflow, automated)
1. **Init / baseline.** If no `done` baseline exists, queue one with `train_steps=0`
   (eval-only) at your eval budget and run it — this is the anchor every claim measures
   against:
   ```bash
   uv run python ledger.py add-checkpoint --id base50 --config '{"train_steps":0,"eval_n":50}'
   uv run python driver.py                   # runs queued nodes; --mock for a no-GPU dry run
   ```
2. **Hypothesize.** State ONE change, why it should help, and the predicted direction. Encode
   it as a config dict — one knob off the parent (controlled ablation). Common knobs:
   `lr`, `kl_coeff`, `train_steps`, `group_size`, `temperature`, `eval_n`.
3. **Write to the ledger.** `add-checkpoint` the config node(s) (`status` defaults to
   `queued`), then `register` the experiment that selects them, with its anchor, metric, and
   decision rule. A continued chain = several nodes where each `--parent` is the previous tip
   (a `path` selector); a comparison = a `contrast` of two arms.
4. **Run.** `uv run python driver.py`. The driver pops each runnable queued node
   (parent `done`), runs `grpo.py` from the parent's checkpoint as a subprocess (root from
   base), writes `metrics.json`, sets the node `done`, regenerates the per-run `tree.json`,
   and **auto-evaluates** any experiment whose nodes are all done. No metric line / crash →
   inspect `outputs/runs/<run>/<id>/logs/train.log`.
5. **Evaluate.** Read the verdict: `uv run python ledger.py read --section experiments --id <eid>`.
   `confirmed` (finding == predicted) / `refuted` / `inconclusive` (within noise →
   underpowered: raise `eval_n` and re-run) / `pending` (nodes unfinished). Components
   (compile/correctness/style/security) are in each node's `metrics`.
6. **Decide.** Read `ledger.py frontier` and the verdicts, then:
   - **branch** — promising node → new config from that parent (fresh GRPO, optimizer reset),
   - **continue** — node still improving → same config, more steps from its tip (a deeper
     `path`), optimizer carried,
   - **stop** — dead end → leave it; don't queue children.
   Dedupe by config (don't re-queue a config already `done`). Keep a frontier of best nodes.
   Each new experiment should set `--motivated-by` to the one that prompted it, growing the
   inquiry DAG.
7. **Loop** until the held-out pass@1 plateaus or the budget is exhausted. Inspect progress
   on the live dashboard (`uv run python dashboard.py`) — inquiry DAG + pass@1 curves.
8. **Export.** Only frontier/leaf models get exported:
   `uv run python .github/skills/repur/tools/export_model.py --checkpoint outputs/runs/<run>/<frontier_id>/model_checkpoint`.

## Helper tools (in `tools/`)
- `read_hardware.py` — GPU/VRAM/CPU/RAM summary for sizing the run.
- `download_dataset.py` — fetch a training/eval dataset from the Hugging Face Hub.
- `pull_model.py` — pull the base/student Qwen model from the Hub.
- `export_model.py` — export a trained frontier/leaf model with its config/score metadata.

## Autonomy
Once the loop has begun, keep going until interrupted — do not pause to ask whether to
continue. One change per experiment; never skip the baseline; never edit `grpo.py`,
`driver.py`, `data.py`, or the reward harness; never hand-edit `ledger.json` or `tree.json`
(the driver generates `tree.json`).

> To pre-approve shell commands for unattended runs, launch the CLI with `--allow-all-tools`
> (review the repo first). For a no-GPU rehearsal of the orchestration, use
> `uv run python driver.py --mock`.
