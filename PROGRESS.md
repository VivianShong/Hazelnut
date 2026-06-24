# GRPO post-training — progress log

_Last updated: 2026-06-23. Branch `grpo` (worktree `/opt/Hazelnut/worktrees/grpo`, off master `03cd578`)._

## Goal
**An autonomous research agent that does RL *science*** on a GRPO substrate (LoRA
post-training of **Qwen3.5-2B-Base** for Python code). The agent proposes **experiments**
(a hypothesis + controlled arms + a pre-registered metric & decision rule + a predicted
outcome), a deterministic driver runs the nodes, and the agent writes **verdicts**
(confirmed / refuted / inconclusive, with evidence) and follow-ups. The tree is an
**experiment ledger**, not a hill-climb to a champion checkpoint.

**Success metric = good science, not best pass@1:** correct dose/metric/regime conclusions,
adequately *powered* experiments, first-class negative results, and conclusions that survive
replication — per unit of compute. MBPP/APPS pass@1 are *substrates* to run experiments on,
not the objective. (Pivot 2026-06-24, after Exp 1-3 showed MBPP pass@1 saturates in ~40 steps
⇒ too shallow to justify a search tree as a checkpoint optimizer; but Exp 1→2→3 *was* exactly
this science loop run by hand — hypothesis → underpowered null → diagnose dose → confirm → re-test.)

_Earlier framing (kept as the substrate, demoted from headline): agent emits hyperparameter
configs, driver runs `train()`, keep/branch/rollback tree. Still true at the node level._

## Environment (verified)
- **GPU: Tesla T4, 16 GB, sm_75 (Turing)** — *not* an H100. No bf16 tensor cores; no flash-attn2/3.
- **Model: `/opt/Hazelnut/models/qwen3.5-2b-Base`** — a vision-language model
  (`Qwen3_5ForConditionalGeneration`, hybrid linear/full attention + MTP). We use the
  **text tower only**. Loads in bf16 (~3.8 GB). `qwen3_5` is natively supported by
  transformers ≥5.x.
- **nvcc 12.9** present (CUDA source builds work); triton 3.5.1.
- **Deps added** to this worktree's `pyproject.toml` (the repo's "no new deps" rule was
  for the pretraining `train.py`; bypassed here by decision): `transformers` (5.12.1),
  `peft`, `accelerate`, `datasets`, `flash-linear-attention` (fla-core), and
  `causal-conv1d` (compiled from source for sm_75).
- **SSM fast-path kernels installed** (fla + causal-conv1d). Warning is gone / fast path
  engages. NOTE: this speeds **prefill/training forward**, not single-token decode —
  generation stays ~20 tok/s on the T4 (decode is the loop bottleneck).

## Files (this worktree)
- `grpo.py` — model load + LoRA + hand-rolled GRPO loop. Exposes `train(**cfg)` + CLI. Prints a structured metrics block.
- `rewards.py` — deterministic reward: compile (`ast`) + correctness (sandboxed) + style (`ruff`, optional) + security (`bandit`, optional). Two grading modes: `tests` (assert-style, MBPP/dummy) and `io` (APPS stdin/stdout or fn-call). Returns scalar + components.
- `data.py` — `load_dataset(split, source=…)`: **mbpp** (default training), **apps** (benchmark; `difficulty=` filter), **dummy** (offline smoke). Caches MBPP to `data/mbpp`.
- `GRPO.md` — usage + hyperparameter list + design notes.
- `_passrate.py` — base-model pass@1 probe (`uv run python -u _passrate.py <source> <N> <max_new>`).

## Locked design decisions
- **Reference policy = base with LoRA adapter disabled** (`model.disable_adapter()`) — no 2nd model copy (fits 16 GB).
- **LoRA only** (`target_modules="all-linear"`, r=16) — base frozen. ~16.8 M trainable (0.89%).
- **Objective**: PPO-style clipped surrogate + per-token k3 KL penalty vs reference; group-normalised-reward advantages.
- **Memory**: gradient checkpointing + `PYTORCH_ALLOC_CONF=expandable_segments` + logprob micro-batch=1 in the grad pass (248k-vocab logits are the bottleneck). Peak ~9 GB.
- **Train on MBPP, benchmark on APPS** (curriculum/transfer story).

## Validation results
- Pipeline runs end-to-end on T4, no OOM (peak ~9 GB).
- Reward harness correct: MBPP reference solutions grade **6/6**; APPS reference (stdin) grades pass.
  - Fixed a harness bug: code was embedded into a Python string literal, which re-interpreted `\n`/`\t` and corrupted the payload. Now payloads pass via a `payload.json` the harness reads.
- **Base model pass@1** (do_sample, 1 sample/problem):
  - **MBPP**: compile 0.81, **pass_any 0.31**, mean_correct 0.31 → real learning signal + within-group variance.
  - **APPS introductory**: compile 0.58, **pass_any 0.00** → too hard for this model; only compile-rate is learnable. Keep as benchmark, not training set.
- Step-0 `kl=0` / `loss≈0` is **expected** (LoRA inits to zero ⇒ policy==reference; first on-policy pass has ratio≡1 ⇒ loss = −mean(normalised adv) ≈ 0). Gradient is still nonzero.

## Run
```bash
cd /opt/Hazelnut/worktrees/grpo
uv run python grpo.py                                   # MBPP defaults
uv run python grpo.py --source apps --difficulty introductory   # train on APPS (hard)
uv run python -u _passrate.py mbpp 16 384               # base pass@1 probe
```
Per-step log line: `step NNN | reward | correct | compile | kl | loss | elapsed`.
Final block: `final_reward / best_reward / mean_reward / steps / peak_vram_mb / checkpoint`.
LoRA adapter saved to `out/grpo/` (`adapter_model.safetensors`).

## Training result (MBPP, 8 steps, num_prompts=3 group_size=4, ~140 s/step)
- **Learning confirmed**: KL grows monotonically `0.0002 → 0.0010` (policy moving off reference); **compile rate climbs 0.83 → 1.00**.
- **Per-step reward is noisy** (0.43–0.96) — only 3 problems sampled per step, so it reflects *which* problems were drawn, not model quality. Need a **fixed held-out eval set** measured every N steps (and/or larger batches) for a clean curve. (`mbpp_train.log`)

## Experiment driver + tree (DONE — P0 core landed)
- `driver.py` — reads `tree.json`, claims `queued` nodes (fcntl lock + atomic write), runs
  `grpo.py` as a subprocess with the node's config, resumes from the parent's LoRA adapter
  via `--init-from`, records `done/failed` + metrics + checkpoint. Dedupe by config_hash;
  crash/timeout/no-metrics isolation per node. Agent owns config; driver owns results.
- `grpo.py` additions: `init_from` (resume parent adapter → real tree edges; KL reference
  stays = base), `evaluate()` (fixed held-out greedy pass@1 → comparable score across nodes),
  per-run `metrics.json` hand-off. Edge = GRPO-continue from parent; multi-child = many `parent`
  pointers to one immutable, retained adapter.

## Experiment 1 — KL strength × chain depth (`results/exp1_kl_depth.md`)
- 11 nodes, two depth-5 chains (kl=0.04 vs 0.01) from base, held-out pass@1 (n=24) per node.
- **Null result, and we know why:** pass@1 flat at base ±noise (n=24 → σ≈0.10; table spans
  0.33–0.50). KL-to-base only reached **~5e-4** after 25 cumulative steps ⇒ the policy barely
  moved ⇒ nothing to compound, and the KL penalty was never active (kl contrast inconclusive).
  compile saturated at 23/24 everywhere. **Per-node dose is far too small.**
- Infra validated end-to-end autonomously (resume chains, one transient OOM isolated + recovered).

## Experiment 2 — extended training on all MBPP (`results/exp2_extended_training.md`)
- One continuous 160-step run, **lr=5e-5** (5× exp1), all 374 MBPP train, pass@1 curve (n=50) every 20 steps.
- **YES — extended training moves pass@1: 0.42 → ~0.62 plateau (+0.20, ~3σ).** Exp 1's null was
  under-dosing: lr 5× pushed KL from ~5e-4 into the active **0.04–0.07** regime, and pass@1 moved.
- Gain is **front-loaded (most by step ~40)** then plateaus within noise; **compile saturates →1.00**;
  KL bounded/oscillating, no divergence. **Working recipe: lr≈5e-5, ~40–60 steps/node, eval n≥50.**
- Caveat: best checkpoint is mid-run, not final → tree should keep **best-eval**, not last.

## Next steps (priority order)
1. **Re-run Exp 1 (KL×depth) at the working dose** (lr=5e-5, ~40 steps/node, eval n≥50): now that a
   single node reaches the plateau, does chaining/depth or KL strength actually matter? Keep best-eval
   checkpoint per node (save per-eval snapshots to distinct dirs).
2. **Repoint agent scaffolding** (`program.md`, `.github/copilot-instructions.md`, `/self-improve` skill) from `train.py`/`val_bpb` → `grpo.py`/reward. (These still describe the pretraining task — stale for this project.)
4. **APPS-after-MBPP transfer eval** — does MBPP training move APPS pass@1 off 0? (headline result.)
5. **Live dashboard** — reward curve + version tree.

## Open caveats
- T4 is slow (~150 s/GRPO step at these sizes); few steps per 5-min node budget. Tune sizes vs throughput.
- Reward sandbox runs model-generated code in a subprocess with a timeout — **not** hardened isolation. Don't run untrusted output on a shared host without OS-level sandboxing.
- `train.py` / `prepare.py` at repo root are Karpathy's *pretraining* setup — orthogonal to this GRPO work; not on our path (don't clean up / delete).
