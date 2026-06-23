# GRPO post-training — progress log

_Last updated: 2026-06-23. Branch `grpo` (worktree `/opt/Hazelnut/worktrees/grpo`, off master `03cd578`)._

## Goal
Autonomous GRPO post-training of **Qwen3.5-2B-Base** to write better Python code, on
top of Karpathy's autoresearch framework. P0: a research agent picks hyperparameter
configs, a deterministic driver runs `train()`, results form a keep/branch/rollback
tree. The agent only emits configs — it never edits training code.

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

## Next steps (priority order)
1. **Held-out eval** (`evaluate(model, n_problems)`) measured every N steps → clean pass@1 curve (replaces noisy per-step training reward as the tracked metric). Likely also bump `num_prompts` for lower-variance steps.
2. **Experiment driver + tree** (P0 core): wrap `train()`, parse the metrics block, persist nodes to `tree.json` (config + reward + checkpoint + status), keep/branch/rollback, crash/timeout handling, dedupe by config hash.
3. **Repoint agent scaffolding** (`program.md`, `.github/copilot-instructions.md`, `/self-improve` skill) from `train.py`/`val_bpb` → `grpo.py`/reward. (These still describe the pretraining task — stale for this project.)
4. **APPS-after-MBPP transfer eval** — does MBPP training move APPS pass@1 off 0? (headline result.)
5. **Live dashboard** — reward curve + version tree.

## Open caveats
- T4 is slow (~150 s/GRPO step at these sizes); few steps per 5-min node budget. Tune sizes vs throughput.
- Reward sandbox runs model-generated code in a subprocess with a timeout — **not** hardened isolation. Don't run untrusted output on a shared host without OS-level sandboxing.
- `train.py` / `prepare.py` at repo root are Karpathy's *pretraining* setup — orthogonal to this GRPO work; not on our path (don't clean up / delete).
