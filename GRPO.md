# GRPO post-training (worktree)

Hand-rolled GRPO that post-trains **Qwen3.5-2B-Base** (local, at
`../../models/qwen3.5-2b-Base`) to write better Python code, with LoRA adapters.
Built on the autoresearch convention: the research agent never edits this code —
it supplies a hyperparameter config to the exposed `train(...)` function.

## Files
- `grpo.py` — model load + LoRA + hand-rolled GRPO loop. Exposes `train(**cfg)` and a CLI.
- `rewards.py` — deterministic reward harness: compile + correctness (sandboxed tests)
  + style (ruff, optional) + security (bandit, optional). Returns a scalar **and** components.
- `data.py` — **placeholder** dummy problem set; swap in the real dataset behind `load_dataset()`.

## Run
```bash
uv run python grpo.py                                  # T4-friendly defaults
uv run python grpo.py --kl-coeff 0.02 --lr 2e-5 --group-size 8
```
Programmatic: `from grpo import train; train(kl_coeff=0.02)`

## Exposed hyperparameters (the knobs the research agent tunes)
`lr`, `kl_coeff`, `clip_eps`, `group_size`, `epochs_per_batch`, `num_prompts`,
`temperature`, `top_p`, `max_new_tokens`, `max_prompt_len`, `train_steps`,
`time_budget`, `lora_r`, `lora_alpha`, `lora_dropout`, `seed`, `logprob_micro_batch`.

## Design notes
- **Reference policy = base model with the LoRA adapter disabled** (`model.disable_adapter()`),
  so there's no second 4.5 GB model copy — fits the 16 GB T4 (smoke run peaked ~9.9 GB).
- Objective: PPO-style clipped surrogate + per-token k3 KL penalty vs the reference;
  advantages are group-normalised rewards.
- Logprobs are computed in micro-batches (`logprob_micro_batch`) to avoid materialising
  the full 248k-vocab logits at once.
- `train(...)` prints a structured summary block (`final_reward`, `best_reward`,
  `peak_vram_mb`, `checkpoint`, ...) for the experiment driver to parse.

## Environment / caveats
- Needs `transformers` (>=5.x, which natively supports the `qwen3_5` arch), `peft`, `accelerate`
  (added to this worktree's `pyproject.toml`). The base model is a VLM — only the text tower is used.
- SSM fast-path kernels (`flash-linear-attention`, `causal-conv1d`) are **not** installed;
  the linear-attention layers run a correct-but-slower torch fallback (~18 tok/s on T4).
  Generation is the loop bottleneck — keep `group_size` / `max_new_tokens` modest on T4.
- The reward sandbox runs model-generated code in a subprocess with a timeout. It is **not**
  hardened isolation — don't run untrusted output on a shared host without OS-level sandboxing.
