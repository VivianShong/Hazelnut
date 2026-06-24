"""Hand-rolled GRPO post-training for Qwen3.5-2B-Base on a Python-coding task.

Single-GPU, single-file. The research agent never edits this file; it supplies a
hyperparameter config to the exposed `train(...)` function (or via CLI flags).

GRPO (Group Relative Policy Optimization), critic-free:
  - For each prompt, sample G completions from the current policy.
  - Score each with the deterministic reward harness (rewards.py).
  - Advantage = group-normalised reward (r - mean) / (std + eps), broadcast to
    every token of that completion.
  - Update the policy with a PPO-style clipped surrogate plus a per-token KL
    penalty (k3 estimator) against a frozen reference policy.

Reference policy = the SAME model with the LoRA adapter DISABLED. This avoids a
second 4.5 GB model copy, which matters on a 16 GB T4.

Only LoRA adapter weights are trained; the base model stays frozen. The exposed
`train(...)` returns a metrics dict and prints a structured summary block that
the experiment driver parses.

Usage:
  uv run python grpo.py                         # defaults (T4-friendly)
  uv run python grpo.py --kl-coeff 0.02 --lr 2e-5
  from grpo import train; train(kl_coeff=0.02)  # programmatic
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Reduce allocator fragmentation on the small (16 GB) T4 before torch loads CUDA.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

from data import load_dataset
from rewards import score_completion

# Local base model (Qwen3.5-2B-Base, a VLM — we use the text tower only).
# Absolute path: stable whether this file lives in the worktree or repo root after merge.
DEFAULT_MODEL_PATH = "/opt/Hazelnut/models/qwen3.5-2b-Base"


@dataclass
class GRPOConfig:
    # --- model / io ---
    model_path: str = DEFAULT_MODEL_PATH
    output_dir: str = "out/grpo"
    init_from: str = ""            # parent LoRA adapter dir to resume from; "" = fresh from base
    seed: int = 42
    # --- LoRA (what we actually train) ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # --- GRPO objective ---
    lr: float = 1e-5
    kl_coeff: float = 0.04         # beta: KL penalty vs frozen reference
    clip_eps: float = 0.2          # PPO ratio clip
    group_size: int = 4            # G completions per prompt
    epochs_per_batch: int = 1      # inner optimisation passes per rollout (mu)
    # --- sampling / rollout ---
    num_prompts: int = 4           # prompts per GRPO step
    temperature: float = 0.9
    top_p: float = 1.0
    max_new_tokens: int = 512       # solutions need room; <256 truncates -> won't compile
    max_prompt_len: int = 1024      # APPS statements are long (700+ tok); don't truncate them away
    data_limit: int = 256           # cap on training problems materialised (raise to use all of MBPP)
    source: str = "mbpp"            # training set: mbpp|apps|dummy (APPS is the hard benchmark)
    difficulty: str = "introductory"  # APPS only: introductory|interview|competition|"" (all)
    max_test_cases: int = 8        # APPS: cap test cases graded per completion
    # --- held-out eval (clean comparable score across tree nodes) ---
    eval_n: int = 0                # held-out problems for greedy pass@1; 0 = skip
    eval_every: int = 0            # also eval mid-training every N steps (pass@1 curve); 0 = only at end
    eval_batch: int = 8            # generation batch size during eval
    # --- budget (stop at whichever comes first) ---
    train_steps: int = 20
    time_budget: float = 300.0     # seconds (matches program.md)
    # --- logprob micro-batching (keep the 248k-vocab logits from OOMing) ---
    logprob_micro_batch: int = 2


def _set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_policy(cfg: GRPOConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = True

    if cfg.init_from:
        # Tree branch: continue GRPO from a parent node's LoRA adapter. The base
        # stays frozen; the reference policy is still base-with-adapter-disabled
        # (KL is measured vs base, not vs the parent — see GRPO.md design notes).
        print(f"Resuming LoRA adapter from {cfg.init_from}")
        model = PeftModel.from_pretrained(model, cfg.init_from, is_trainable=True)
    else:
        lora = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    # Needed for gradient checkpointing to propagate grads to LoRA params while
    # the base model stays frozen.
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tok


@torch.no_grad()
def _generate(model, tok, prompts: list[str], cfg: GRPOConfig):
    """Sample G completions per prompt. Returns (sequences, prompt_len, gen_mask).

    All prompts are left-padded to a common length, so the completion region is
    a fixed slice [prompt_len:] across the batch. gen_mask marks real (non-pad)
    completion tokens.
    """
    chats = [[{"role": "user", "content": p}] for p in prompts]
    enc = tok.apply_chat_template(
        chats, add_generation_prompt=True, return_tensors="pt",
        return_dict=True, padding=True, truncation=True, max_length=cfg.max_prompt_len,
    ).to("cuda")
    prompt_len = enc["input_ids"].shape[1]

    model.eval()
    model.config.use_cache = True          # fast autoregressive decode
    model.gradient_checkpointing_disable()  # no grad here; keep decode fast
    out = model.generate(
        **enc,
        do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens, num_return_sequences=cfg.group_size,
        pad_token_id=tok.pad_token_id,
    )
    # out: (num_prompts * group_size, prompt_len + gen_len)
    gen = out[:, prompt_len:]
    gen_mask = (gen != tok.pad_token_id).long()
    return out, prompt_len, gen_mask


def _seq_logprobs(model, sequences, prompt_len, gen_mask, micro_batch):
    """Per-token logprobs of the completion region, in micro-batches.

    Returns a tensor (N, gen_len) of log p(token_t | <t) for completion tokens
    (zeros where masked). Gradients flow iff model is in grad mode.
    """
    N = sequences.shape[0]
    all_lp = []
    for i in range(0, N, micro_batch):
        seq = sequences[i:i + micro_batch]
        logits = model(seq).logits  # (b, T, V)
        # predict token t from logits at t-1
        logits = logits[:, prompt_len - 1:-1, :]          # (b, gen_len, V)
        targets = seq[:, prompt_len:]                      # (b, gen_len)
        logp = F.log_softmax(logits.float(), dim=-1)
        tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (b, gen_len)
        all_lp.append(tok_lp)
    return torch.cat(all_lp, dim=0) * gen_mask


@torch.no_grad()
def evaluate(model, tok, cfg: GRPOConfig) -> dict:
    """Greedy pass@1 on a FIXED held-out MBPP slice (disjoint from training).

    Deterministic (do_sample=False, fixed problem order) so the score is
    comparable across tree nodes/checkpoints — this is the metric the frontier
    should rank on, unlike the noisy per-step training reward. Evaluates the
    *current* policy (LoRA adapter active).
    """
    probs = load_dataset("eval", source=cfg.source, limit=cfg.eval_n,
                         difficulty=(cfg.difficulty or None))
    if not probs:
        return {}
    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    chats = [[{"role": "user", "content": p["prompt"]}] for p in probs]
    enc = tok.apply_chat_template(
        chats, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        padding=True, truncation=True, max_length=cfg.max_prompt_len,
    ).to("cuda")
    plen = enc["input_ids"].shape[1]
    texts: list[str] = []
    for i in range(0, len(probs), cfg.eval_batch):
        sub = {k: v[i:i + cfg.eval_batch] for k, v in enc.items()}
        out = model.generate(**sub, do_sample=False, max_new_tokens=cfg.max_new_tokens,
                             pad_token_id=tok.pad_token_id)
        texts.extend(tok.batch_decode(out[:, plen:], skip_special_tokens=True))
    npass = ncomp = 0
    for text, prob in zip(texts, probs):
        bd = score_completion(text, prob, max_cases=cfg.max_test_cases)
        npass += 1 if bd.correctness > 0 else 0
        ncomp += 1 if bd.compiles else 0
    n = len(probs)
    return {"eval_passrate": npass / n, "eval_compile": ncomp / n, "eval_n": n}


def train(**overrides) -> dict:
    """Run GRPO with the given hyperparameter overrides. Returns a metrics dict."""
    cfg = GRPOConfig(**overrides)
    _set_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    t_start = time.time()

    model, tok = _load_policy(cfg)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
    )

    train_problems = load_dataset("train", source=cfg.source, limit=cfg.data_limit,
                                  difficulty=(cfg.difficulty or None))
    print(f"Loaded {len(train_problems)} training problems (source={cfg.source})")

    eps = 1e-4
    rng = torch.Generator().manual_seed(cfg.seed)
    step = 0
    reward_hist: list[float] = []
    eval_curve: list[dict] = []
    last_breakdown: dict | None = None

    while step < cfg.train_steps and (time.time() - t_start) < cfg.time_budget:
        # --- sample a batch of prompts ---
        idx = torch.randint(0, len(train_problems), (cfg.num_prompts,), generator=rng)
        batch = [train_problems[int(i)] for i in idx]
        prompts = [b["prompt"] for b in batch]

        # --- rollout: G completions per prompt ---
        seqs, prompt_len, gen_mask = _generate(model, tok, prompts, cfg)
        gen_text = tok.batch_decode(seqs[:, prompt_len:], skip_special_tokens=True)

        # --- score completions; build group-normalised advantages ---
        rewards = torch.zeros(len(gen_text))
        comps = {"correctness": [], "compiles": [], "style": [], "security": []}
        for j, text in enumerate(gen_text):
            prob = batch[j // cfg.group_size]
            bd = score_completion(text, prob, max_cases=cfg.max_test_cases)
            rewards[j] = bd.reward
            for k in comps:
                comps[k].append(getattr(bd, k))
            last_breakdown = bd.as_dict()

        adv = rewards.view(cfg.num_prompts, cfg.group_size)
        adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True) + eps)
        adv = adv.view(-1).to("cuda")  # (N,)

        # --- old (pre-update) and reference logprobs (no grad) ---
        model.eval()
        with torch.no_grad():
            old_lp = _seq_logprobs(model, seqs, prompt_len, gen_mask, cfg.logprob_micro_batch)
            with model.disable_adapter():  # reference = frozen base
                ref_lp = _seq_logprobs(model, seqs, prompt_len, gen_mask, cfg.logprob_micro_batch)

        mask = gen_mask.float()
        tok_per_seq = mask.sum(dim=1).clamp_min(1.0)

        # --- inner optimisation passes (PPO-style clipped surrogate + KL) ---
        model.train()
        model.config.use_cache = False          # required with checkpointing
        model.gradient_checkpointing_enable()    # trade compute for activation memory
        for _ in range(cfg.epochs_per_batch):
            # micro-batch 1 in the grad pass: the backward graph over long APPS
            # sequences (x 248k-vocab logits) is the memory bottleneck on a T4.
            new_lp = _seq_logprobs(model, seqs, prompt_len, gen_mask, 1)
            ratio = torch.exp(new_lp - old_lp)
            a = adv.unsqueeze(1)
            surr = torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a)
            # k3 KL estimator: exp(ref-new) - (ref-new) - 1  >= 0
            d = ref_lp - new_lp
            kl = torch.exp(d) - d - 1.0
            per_tok = -(surr - cfg.kl_coeff * kl) * mask
            loss = (per_tok.sum(dim=1) / tok_per_seq).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()

        mean_r = rewards.mean().item()
        reward_hist.append(mean_r)
        mean_kl = (kl * mask).sum().item() / mask.sum().clamp_min(1).item()
        elapsed = time.time() - t_start
        print(
            f"step {step:03d} | reward {mean_r:.3f} | "
            f"correct {sum(comps['correctness'])/len(comps['correctness']):.3f} | "
            f"compile {sum(comps['compiles'])/len(comps['compiles']):.3f} | "
            f"kl {mean_kl:.4f} | loss {loss.item():.4f} | "
            f"{elapsed:.0f}s/{cfg.time_budget:.0f}s",
            flush=True,
        )
        step += 1

        # --- mid-training held-out eval + crash-safe checkpoint snapshot ---
        if cfg.eval_n and cfg.eval_every and step % cfg.eval_every == 0:
            em = evaluate(model, tok, cfg)
            em["step"] = step
            eval_curve.append(em)
            print(f"[eval@{step}] pass@1={em.get('eval_passrate'):.3f} "
                  f"compile={em.get('eval_compile'):.3f}", flush=True)
            out = Path(cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "eval_curve.jsonl", "a") as f:
                f.write(json.dumps(em) + "\n")
            model.save_pretrained(str(out))  # latest snapshot survives a later crash
            model.train()                    # restore train mode for the next step

    # --- held-out eval (clean comparable score across nodes) ---
    eval_metrics = {}
    if cfg.eval_n:
        eval_metrics = evaluate(model, tok, cfg)
        print(f"eval (held-out, greedy, n={eval_metrics.get('eval_n')}): "
              f"pass@1={eval_metrics.get('eval_passrate'):.3f} "
              f"compile={eval_metrics.get('eval_compile'):.3f}", flush=True)

    # --- save adapter ---
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    final_reward = reward_hist[-1] if reward_hist else 0.0
    best_reward = max(reward_hist) if reward_hist else 0.0
    metrics = {
        "final_reward": final_reward,
        "best_reward": best_reward,
        "mean_reward": (sum(reward_hist) / len(reward_hist)) if reward_hist else 0.0,
        "steps": step,
        "peak_vram_mb": peak_vram_mb,
        "total_seconds": time.time() - t_start,
        "checkpoint": str(out),
        "last_breakdown": last_breakdown,
        "eval_curve": eval_curve,
        **eval_metrics,
    }

    # Machine-readable hand-off for the experiment driver (robust vs stdout parsing).
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("---")
    print(f"final_reward:     {metrics['final_reward']:.6f}")
    print(f"best_reward:      {metrics['best_reward']:.6f}")
    print(f"mean_reward:      {metrics['mean_reward']:.6f}")
    print(f"steps:            {metrics['steps']}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_seconds:    {metrics['total_seconds']:.1f}")
    print(f"checkpoint:       {metrics['checkpoint']}")
    return metrics


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="GRPO post-training (exposed hyperparameter API).")
    defaults = GRPOConfig()
    for f in defaults.__dataclass_fields__.values():
        flag = "--" + f.name.replace("_", "-")
        if f.type == bool:
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"),
                            default=getattr(defaults, f.name))
        else:
            ap.add_argument(flag, type=type(getattr(defaults, f.name)),
                            default=getattr(defaults, f.name))
    return ap


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    train(**vars(args))
