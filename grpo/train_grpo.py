"""
Minimal, from-scratch GRPO trainer for the "write good Python" environment.

GRPO (Group Relative Policy Optimization) replaces PPO's value network with a
group baseline: for each prompt we sample a *group* of G completions, score
them with :func:`grpo.code_reward.reward_fn`, and use the group's mean/std to
normalize rewards into advantages. The policy is then updated with a clipped
surrogate objective plus a KL penalty to a frozen reference model.

No ``trl`` dependency — only ``torch`` + ``transformers``.

Pipeline per optimizer step:
  1. Rollout:  sample ``prompts_per_step`` groups of ``group_size`` completions,
               score each completion, compute group-relative advantages, and
               cache old/reference log-probs.
  2. Update:   run ``inner_epochs`` clipped-surrogate + KL updates over the
               cached rollout.

Example
-------
    uv run python -m grpo.train_grpo \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --group-size 8 \
        --prompts-per-step 2 \
        --max-steps 200 \
        --output-dir runs/grpo-python

The ``--model`` flag is required (no default model is hardcoded).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from statistics import fmean

import torch
import torch.nn.functional as F

from grpo.code_reward import evaluate_code
from grpo.prompts import TASKS, build_prompt


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="From-scratch GRPO for clean Python generation.")
    parser.add_argument("--model", required=True,
                        help="HF model id or local path of the causal LM to fine-tune.")
    parser.add_argument("--output-dir", default="runs/grpo-python",
                        help="Where to save the fine-tuned policy.")

    # GRPO sampling
    parser.add_argument("--group-size", type=int, default=8,
                        help="Completions sampled per prompt (the GRPO group, G).")
    parser.add_argument("--prompts-per-step", type=int, default=2,
                        help="Prompts (groups) sampled per optimizer step.")
    parser.add_argument("--max-steps", type=int, default=200)

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-beta", type=float, default=0.04,
                        help="Weight of the KL penalty to the reference model.")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="PPO-style ratio clipping epsilon.")
    parser.add_argument("--inner-epochs", type=int, default=1,
                        help="Optimization passes over each rollout batch.")
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Misc
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-sample", action="store_true",
                        help="Print the best generated completion per step.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Log-prob helpers
# ---------------------------------------------------------------------------


def sequence_token_logprobs(model, input_ids, attention_mask):
    """Per-token log-probs of ``input_ids`` under ``model``.

    Returns a tensor of shape ``[B, L-1]`` aligned to ``input_ids[:, 1:]``
    (the log-prob of each token given everything before it).
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def completion_mask(sequences, prompt_len, pad_id):
    """Boolean mask over target positions (``[:, 1:]``) selecting generated tokens.

    A target token at index ``i`` (predicting ``sequences[:, i + 1]``) is kept
    when its absolute position is past the prompt and it is not padding.
    """
    targets = sequences[:, 1:]
    positions = torch.arange(1, sequences.size(1), device=sequences.device)
    is_completion = positions.unsqueeze(0) >= prompt_len
    not_pad = targets != pad_id
    return is_completion & not_pad


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    """Cached data for one group, reused across inner epochs."""

    sequences: torch.Tensor   # [G, L]
    attn: torch.Tensor        # [G, L]
    mask: torch.Tensor        # [G, L-1] float, completion tokens
    advantages: torch.Tensor  # [G] float
    old_logp: torch.Tensor    # [G, L-1] float, detached
    ref_logp: torch.Tensor    # [G, L-1] float, detached


@torch.no_grad()
def collect_rollout(task, policy, reference, tokenizer, pad_id, args,
                    reward_log, breakdown_log):
    """Sample and score one group; return a :class:`Rollout` or ``None``.

    Returns ``None`` when the group's rewards have (near) zero variance, since
    GRPO has no relative signal to learn from in that case.
    """
    prompt_text = build_prompt(tokenizer, task)
    enc = tokenizer(prompt_text, return_tensors="pt").to(args.device)
    prompt_len = enc.input_ids.size(1)

    policy.eval()
    sequences = policy.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.group_size,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=pad_id,
    )
    policy.train()

    attn = (sequences != pad_id).long()
    attn[:, :prompt_len] = 1  # always attend to the prompt prefix

    rewards = []
    for i in range(sequences.size(0)):
        completion = tokenizer.decode(sequences[i, prompt_len:], skip_special_tokens=True)
        result = evaluate_code(completion)
        rewards.append(result.reward)
        reward_log.append(result.reward)
        breakdown_log["compile"].append(result.score_compile)
        breakdown_log["standard"].append(result.score_standard)
        breakdown_log["security"].append(result.score_security)

    if args.log_sample:
        best = max(range(len(rewards)), key=lambda j: rewards[j])
        sample = tokenizer.decode(sequences[best, prompt_len:], skip_special_tokens=True)
        print(f"  [best r={rewards[best]:+.3f}] {sample[:200]!r}")

    reward_t = torch.tensor(rewards, dtype=torch.float32, device=args.device)
    std = reward_t.std(unbiased=False)
    if std < 1e-6:
        return None
    advantages = (reward_t - reward_t.mean()) / (std + 1e-6)

    mask = completion_mask(sequences, prompt_len, pad_id).float()
    old_logp = sequence_token_logprobs(policy, sequences, attn)
    ref_logp = sequence_token_logprobs(reference, sequences, attn)

    return Rollout(sequences, attn, mask, advantages, old_logp, ref_logp)


def grpo_loss(policy, rollout: Rollout, args):
    """Clipped GRPO surrogate + KL penalty for one cached rollout."""
    new_logp = sequence_token_logprobs(policy, rollout.sequences, rollout.attn)
    adv = rollout.advantages.unsqueeze(1)

    ratio = torch.exp(new_logp - rollout.old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
    policy_loss = -torch.min(unclipped, clipped)

    # k3 KL estimator: exp(ref - new) - (ref - new) - 1  (>= 0, low variance)
    diff = rollout.ref_logp - new_logp
    kl = torch.exp(diff) - diff - 1.0

    per_token = policy_loss + args.kl_beta * kl
    return (per_token * rollout.mask).sum() / rollout.mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    print(f"Loading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    policy = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    reference = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    for step in range(1, args.max_steps + 1):
        prompts = random.sample(TASKS, k=min(args.prompts_per_step, len(TASKS)))

        reward_log: list[float] = []
        breakdown_log = {"compile": [], "standard": [], "security": []}

        # --- Rollout phase ---
        rollouts = []
        for task in prompts:
            rollout = collect_rollout(
                task, policy, reference, tokenizer, pad_id, args, reward_log, breakdown_log
            )
            if rollout is not None:
                rollouts.append(rollout)

        # --- Update phase ---
        if rollouts:
            for _ in range(args.inner_epochs):
                optimizer.zero_grad(set_to_none=True)
                for rollout in rollouts:
                    loss = grpo_loss(policy, rollout, args) / len(rollouts)
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
            update_note = ""
        else:
            update_note = " (no usable groups, skipped update)"

        mean_reward = fmean(reward_log) if reward_log else float("nan")
        comp = fmean(breakdown_log["compile"]) if breakdown_log["compile"] else float("nan")
        std = fmean(breakdown_log["standard"]) if breakdown_log["standard"] else float("nan")
        sec = fmean(breakdown_log["security"]) if breakdown_log["security"] else float("nan")
        print(
            f"step {step:4d} | reward {mean_reward:+.3f} | "
            f"compile {comp:.3f} standard {std:.3f} security {sec:.3f}{update_note}"
        )

        if step % args.save_every == 0:
            _save(policy, tokenizer, args.output_dir, step)

    _save(policy, tokenizer, args.output_dir, "final")
    print(f"Done. Final policy saved under {args.output_dir}")


def _save(model, tokenizer, output_dir, tag):
    path = f"{output_dir}/step-{tag}"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"  saved checkpoint to {path}")


if __name__ == "__main__":
    main()
