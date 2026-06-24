# Experiment 1 — KL strength × chain depth

_Run 2026-06-23, autonomous via `driver.py` on the checkpoint tree (`tree.json`)._

## Question
Does chaining GRPO-continue nodes down the tree **compound** held-out pass@1, or
plateau/regress — and does the KL-to-base anchor (locked design decision: reference =
base with adapter disabled) **gate** how far a chain can go?

## Setup
Two depth-5 GRPO-continue chains from base, identical except `kl_coeff`, plus a
0-step base anchor. Held-out greedy pass@1 on a **fixed** 24-problem MBPP-test slice
(disjoint from training) measured at every node.
Per node: `source=mbpp, num_prompts=2, group_size=4, max_new_tokens=384, train_steps=5,
lr=1e-5, seed=42`. Arm A `kl_coeff=0.04`, Arm B `kl_coeff=0.01`.

## Results (held-out greedy pass@1, n=24)

| cum. steps | 0 (base) | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| **Arm A** (kl=0.04) | 0.500 | 0.458 | 0.417 | 0.458 | 0.500 | 0.417 |
| **Arm B** (kl=0.01) | 0.500 | 0.417 | 0.500 | 0.500 | 0.333 | 0.458 |

compile rate = **0.958 (23/24) at every node**, base included. Per-node KL vs base
stayed **~1e-4 → ~5e-4** even at cumulative 25 steps. Peak VRAM flat ~8.1 GB at all depths.

## Conclusion — null result, and *why* it's null
- **No compounding, no depth trend, no KL effect.** Every cell is within ~1σ of base:
  at p=0.5, n=24 the binomial std is √(0.25/24) ≈ **0.10**, and the whole table spans
  0.333–0.500. The apparent end-of-chain dips (A: −0.083, B: −0.042) are noise.
- **The policy barely moved.** KL-to-base ≈ 5e-4 after 25 steps means the LoRA policy is
  essentially still the base model. The per-node "dose" (5 steps × 8 completions × lr 1e-5)
  is far too small to shift held-out behavior. compile is saturated (23/24) so it carries
  no signal; pass@1 is the only live metric and it's noise-dominated.
- **The KL-gating question is inconclusive by construction:** the KL penalty (coeff × ~5e-4)
  was never large enough to be active, so we never entered the regime the experiment meant
  to probe. kl=0.04 vs 0.01 is indistinguishable here.

## What this *does* establish
- **The tree/driver infra works autonomously end-to-end:** 11 nodes, resume-from-parent
  chains verified (`Resuming LoRA adapter from runs/<parent>`), immutable per-node
  checkpoints, one transient external-OOM on `a4` cleanly isolated (`failed`) and recovered
  by re-queue — exactly the crash-isolation the driver was built for.
- **The eval is underpowered:** n=24 → ±0.10. Can't resolve sub-0.1 effects.

## Next actions (to actually answer the question)
1. **Bigger dose per node** so the policy enters a non-trivial KL regime (~0.01–0.1):
   raise `train_steps` (≥20/node) and/or `lr` (try 5e-5–1e-4). Watch KL climb before
   trusting any pass@1 comparison.
2. **Bigger eval set** (n ≥ 50, ideally the full MBPP test split) for ±0.05 resolution;
   compile is saturated — drop it as a tracked metric.
3. **Re-run KL×depth only once a single chain visibly moves pass@1** — otherwise both arms
   sit on the base plateau and the contrast is meaningless.
