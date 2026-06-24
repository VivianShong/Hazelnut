# Experiment 2 — extended training on all MBPP (does pass@1 move, not just compile?)

_Run 2026-06-23/24. One continuous GRPO loop, `runs/exp2_long`._

## Question
Exp 1 showed compile saturates but held-out pass@1 stayed at base — because the dose was
tiny (KL ~5e-4). Does **extended training with an adequate dose** move problem success?

## Setup
Single continuous GRPO run on **all 374 MBPP train problems**:
`train_steps=160, lr=5e-5 (5x exp1), kl_coeff=0.04, num_prompts=2, group_size=4,
max_new_tokens=384`. Held-out greedy pass@1 on a **fixed 50-problem MBPP-test slice** every
20 steps. Base anchored on the same 50 (`runs/base50`). Wall time ~4.4 h, peak 8.2 GB, no errors.

## Results

| step | 0 (base) | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 |
|---|---|---|---|---|---|---|---|---|---|
| **pass@1** (n=50) | 0.42 | 0.58 | 0.62 | 0.62 | 0.58 | 0.66 | 0.66 | 0.62 | 0.60 |
| **compile** | 0.92 | 0.94 | 0.96 | 0.98 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 |
| **KL vs base** | 0 | 0.005 | 0.041 | 0.032 | 0.059 | 0.073 | 0.037 | 0.038 | ~0.04 |

## Conclusion — YES, extended training moves pass@1

- **pass@1: 0.42 → ~0.62 plateau** (+0.20 absolute, +48% relative). With n=50, σ≈0.07 at
  p=0.5, so this is ~3σ — a **real** gain, not noise. The Exp 1 null result was **under-dosing**,
  not a real ceiling: raising lr 5× pushed KL from ~5e-4 into the **active 0.04–0.07 regime**,
  and that's where pass@1 moved.
- **The gain is front-loaded and then plateaus.** Most of it lands by **step ~40** (0.42→0.62);
  steps 40–160 wander 0.58–0.66, all within ~1σ of each other → a plateau, not continued climb.
  Training past ~40–60 steps at this lr buys little. (The step-100/120 0.66 vs step-160 0.60 dip
  is within noise — don't over-read it as regression.)
- **Compile fully saturates (→1.00 by step 120)** — confirmed dead as a tracked metric; pass@1
  is the only live signal.
- **Stable**: KL grew but oscillated/bounded (clip + KL penalty held); no divergence or collapse
  over 160 steps.

## Implications
1. **Working recipe for "real" GRPO nodes**: lr≈5e-5, ~40–60 steps/node is enough to capture
   most of the gain; n≥50 eval to resolve it. Update node sizing accordingly.
2. **Best checkpoint is mid-run, not final** — the plateau means the last step isn't necessarily
   best. The tree should **keep the best-eval checkpoint**, not the last. (This run only kept the
   final snapshot; add per-eval distinct snapshots when best-keeping matters.)
3. **Now Exp 1's question is worth re-asking** with this dose: does chaining/depth or KL strength
   matter once a single node actually reaches the plateau? Re-run KL×depth at lr=5e-5, ~40 steps/node.
