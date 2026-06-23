 ---
  Project: Autonomous GRPO Post-Training System
  
  Build a tree-based autonomous system that uses an LLM research agent to navigate GRPO hyperparameter space, post-training Qwen3.5-2B-base to write better
  Python code. Built on top of Karpathy's autoresearch framework.

  P0 (Hackathon scope): Tree-based experiment management. Each node records a hyperparameter config and benchmark score. A research agent decides how the
  tree grows: branch (new config from parent checkpoint, optimizer reset), continue (same config, more steps from parent checkpoint, optimizer carried), or
  stop (prune low-potential nodes). The agent only emits config dicts — it never touches training code. A deterministic experiment driver receives configs
  and runs training as a subprocess.

  P1 (Out of scope): Let the research agent propose algorithmic changes (code as config). Requires sandboxed code execution and validation.

  ---
  Architecture: 2-Model System

  Research Agent (large hosted model)
  - Understands the user's goal, forms hypotheses, chooses hyperparameters
  - Input: user goal, full research tree (all nodes with configs + results)
  - Output: one structured action — branch, continue, or stop — with a config dict
  - Tools: web search, bash (for hardware detection)
  - Does NOT edit train.py; calls python train.py --lr 1e-5 --kl-coeff 0.04 ... via subprocess

  Trainee (Qwen3.5-2B-base, local GPU)
  - Post-trained with GRPO on Python coding tasks
  - Input: coding prompts; Output: generated code
  - Scored by a deterministic reward harness (not the LLM)

  ---
  Workflow

  1. Init: Research agent reads hardware constraints (nvidia-smi, VRAM), auto-fits batch size / sequence length. Establishes baseline by running GRPO with
  default config from Qwen3.5-2B-base (root node).
  2. Hypothesize: Agent proposes a hypothesis (e.g. "lower KL coefficient will allow more exploration") and encodes it as a config dict. One change at a time
  — controlled ablation.
  3. Run: Experiment driver calls python train.py --checkpoint <parent> --output-dir <node_dir> .... Training loads the parent node's checkpoint (or
  Qwen3.5-2B-base for the root), runs GRPO: sample completions, execute in Python sandbox, score, compute within-group advantages, update policy. Saves
  checkpoint + prints structured metrics.
  4. Evaluate: Deterministic reward harness scores on held-out eval set. Component scores logged individually:
    - Compile errors → py_compile / ast.parse
    - Style → ruff / flake8 warning count
    - Security → bandit findings
    - Correctness → unit test pass rate
    - Combined into scalar reward for GRPO, but components logged separately for agent reasoning.
  5. Decide: Agent reads results, checks hypothesis. Branch from promising nodes (new config, loads parent checkpoint, resets optimizer), continue nodes
  still improving (same config, loads parent checkpoint + optimizer state), stop dead ends. Deduplicates configs by content hash. Maintains a frontier of
  best-scoring nodes.
  6. Loop until benchmark score plateaus or budget exhausted.

  ---
  Key Design Decisions
  - Reference policy: always frozen Qwen3.5-2B-base (never updated)
  - Every node trains from its parent's checkpoint (root node starts from Qwen3.5-2B-base)
  - Continue carries optimizer state; branch resets it
  - Agent never produces code, only config dicts
  - Single GPU, one experiment at a time
  - Trained models exported only from frontier/leaf nodes

  ---
  Components


| File | Role |
| --- | --- |
| `train.py` | GRPO implementation. CLI API — all config via flags. Loads parent checkpoint, runs training, saves checkpoint, prints metrics. |
| `experiment_driver.py` | Translates agent actions $\rightarrow$ subprocess calls. Parses metrics. Updates `tree.json`. Handles crashes/timeouts. |
| `research_agent.py` | LLM decision loop. Reads tree, emits one action per cycle. |
| `tree.json` | Persistent state. All nodes with config, result, checkpoint path, status. |
| `reward_harness.py` | Deterministic scoring: compile + style + security + correctness $\rightarrow$ scalar + components. |
| `analysis.ipynb` | Post-hoc dashboard: progress curves, tree visualization, component breakdowns. |
  ---
  Plan

  1. GRPO training script (train.py) — CLI API, loads parent checkpoint (or Qwen3.5-2B-base for root), GRPO loop with Python sandbox execution, checkpoint
  save, structured metrics output
  2. Reward harness — py_compile + ruff + bandit + pytest, component + scalar scoring
  3. Experiment driver — subprocess orchestration, tree persistence, crash handling
  4. Research agent — LLM loop with hardware init, hypothesis formation, tree-based decision making, config dedup
  5. Export + serve — save frontier model, serve via vLLM/Ollama
  6. Dashboard — live progress curves + tree visualization
