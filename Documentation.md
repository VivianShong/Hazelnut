**Project requirements**
The goal of this project is to build an auto-self-improving agent, on top of copilot cli and the karparthy's auto-research framework, that could finetune the model, adjust prompt, etc. by itself in order to score better on the benchmark provided by the user.

**Purpose**
LLM API is costly, if we could use large model as an ML engineer to help us train or finetune a local small model like Qwen, it will save us cost, and also effort. We can just let it run overnight to find the best way to finetune the smaller model with reinforcement learning. For the hackathon we are going for the AI and hardware track, so it's best if we could let the LLM tune the training config to suit the hardware of the machine. For visualization, we want to show how well the training becomes through a curve updated in real time on a clean, minimalist style dashboard. Also we want to visualize the nodes for configuration in the version tree and their corresponding performance.

**2-model architecture**
1. Research model: big model to undertand user's task and goal, search the web, choose dataset and benchmarking tests, make hypothesis about how to improve the training of a smaller model, and choose the training hyperparameters (only apply small changes one at a time like a professional ai/ml researcher, and use a version tree to try branching, and rollback when needed).
Input: user prompt, last training result if given
Tools: websearch, bash commands
Output: a hypothesis + a hyperparameter config (a set of values) for each iteration. The research model does NOT edit `train.py`; it calls the exposed `train(param1, param2, ...)` function with the chosen config. The training/eval code is fixed after setup so runs stay directly comparable.

2. Student model: small local model to be fine-tuned with RL. 
Input: train/test prompts
Output: train/test prompt results
These training/testing results will be automatically evaluated by a deterministic python function, and the score and the model outputs should be fed to the larger model.

**Workflow**
1. Research model takes user requirement, runs command to get the system hardware constraints and searches the web for the suitable training algorithm and config. It suggests hypothesis, like the idea from an ai/ml researcher. It encodes the hypothesis as a hyperparameter config and calls the exposed `train(...)` function with it (it does NOT edit `train.py`). It may try multiple configs, for example different values of a single hyperparameter, recording each as a node in the version tree.
2. Run the training on the student model, collect result and evaluation score
3. Research model takes the results and checks if the previous hypothesis is correct. If not, it could generate new hypothesis. Then it decides whether to keep searching for a better config or roll back to the best config node so far (the configs form a version tree). Every node trains from the **same fixed base model** — nodes branch on the *config*, not on a continued checkpoint, so runs stay directly comparable. Keep track of the config versions with a tree structure, keep the best nodes in the frontier, better use heuristics and avoid trying the same config twice. Only the trained model(s) at the **leaf (frontier) nodes** are exported.
4. Loop until the benchmark test result can not be better.

**Instructions**
Help me set up a GRPO learning environment. As an example, the agent's task is to write good python code. Use the built-in python compiler. The agent's reward should be based on the evaluation of 1. how many compiler errors and warnings 2. coding standard 3. security, etc.

Use the existing auto-research framework in this repo to modify the configurations of the training, including the training algorithm, hyper parameters, etc.

After the training, export the model. Enable switching models.

**My questions**
1. Can the architecture be improved according to our requirements and goal?
2. Is it possible to integrate all this into copilot cli?
3. Is it possible to let the agent switch its model to the exported fine-tuned model?
4. If copilot cli can't switch model, what can we do?
5. What is the best way to showcase the agent is improving itself visually?

**Answers**

1. **Yes — make the Research model behave like a disciplined ML engineer, not a random search.** Keep the 2-model split but tighten the research loop so it mirrors how an ML engineer actually works:
   - *Read the hardware first.* On init, run `nvidia-smi` (+ CPU/RAM) and auto-fit batch size / seq len / depth to the available VRAM before any training. This is the "AI + hardware" angle of the track.
   - *One change at a time + baseline.* Always establish a baseline run, then change a single hyperparameter per hypothesis (controlled experiment / ablation). Record the hypothesis, the change, the expected effect, and the observed effect.
   - *Version everything as a tree.* Use git branches/worktrees (the repo already ignores `worktrees/`) — one branch per hypothesis. Keep a config tree where each node = a config + its score; maintain a "frontier" of best nodes and roll back to the best parent when a branch regresses. Every node trains from the same fixed base model (branch on the config, not on a continued checkpoint); only the trained models at the leaf/frontier nodes are exported.
   - *Don't repeat work.* Hash each config and skip configs already tried (dedupe). Prefer a cheap heuristic search over brute force — e.g. coordinate descent / successive-halving / a simple bandit over hyperparameters rather than full grid.
   - *Deterministic, decomposed reward.* For the "write good python code" task, the student's output is scored by a deterministic harness and the **component scores** (not just the total) are fed back to the Research model so it can reason about *why* it improved: 
     - compile errors/warnings → `py_compile` / `ast.parse` (hard errors) ;
     - coding standard → `ruff`/`flake8` warning count + style ;
     - security → `bandit` findings (weighted by severity) ;
     - optional correctness → run unit tests.
     Combine into a single scalar reward for GRPO, but log each component to a TSV so curves can be plotted.
   - *Reproducibility.* Pin seeds, log the hardware + git SHA + full config with every run.

2. **Yes.** Copilot CLI is designed to be extended exactly this way:
   - *Slash command / skill* — the `/self-improve` prompt (already installed by the script) drives the loop.
   - *Custom instructions* — `.github/copilot-instructions.md` + `AGENTS.md` give the Research model its "ML engineer" persona and rules ("one change at a time", "use git branches", etc.).
   - *Custom agents* — define a Research agent under `.github/agents/` (repo scope) so the orchestrator role is explicit and reusable.
   - *MCP server / hooks* — expose `train`, `evaluate`, `export`, `switch-model` as tools (MCP) or wire them as hooks so the agent calls them deterministically instead of free-form shell.
   - *Headless overnight runs* — `copilot -p "/self-improve" --allow-all-tools` (ideally inside `/sandbox enable` or `--cloud`) lets it loop unattended.

3. **Yes, with caveats.** Copilot CLI supports a custom model provider via environment variables, so the CLI itself can be pointed at the exported fine-tuned model:
   - `COPILOT_PROVIDER_BASE_URL` — your OpenAI-compatible endpoint
   - `COPILOT_PROVIDER_TYPE` — `openai` (works with Ollama / vLLM), `azure`, or `anthropic`
   - `COPILOT_PROVIDER_API_KEY` — key (not needed for local Ollama)
   - `COPILOT_MODEL` (or the `--model` flag / `/model` slash command) — the model name
   - **Caveat:** the model must support **tool calling + streaming** and ideally ≥128k context. A small GRPO-tuned Qwen specialised in "writing python" is a poor *agent driver*. So the practical pattern is: the **Research/orchestrator stays a strong hosted model**, and the **fine-tuned student is served behind an OpenAI-compatible endpoint** (vLLM/Ollama) that the reward harness and the demo call — not used as the CLI's own brain. You *can* still flip the CLI to the student model for a demo via the env vars above to show the switch working.

4. **It can — but for the switch you actually want, run a local OpenAI-compatible server.** Export the fine-tuned weights, serve them with `vllm serve` or `ollama`, then either (a) set the `COPILOT_PROVIDER_*` env vars / `--model` to make Copilot CLI talk to it, or (b) keep Copilot CLI on the hosted orchestrator and have the orchestrator hit the student endpoint as a tool. A small `switch-model` script/MCP tool just rewrites the env (or an `mcp-config.json`) and restarts the session — this is what `/switch-model` should do.

5. **Show two synced, minimalist live views, updating in real time as runs land in `results.tsv`:**
   - *Progress curve* — best `val_bpb` / reward (and its components) vs. experiment number / wall-clock, "best so far" line stepping down. Stream it (FastAPI + websockets, or a Textual/Plotly app tailing the TSV) so it animates while the agent works.
   - *Version tree* — the config DAG: each node = a tried config, edges = branch/rollback, node colour/size = score, frontier highlighted. This visually tells the "self-improvement" story — you literally watch it branch, fail, roll back, and climb.
   Clean light theme, few colours, big numbers (current best, # experiments, GPU). `analysis.ipynb` already plots the curve; `dashboard.py` is the start of the live version.

**Plan**
1. **Self-improve skill** — `/self-improve` installed to Copilot CLI (script done). Finish prompt engineering so it: reads hardware → forms one hypothesis → branches in git → trains 5 min → scores → keeps/rolls back → loops. Add the matching `/export-model` and `/switch-model` prompts.
2. **Research-model persona (ML-engineer discipline)** — author `.github/copilot-instructions.md` + a `.github/agents/research.md` custom agent encoding: baseline first, one change at a time, controlled ablations, git-branch-per-hypothesis, dedupe configs, maintain frontier, log hypothesis + observed effect.
3. **GRPO training harness** — set up the `grpo` package (algorithm + hyperparameters) on top of `train.py`; `train.py` exposes a parameterized `train(param1, param2, ...)` function and the Research model only supplies a config to it (it does not edit `train.py`). Student = small local model (Qwen). Auto-fit config to detected hardware.
4. **Deterministic reward harness** — `py_compile`/`ast` (errors) + `ruff`/`flake8` (warnings/style) + `bandit` (security) [+ pytest correctness], emitting per-component scores and one scalar reward; append to `results.tsv`.
5. **Experiment/version tracking** — config-tree with content-hash dedupe, frontier of best nodes, git worktrees per branch, heuristic search (coordinate descent / successive halving) instead of grid/random.
6. **Export + model switching** — `export-model` writes the fine-tuned checkpoint from a leaf/frontier node (the base model is fixed at every node; only trained leaf models are exported); serve via vLLM/Ollama (OpenAI-compatible); `switch-model` sets `COPILOT_PROVIDER_*` / `--model` (or rewrites `mcp-config.json`) to flip the active model. (`grpo.model_registry` tracks available models.)
7. **Live dashboard** — minimalist real-time UI: streaming progress/reward curves + interactive version tree (node colour = score, frontier highlighted), tailing `results.tsv`.
8. **Overnight autonomy** — run headless via `copilot -p "/self-improve" --allow-all-tools` inside a sandbox; wake up to a log of experiments + the best model on the frontier.