"""Generate tree.json for the KL x depth experiment.

Two depth-5 GRPO-continue chains from base, differing only in kl_coeff, plus a
0-step base anchor. Held-out greedy pass@1 (eval_n) is measured at every node,
so the comparable metric is eval_passrate vs cumulative training steps.
"""
import json
from datetime import datetime, timezone

BASE_CFG = dict(
    source="mbpp", num_prompts=2, group_size=4, max_new_tokens=384,
    train_steps=5, time_budget=800, eval_n=24, eval_batch=8, seed=42,
)
DEPTH = 5
ARMS = {"a": 0.04, "b": 0.01}  # node-id prefix -> kl_coeff


def ts(i):
    return datetime(2026, 6, 23, 19, 0, i, tzinfo=timezone.utc).isoformat()


nodes = {}
clock = 0

# depth-0 base anchor: 0 steps -> evaluates the base model on the eval set.
nodes["base"] = {
    "id": "base", "parent": None, "status": "queued",
    "config": {**BASE_CFG, "train_steps": 0},
    "rationale": "depth-0 anchor: base-model held-out pass@1 (both chains start here)",
    "priority": 1.0, "created_at": ts(clock),
}
clock += 1

for prefix, kl in ARMS.items():
    parent = None
    for d in range(DEPTH):
        nid = f"{prefix}{d}"
        nodes[nid] = {
            "id": nid, "parent": parent, "status": "queued",
            "config": {**BASE_CFG, "kl_coeff": kl},
            "rationale": (f"arm {prefix} (kl={kl}) depth {d+1}: "
                          f"{'GRPO from base' if d == 0 else f'continue {parent}'}, +5 steps"),
            # shallower nodes first so both chains advance in lockstep if time runs short
            "priority": 0.95 - 0.1 * d, "created_at": ts(clock),
        }
        parent = nid
        clock += 1

tree = {
    "meta": {
        "schema_version": 1,
        "task": "grpo-qwen3.5-2b-code",
        "base_model": "/opt/Hazelnut/models/qwen3.5-2b-Base",
        "experiment": "kl-strength x chain-depth: does chained GRPO compound held-out pass@1?",
        "root_id": "base",
        "updated_at": ts(clock),
    },
    "nodes": nodes,
}
with open("tree.json", "w") as f:
    json.dump(tree, f, indent=2)
print(f"wrote tree.json: {len(nodes)} nodes")
for n in nodes.values():
    print(f"  {n['id']:5} parent={str(n['parent']):5} kl={n['config'].get('kl_coeff','-')} "
          f"steps={n['config']['train_steps']} prio={n['priority']:.2f}")
