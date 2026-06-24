"""Generate tree.json for Exp 3 — KL x depth re-run at the working dose.

Exp 1 was under-dosed (KL ~5e-4, pass@1 flat). Exp 2 found the working recipe
(lr=5e-5, gain plateaus by ~step 40, pass@1 0.42->~0.62). Now re-ask the Exp 1
question at that dose: once a single node reaches the plateau, does chaining/depth
or KL strength matter?

Two depth-4 GRPO-continue chains from base, identical except kl_coeff. 40 steps/node
spans cumulative 40/80/120/160 — the same range Exp 2's single run covered, so arm C
(kl=0.04) also serves as a chain-vs-flat check against Exp 2. Depth-0 anchor reuses
runs/base50 (pass@1=0.42 on the same n=50 set).
"""
import json
from datetime import datetime, timezone

BASE_CFG = dict(
    source="mbpp", num_prompts=2, group_size=4, max_new_tokens=384,
    train_steps=40, lr=5e-5, eval_n=50, eval_every=0, eval_batch=8,
    data_limit=500, time_budget=5000, seed=42,
)
DEPTH = 4
ARMS = {"c": 0.04, "d": 0.01}  # node-id prefix -> kl_coeff


def ts(i):
    return datetime(2026, 6, 24, 6, 0, i, tzinfo=timezone.utc).isoformat()


nodes = {}
clock = 0
for prefix, kl in ARMS.items():
    parent = None
    for depth in range(DEPTH):
        nid = f"{prefix}{depth}"
        nodes[nid] = {
            "id": nid, "parent": parent, "status": "queued",
            "config": {**BASE_CFG, "kl_coeff": kl},
            "rationale": (f"arm {prefix} (kl={kl}) depth {depth+1}: "
                          f"{'GRPO from base' if depth == 0 else f'continue {parent}'}, +40 steps @ lr5e-5"),
            "priority": 0.95 - 0.1 * depth,  # shallower first -> chains advance in lockstep
            "created_at": ts(clock),
        }
        parent = nid
        clock += 1

tree = {
    "meta": {
        "schema_version": 1,
        "task": "grpo-qwen3.5-2b-code",
        "base_model": "/opt/Hazelnut/models/qwen3.5-2b-Base",
        "experiment": "exp3: KL x depth at working dose (lr5e-5, 40 steps/node); base anchor = runs/base50 (0.42)",
        "root_id": "c0",
        "updated_at": ts(clock),
    },
    "nodes": nodes,
}
with open("tree.json", "w") as f:
    json.dump(tree, f, indent=2)
print(f"wrote tree.json: {len(nodes)} nodes")
for n in nodes.values():
    print(f"  {n['id']:4} parent={str(n['parent']):4} kl={n['config']['kl_coeff']} "
          f"steps={n['config']['train_steps']} lr={n['config']['lr']} prio={n['priority']:.2f}")
