"""Summarize the KL x depth experiment from tree.json.

Prints held-out greedy pass@1 vs cumulative training steps for each chain, so we
can see whether chained GRPO compounds and whether KL strength gates depth.
"""
import json
from pathlib import Path

tree = json.loads(Path("tree.json").read_text())
nodes = tree["nodes"]
steps_per_node = 5

base = nodes.get("base", {})
bm = base.get("metrics", {})
print(f"experiment: {tree['meta'].get('experiment','')}")
print(f"\ndepth-0 base anchor: pass@1={bm.get('eval_passrate')} "
      f"compile={bm.get('eval_compile')} (n={bm.get('eval_n')}, status={base.get('status')})\n")

for prefix, kl in (("a", 0.04), ("b", 0.01)):
    print(f"=== arm {prefix}  (kl_coeff={kl}) ===")
    print(f"{'node':5} {'cum_steps':>9} {'status':8} {'pass@1':>7} {'compile':>8} {'mean_rwd':>9}")
    # base row
    print(f"{'base':5} {0:>9} {base.get('status',''):8} "
          f"{str(bm.get('eval_passrate','-')):>7} {str(bm.get('eval_compile','-')):>8} {'-':>9}")
    cum = 0
    for d in range(5):
        nid = f"{prefix}{d}"
        n = nodes.get(nid)
        if not n:
            continue
        cum += steps_per_node
        m = n.get("metrics", {})
        pr = m.get("eval_passrate")
        co = m.get("eval_compile")
        mr = m.get("mean_reward")
        print(f"{nid:5} {cum:>9} {n.get('status',''):8} "
              f"{(f'{pr:.3f}' if pr is not None else '-'):>7} "
              f"{(f'{co:.3f}' if co is not None else '-'):>8} "
              f"{(f'{mr:.3f}' if mr is not None else '-'):>9}")
    print()

# crude verdict
def final(prefix):
    for d in range(4, -1, -1):
        m = nodes.get(f"{prefix}{d}", {}).get("metrics", {})
        if m.get("eval_passrate") is not None:
            return d, m["eval_passrate"]
    return None, None

da, pa = final("a")
db, pb = final("b")
b0 = bm.get("eval_passrate")
if b0 is not None and pa is not None and pb is not None:
    print(f"base pass@1={b0:.3f}  ->  arm A (kl0.04) depth{da+1}={pa:.3f} (Δ{pa-b0:+.3f})  "
          f"|  arm B (kl0.01) depth{db+1}={pb:.3f} (Δ{pb-b0:+.3f})")
