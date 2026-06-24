"""Back-fill Exp 1/2/3 into the v2 ledger from real run data, then compute verdicts.

This is the end-to-end exercise of the new schema on genuine results:
  ingest checkpoints (facts) -> register experiments (claims w/ selectors) ->
  resolve selectors over the checkpoint tree -> verdict engine reads real metrics ->
  inquiry DAG. No GPU needed; does not touch the running Exp 3 tree.json/driver.
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ledger import Ledger  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
L = Ledger()


def ingest_tree(tree_path, only_prefixes=None):
    """Pull checkpoints (id/parent/config/status/metrics) from an old tree.json."""
    if not Path(tree_path).exists():
        return 0
    nodes = json.loads(Path(tree_path).read_text())["nodes"]
    k = 0
    for nid, n in nodes.items():
        if only_prefixes and not any(nid.startswith(p) for p in only_prefixes):
            continue
        # A node still owned by another (running) driver is "external": ingest it as a
        # fact placeholder, but never let ledger_driver claim/run it. Only truly finished
        # nodes (eval recorded) become "done".
        done = "eval_passrate" in (n.get("metrics") or {})
        L.add_checkpoint(nid, parent=n.get("parent"), config=n.get("config", {}),
                         status="done" if done else "external", metrics=n.get("metrics", {}),
                         checkpoint=n.get("checkpoint"))
        k += 1
    return k


def ingest_run(cid, config, parent=None):
    """Pull one checkpoint's metrics straight from runs/<cid>/metrics.json."""
    mp = HERE / "runs" / cid / "metrics.json"
    metrics = json.loads(mp.read_text()) if mp.exists() else {}
    L.add_checkpoint(cid, parent=parent, config=config,
                     status="done" if metrics else "missing", metrics=metrics,
                     checkpoint=str(HERE / "runs" / cid) if metrics else None)


# --- Layer 1: checkpoints (facts) ---
print("ingested exp1 nodes:", ingest_tree(HERE / "results/exp1_tree.json"))
print("ingested exp3 nodes:", ingest_tree(HERE / "tree.json", only_prefixes=("c", "d")))
ingest_run("base50", {"train_steps": 0, "eval_n": 50})
ingest_run("exp2_long", {"train_steps": 160, "lr": 5e-5, "kl_coeff": 0.04, "eval_n": 50})

# --- Layer 2: experiments (claims) + inquiry DAG ---
L.register(
    "exp1", motivated_by=None,
    question="Does chained GRPO compound pass@1, and does KL strength gate depth?",
    hypothesis="kl=0.04 vs 0.01 chains diverge with depth (KL gates how far a chain compounds)",
    predicted="arm_effect",
    selector={"type": "contrast", "arms": {
        "kl0.04": {"type": "path", "tip": "a4"},
        "kl0.01": {"type": "path", "tip": "b4"}}},
    anchor="base", metric={"name": "eval_passrate", "n": 24},
    decision_rule={"type": "contrast_at_tip", "k_sigma": 2},
)
L.register(
    "exp2", motivated_by="exp1",
    question="At an adequate dose (lr=5e-5), does extended training move pass@1?",
    hypothesis="extended training raises held-out pass@1 above base beyond noise",
    predicted="improved",
    selector={"type": "curve", "node": "exp2_long"},
    anchor="base50", metric={"name": "eval_passrate", "n": 50},
    decision_rule={"type": "trend_vs_anchor", "k_sigma": 2},
)
L.register(
    "exp3", motivated_by="exp2",
    question="At the working dose, does KL strength (0.04 vs 0.01) change the plateau?",
    hypothesis="both arms reach the same plateau — KL strength does not matter once it's active",
    predicted="no_arm_effect",
    selector={"type": "contrast", "arms": {
        "kl0.04": {"type": "path", "tip": "c3"},
        "kl0.01": {"type": "path", "tip": "d3"}}},
    anchor="base50", metric={"name": "eval_passrate", "n": 50},
    decision_rule={"type": "contrast_at_tip", "k_sigma": 2},
)

for eid in ("exp1", "exp2", "exp3"):
    L.evaluate(eid)
L.save()

print(f"\nledger.json: {len(L.checkpoints)} checkpoints, {len(L.experiments)} experiments\n")
print(L.inquiry_dag())
print("\n--- verdicts ---")
for eid in ("exp1", "exp2", "exp3"):
    v = L.experiments[eid]["verdict"]
    print(f"\n[{eid}] {v['result'].upper()}  finding={v.get('finding')}")
    print("  power:", L.experiments[eid]["power_check"]["note"])
    ev = v.get("evidence", {})
    for kk in ("arm_values", "diff", "threshold", "anchor", "best", "delta_vs_anchor",
               "plateau_onset_step"):
        if kk in ev:
            print(f"  {kk}: {ev[kk]}")
