"""Deterministic experiment driver — the forward loop of the science engine.

The research agent never runs training and never hand-edits ``tree.json``. It only
writes ``outputs/ledger.json`` (via ``ledger.py``): it appends checkpoints with
``status: "queued"`` and a hyperparameter ``config``, and registers experiments.
This driver is the *only* thing that executes GRPO. For each runnable queued
checkpoint it:

  1. claims it (``queued`` -> ``running``) in the ledger,
  2. launches ``grpo.py`` as a subprocess with the checkpoint's config, resuming
     from the parent's LoRA adapter (``--init-from``) when it has a parent,
  3. writes the result back to the ledger (``done`` + metrics + checkpoint, or
     ``failed`` + error),
  4. auto-evaluates any experiment whose resolved nodes are now all ``done``,
  5. **regenerates** the per-run ``outputs/runs/<run>/tree.json`` — the *what*
     view of checkpoint nodes. The driver owns this file (the agent treats it as
     read-only); it is derived from the ledger, never authored by the agent.

Output layout (see README "Repo structure"):
  outputs/ledger.json                           — global research log (driver writes)
  outputs/runs/<run>/tree.json                  — this run's checkpoint nodes (driver writes)
  outputs/runs/<run>/<node>/model_checkpoint/   — LoRA adapter (gitignored)
  outputs/runs/<run>/<node>/logs/               — train.log, metrics.json, *.jsonl

--mock skips the GPU and writes synthetic metrics, so the orchestration can be
tested end-to-end without a real experiment.

Usage:
  uv run python driver.py                 # drain the queue once, then exit
  uv run python driver.py --mock          # synthetic metrics, no GPU
  uv run python driver.py --run run-002   # drive a different run dir
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ledger import Ledger

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = "run-001"


def _run_dir(run: str) -> Path:
    return HERE / "outputs" / "runs" / run


def _node_dir(run: str, cid: str) -> Path:
    return _run_dir(run) / cid


def _runnable(cid, L):
    c = L.checkpoints[cid]
    if c["status"] != "queued":
        return False
    p = c.get("parent")
    return p is None or (p in L.checkpoints and L.checkpoints[p]["status"] == "done")


def _execute_mock(cid, L, run):
    c = L.checkpoints[cid]
    cfg = c["config"]
    return {"eval_passrate": cfg.get("mock_pass", 0.5), "eval_compile": 1.0,
            "eval_n": cfg.get("eval_n", 50), "steps": cfg.get("train_steps", 0)}


def _execute_real(cid, L, run):
    c = L.checkpoints[cid]
    out = _node_dir(run, cid)
    log_dir = out / "logs"                # receipts (committed)
    log_dir.mkdir(parents=True, exist_ok=True)
    # grpo.py splits its --output-dir into model_checkpoint/ (adapter) and logs/.
    cmd = ["uv", "run", "python", "-u", "grpo.py", "--output-dir", str(out)]
    if c.get("parent"):
        cmd += ["--init-from", str(_node_dir(run, c["parent"]) / "model_checkpoint")]
    for k, v in c["config"].items():
        if k.startswith("mock_"):
            continue
        cmd += ["--" + k.replace("_", "-"), str(v)]
    with open(log_dir / "train.log", "w") as log:
        subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((log_dir / "metrics.json").read_text())


def _sync_tree(L, run: str) -> None:
    """(Re)generate outputs/runs/<run>/tree.json from the ledger checkpoints.

    The *what* layer: a per-run projection of the checkpoint nodes the agent reads
    but never edits. Driver-owned, derived from the ledger.
    """
    root_id = next((cid for cid, c in L.checkpoints.items() if not c.get("parent")), None)
    nodes = {
        cid: {
            "id": cid,
            "parent": c.get("parent"),
            "status": c.get("status"),
            "config": c.get("config", {}),
            "metrics": c.get("metrics", {}),
            "checkpoint": c.get("checkpoint"),
            "run": {"log_path": f"outputs/runs/{run}/{cid}/logs/train.log"},
        }
        for cid, c in L.checkpoints.items()
    }
    tree = {
        "meta": {"schema_version": 1, "task": "grpo-qwen3.5-2b-code",
                 "run": run, "root_id": root_id, "updated_at": L.meta.get("updated_at")},
        "nodes": nodes,
    }
    run_dir = _run_dir(run)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "tree.json.tmp"
    tmp.write_text(json.dumps(tree, indent=2))
    tmp.replace(run_dir / "tree.json")  # atomic; driver-owned


def run(mock=False, run="run-001"):
    L = Ledger()
    executor = _execute_mock if mock else _execute_real
    while True:
        nxt = next((cid for cid in L.checkpoints if _runnable(cid, L)), None)
        if nxt is None:
            break
        L.checkpoints[nxt]["status"] = "running"; L.save(); _sync_tree(L, run)
        print(f"[driver] running {nxt} ({'mock' if mock else 'real'})", flush=True)
        try:
            metrics = executor(nxt, L, run)
        except Exception as e:  # noqa: BLE001
            L.checkpoints[nxt]["status"] = "failed"; L.checkpoints[nxt]["metrics"] = {"error": str(e)}
            L.save(); _sync_tree(L, run)
            print(f"[driver] {nxt} FAILED: {e}", flush=True); continue
        L = Ledger()  # reload (real runs may have been long; keep it simple)
        L.checkpoints[nxt]["status"] = "done"
        L.checkpoints[nxt]["metrics"] = metrics
        L.checkpoints[nxt]["checkpoint"] = str(_node_dir(run, nxt) / "model_checkpoint")
        # auto-evaluate any experiment now fully resolved
        for eid, exp in L.experiments.items():
            L.refresh_resolved(eid)
            if all(L.checkpoints.get(n, {}).get("status") == "done" for n in exp["resolved_nodes"]):
                v = L.evaluate(eid)
                print(f"[driver] {eid} -> {v['result'].upper()} ({v.get('finding')})", flush=True)
        L.save(); _sync_tree(L, run)
    print("[driver] no runnable queued checkpoints; done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GRPO ledger driver: run queued checkpoints, sync tree.json.")
    ap.add_argument("--mock", action="store_true", help="synthetic metrics, no GPU")
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help="run id under outputs/runs/ to drive (default: run-001)")
    run(**vars(ap.parse_args()))
