"""Driver for the v2 ledger: execute queued checkpoints, then auto-evaluate experiments.

Forward loop of the science engine:
  queued checkpoint -> execute (real GRPO subprocess, or --mock) -> metrics ->
  any experiment whose resolved_nodes are now all done -> verdict.

Real execution reuses grpo.py exactly like driver.py (init_from=parent checkpoint).
--mock skips the GPU and writes synthetic metrics, so the orchestration can be
tested end-to-end without contending with a running real experiment.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ledger import Ledger

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def _runnable(cid, L):
    c = L.checkpoints[cid]
    if c["status"] != "queued":
        return False
    p = c.get("parent")
    return p is None or (p in L.checkpoints and L.checkpoints[p]["status"] == "done")


def _execute_mock(cid, L):
    c = L.checkpoints[cid]
    cfg = c["config"]
    return {"eval_passrate": cfg.get("mock_pass", 0.5), "eval_compile": 1.0,
            "eval_n": cfg.get("eval_n", 50), "steps": cfg.get("train_steps", 0)}


def _execute_real(cid, L):
    c = L.checkpoints[cid]
    out = RUNS / cid
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "python", "-u", "grpo.py", "--output-dir", str(out)]
    if c.get("parent"):
        cmd += ["--init-from", str(RUNS / c["parent"])]
    for k, v in c["config"].items():
        if k.startswith("mock_"):
            continue
        cmd += ["--" + k.replace("_", "-"), str(v)]
    with open(out / "train.log", "w") as log:
        subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((out / "metrics.json").read_text())


def run(mock=False):
    L = Ledger()
    executor = _execute_mock if mock else _execute_real
    while True:
        nxt = next((cid for cid in L.checkpoints if _runnable(cid, L)), None)
        if nxt is None:
            break
        L.checkpoints[nxt]["status"] = "running"; L.save()
        print(f"[ledger-driver] running {nxt} ({'mock' if mock else 'real'})", flush=True)
        try:
            metrics = executor(nxt, L)
        except Exception as e:  # noqa: BLE001
            L.checkpoints[nxt]["status"] = "failed"; L.checkpoints[nxt]["metrics"] = {"error": str(e)}
            L.save(); print(f"[ledger-driver] {nxt} FAILED: {e}", flush=True); continue
        L = Ledger()  # reload (real runs may have been long; keep it simple)
        L.checkpoints[nxt]["status"] = "done"
        L.checkpoints[nxt]["metrics"] = metrics
        L.checkpoints[nxt]["checkpoint"] = str(RUNS / nxt)
        # auto-evaluate any experiment now fully resolved
        for eid, exp in L.experiments.items():
            L.refresh_resolved(eid)
            if all(L.checkpoints.get(n, {}).get("status") == "done" for n in exp["resolved_nodes"]):
                v = L.evaluate(eid)
                print(f"[ledger-driver] {eid} -> {v['result'].upper()} ({v.get('finding')})", flush=True)
        L.save()
    print("[ledger-driver] no runnable queued checkpoints; done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="synthetic metrics, no GPU")
    run(**vars(ap.parse_args()))
