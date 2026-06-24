"""Deterministic experiment driver for the GRPO checkpoint tree.

The research agent never runs training directly. It only edits ``tree.json``:
it appends nodes with ``status: "queued"`` and a hyperparameter ``config``. This
driver is the *only* thing that executes GRPO. It:

  1. reads tree.json (fcntl-locked, atomic writes),
  2. pops a queued node whose parent is finished (FIFO, priority tie-break),
  3. claims it (``queued`` -> ``running``),
  4. launches ``grpo.py`` as a subprocess with the node's config, resuming from
     the parent's LoRA adapter (``--init-from``) when the node has a parent,
  5. records the result (``done`` + metrics + checkpoint, or ``failed`` + reason).

Field ownership is strict: the agent owns config/rationale/priority/parent; the
driver owns status transitions, run/, checkpoint, metrics, config_hash. The only
shared field is ``status``, and the driver only ever moves nodes *out* of
``queued``. That invariant is what makes the plain file lock sufficient.

Usage:
  uv run python driver.py            # drain the queue once, then exit
  uv run python driver.py --watch    # keep polling for new queued nodes
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE / "tree.json"
RUNS = HERE / "runs"
LOCK = HERE / "tree.json.lock"

# Config keys the driver passes through to grpo.py as CLI flags. output_dir and
# init_from are injected by the driver itself, so they are not accepted here.
_FORBIDDEN_KEYS = {"output_dir", "init_from"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def _locked():
    """Exclusive lock around a read-modify-write of tree.json."""
    LOCK.touch(exist_ok=True)
    with open(LOCK, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _read_tree() -> dict:
    return json.loads(TREE.read_text())


def _write_tree(tree: dict) -> None:
    tree["meta"]["updated_at"] = _now()
    tmp = TREE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tree, indent=2))
    os.replace(tmp, TREE)  # atomic


def _config_hash(parent: str | None, config: dict) -> str:
    payload = json.dumps({"parent": parent or "", "config": config}, sort_keys=True)
    return "sha1:" + hashlib.sha1(payload.encode()).hexdigest()[:12]


def _runnable(node: dict, nodes: dict) -> bool:
    """A queued node is runnable once its parent is done with a checkpoint."""
    parent = node.get("parent")
    if not parent:
        return True  # root: train from base
    p = nodes.get(parent)
    return bool(p and p.get("status") == "done" and p.get("checkpoint"))


def _pick(tree: dict) -> dict | None:
    nodes = tree["nodes"]
    cands = [n for n in nodes.values() if n["status"] == "queued" and _runnable(n, nodes)]
    if not cands:
        return None
    # Highest priority first, then oldest (FIFO) — a plain queue with a hint.
    cands.sort(key=lambda n: (-n.get("priority", 0.0), n.get("created_at", "")))
    return cands[0]


def _claim() -> dict | None:
    """Atomically move one runnable queued node to 'running' and return it."""
    with _locked():
        tree = _read_tree()
        node = _pick(tree)
        if node is None:
            return None
        # Dedupe: skip configs already completed in this tree.
        chash = _config_hash(node.get("parent"), node.get("config", {}))
        for other in tree["nodes"].values():
            if other["id"] != node["id"] and other.get("config_hash") == chash \
                    and other.get("status") == "done":
                node["status"] = "skipped"
                node["config_hash"] = chash
                node["run"] = {"note": f"duplicate of {other['id']}"}
                _write_tree(tree)
                return _claim()  # try the next one
        node["status"] = "running"
        node["config_hash"] = chash
        node["run"] = {"started_at": _now(), "log_path": f"runs/{node['id']}/train.log"}
        _write_tree(tree)
        return node


def _finish(node_id: str, status: str, *, checkpoint=None, metrics=None,
            exit_reason=None, error=None) -> None:
    with _locked():
        tree = _read_tree()
        node = tree["nodes"][node_id]
        node["status"] = status
        node["run"]["ended_at"] = _now()
        node["run"]["exit"] = exit_reason
        if error:
            node["run"]["error"] = error
        if checkpoint:
            node["checkpoint"] = checkpoint
        if metrics:
            node["metrics"] = metrics
        _write_tree(tree)


def _build_cmd(node: dict, tree: dict) -> tuple[list[str], Path]:
    out_dir = RUNS / node["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "python", "-u", "grpo.py", "--output-dir", str(out_dir)]
    parent = node.get("parent")
    if parent:
        cmd += ["--init-from", str(RUNS / parent)]
    for key, val in node.get("config", {}).items():
        if key in _FORBIDDEN_KEYS:
            continue
        cmd += ["--" + key.replace("_", "-"), str(val)]
    return cmd, out_dir


def _run_node(node: dict, tree: dict) -> None:
    cmd, out_dir = _build_cmd(node, tree)
    # Hard wall: time_budget + generous slack for model load + final save.
    budget = float(node.get("config", {}).get("time_budget", 300.0))
    timeout = budget + 600.0
    log_path = out_dir / "train.log"
    print(f"[driver] {node['id']}: {' '.join(cmd)}  (timeout {timeout:.0f}s)", flush=True)

    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            _finish(node["id"], "failed", exit_reason="timeout",
                    error=f"exceeded {timeout:.0f}s")
            print(f"[driver] {node['id']}: TIMEOUT", flush=True)
            return

    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-15:])
        _finish(node["id"], "failed", exit_reason="crash",
                error=f"returncode={proc.returncode}\n{tail}")
        print(f"[driver] {node['id']}: CRASH (rc={proc.returncode})", flush=True)
        return

    metrics_file = out_dir / "metrics.json"
    if not metrics_file.exists():
        _finish(node["id"], "failed", exit_reason="no_metrics",
                error="grpo.py exited 0 but wrote no metrics.json")
        print(f"[driver] {node['id']}: no metrics.json", flush=True)
        return

    metrics = json.loads(metrics_file.read_text())
    _finish(node["id"], "done", checkpoint=str(out_dir), metrics=metrics,
            exit_reason="ok")
    print(f"[driver] {node['id']}: DONE  mean_reward={metrics.get('mean_reward'):.3f} "
          f"steps={metrics.get('steps')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO experiment-tree driver.")
    ap.add_argument("--watch", action="store_true",
                    help="keep polling for new queued nodes instead of exiting")
    ap.add_argument("--poll", type=float, default=15.0, help="--watch poll interval (s)")
    args = ap.parse_args()

    RUNS.mkdir(exist_ok=True)
    while True:
        node = _claim()
        if node is not None:
            _run_node(node, _read_tree())
            continue
        if not args.watch:
            print("[driver] queue drained; exiting.", flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
