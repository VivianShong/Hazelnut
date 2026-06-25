"""Experiment ledger — the headline structure for the RL-science agent.

Three nested ideas (see PROGRESS.md goal):
  checkpoints (facts)  ⊂  experiments (claims over node-sets, via selectors)  ⊂  inquiry DAG

Layer 1 — `checkpoints`: facts. {id -> parent, config, status, metrics, checkpoint}.
  Identical in spirit to the old tree.json nodes. A checkpoint is a *point* (model
  weights produced by one GRPO run from a parent). Immutable once `done`.

Layer 2 — `experiments`: claims. Each holds a hypothesis + a **selector** that
  *references* (does not own) a set of checkpoints, a pre-registered metric +
  decision_rule + predicted finding, and — after running — a machine-computed
  verdict (confirmed/refuted/inconclusive + evidence). A checkpoint may belong to
  many experiments (many-to-many) → selectors reference, they don't contain.

Inquiry DAG: experiments link via `motivated_by` (Exp1 null → Exp2 diagnose →
  Exp3 re-test). This DAG of *questions* is the headline; the checkpoint tree is
  the substrate each experiment indexes into.

Selectors (the structural shape the hypothesis is about):
  {"type":"point","node":X}          one checkpoint (vs the experiment's anchor)
  {"type":"path","tip":X}            the chain root→X (parent-walk); trend over cumulative steps
  {"type":"curve","node":X}          one run's intra-node eval curve (step→metric)
  {"type":"contrast","arms":{name:<selector>,...}}   compare arms (KL, depth, ...)
The segment-tree intuition lives in `path`/`curve`: a trend is a *segment over the
step axis*, and aggregates (argmax checkpoint, plateau onset, slope) are range
queries over it.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "outputs" / "ledger.json"

# The standing research goal (top-level, mirrors the README Ledger.json example).
# Every experiment in the inquiry DAG is in service of this objective.
DEFAULT_GOAL = "improve Qwen3.5-2B-base's pass rate on MBPP dataset"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _binom_sigma(p: float, n: int) -> float:
    """Std error of a pass-rate metric estimated from n samples."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.sqrt(p * (1 - p) / max(n, 1))


class Ledger:
    def __init__(self, path: Path = LEDGER):
        self.path = Path(path)
        if self.path.exists():
            d = json.loads(self.path.read_text())
        else:
            d = {"meta": {"schema_version": 2, "headline": "inquiry-DAG over a checkpoint substrate"},
                 "checkpoints": {}, "experiments": {}}
        self.meta = d["meta"]
        self.meta.setdefault("goal", DEFAULT_GOAL)  # standing objective (README Goal field)
        self.checkpoints = d["checkpoints"]
        self.experiments = d["experiments"]

    # ---------------- goal (top-level objective) ----------------
    @property
    def goal(self) -> str:
        return self.meta.get("goal", DEFAULT_GOAL)

    def set_goal(self, goal: str) -> None:
        self.meta["goal"] = goal

    # ---------------- persistence ----------------
    def save(self) -> None:
        self.meta["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"meta": self.meta, "checkpoints": self.checkpoints, "experiments": self.experiments},
            indent=2))
        tmp.replace(self.path)

    # ---------------- layer 1: checkpoints ----------------
    def add_checkpoint(self, cid, parent=None, config=None, status="queued",
                       metrics=None, checkpoint=None) -> dict:
        node = {"id": cid, "parent": parent, "status": status,
                "config": config or {}, "metrics": metrics or {}, "checkpoint": checkpoint}
        self.checkpoints[cid] = node
        return node

    def _path_to(self, tip: str) -> list[str]:
        """Chain root→tip by walking parent pointers."""
        chain = []
        cur = tip
        seen = set()
        while cur is not None and cur in self.checkpoints and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = self.checkpoints[cur].get("parent")
        return list(reversed(chain))

    # ---------------- selector resolution ----------------
    def resolve(self, selector: dict) -> list[str]:
        """Resolve a selector to the (deduped, ordered) checkpoint ids it references."""
        t = selector["type"]
        if t == "point":
            return [selector["node"]]
        if t in ("path",):
            return self._path_to(selector["tip"])
        if t == "curve":
            return [selector["node"]]
        if t == "contrast":
            out: list[str] = []
            for sub in selector["arms"].values():
                for nid in self.resolve(sub):
                    if nid not in out:
                        out.append(nid)
            return out
        raise ValueError(f"unknown selector type: {t}")

    # ---------------- metric access ----------------
    def _final(self, nid: str, metric: str):
        return self.checkpoints.get(nid, {}).get("metrics", {}).get(metric)

    def _cum_steps(self, nid: str) -> int:
        return sum(int(self.checkpoints[c]["config"].get("train_steps", 0))
                   for c in self._path_to(nid))

    def _trajectory(self, selector: dict, metric: str, anchor: str | None):
        """[(cum_step, value)] for a path or curve selector, anchor prepended at step 0."""
        pts = []
        if anchor is not None and self._final(anchor, metric) is not None:
            pts.append((0, self._final(anchor, metric)))
        if selector["type"] == "path":
            for nid in self.resolve(selector):
                v = self._final(nid, metric)
                if v is not None:
                    pts.append((self._cum_steps(nid), v))
        elif selector["type"] == "curve":
            cur = self.checkpoints[selector["node"]]["metrics"].get("eval_curve", [])
            for e in cur:
                pts.append((e["step"], e.get(metric)))
        return [(s, v) for s, v in pts if v is not None]

    # ---------------- layer 2: experiments ----------------
    def register(self, eid, question, hypothesis, predicted, selector, metric,
                 decision_rule, anchor=None, motivated_by=None) -> dict:
        """Pre-register an experiment: cache resolved node-ids and run the power check."""
        resolved = self.resolve(selector)
        n = int(metric.get("n", 0))
        sigma = _binom_sigma(0.5, n) if n else None
        mde = round(2 * sigma, 3) if sigma else None  # min effect detectable at ~2σ
        power = {"eval_n": n, "sigma_at_p0.5": round(sigma, 3) if sigma else None,
                 "min_detectable_effect_2sigma": mde,
                 "note": (f"can't resolve effects < {mde}" if mde else "no n given")}
        exp = {
            "id": eid, "question": question, "hypothesis": hypothesis,
            "predicted": predicted, "selector": selector, "anchor": anchor,
            "resolved_nodes": resolved, "metric": metric,
            "decision_rule": decision_rule, "power_check": power,
            "motivated_by": motivated_by, "verdict": None, "created_at": _now(),
        }
        self.experiments[eid] = exp
        return exp

    def refresh_resolved(self, eid: str) -> None:
        self.experiments[eid]["resolved_nodes"] = self.resolve(self.experiments[eid]["selector"])

    # ---------------- verdict engine ----------------
    def evaluate(self, eid: str) -> dict:
        """Compute evidence from real metrics, derive a finding, set the verdict.

        verdict.result: confirmed (finding == predicted) / refuted (contradicts) /
        inconclusive (effect within noise → underpowered to decide).
        """
        exp = self.experiments[eid]
        self.refresh_resolved(eid)
        rule = exp["decision_rule"]
        metric = exp["metric"]["name"]
        n = int(exp["metric"].get("n", 0))
        anchor = exp.get("anchor")
        rtype = rule["type"]
        k = rule.get("k_sigma", 2)
        ev: dict = {}
        finding = None
        inconclusive = False

        if rtype == "trend_vs_anchor":
            traj = self._trajectory(exp["selector"], metric, anchor)
            if len(traj) < 2:
                return self._pending(eid, "not enough points yet")
            base = traj[0][1]
            best_step, best = max(traj[1:], key=lambda sv: sv[1])
            sigma = _binom_sigma(best, n)
            delta = best - base
            # plateau onset: first step within 1σ of best
            onset = next((s for s, v in traj[1:] if v >= best - sigma), best_step)
            ev = {"trajectory": traj, "anchor": base, "best": best, "best_step": best_step,
                  "delta_vs_anchor": round(delta, 3), "sigma": round(sigma, 3),
                  "threshold": round(k * sigma, 3), "plateau_onset_step": onset}
            if delta > k * sigma:
                finding = "improved"
            elif delta < -k * sigma:
                finding = "regressed"
            else:
                finding = "flat"; inconclusive = True

        elif rtype == "contrast_at_tip":
            arms = exp["selector"]["arms"]
            vals = {}
            for name, sub in arms.items():
                tip = sub.get("tip") or sub.get("node")
                vals[name] = self._final(tip, metric)
            if any(v is None for v in vals.values()):
                return self._pending(eid, "arms not all complete")
            names = list(vals)
            diff = vals[names[0]] - vals[names[1]]
            sigma = math.hypot(_binom_sigma(vals[names[0]], n), _binom_sigma(vals[names[1]], n))
            ev = {"arm_values": vals, "diff": round(diff, 3), "pooled_sigma": round(sigma, 3),
                  "threshold": round(k * sigma, 3)}
            if abs(diff) > k * sigma:
                finding = "arm_effect"
            else:
                finding = "no_arm_effect"; inconclusive = abs(diff) < sigma
        else:
            raise ValueError(f"unknown decision_rule type: {rtype}")

        result = "confirmed" if finding == exp["predicted"] else "refuted"
        if inconclusive:
            result = "inconclusive"
        verdict = {"result": result, "finding": finding, "evidence": ev,
                   "power": exp["power_check"], "written_at": _now()}
        exp["verdict"] = verdict
        return verdict

    def _pending(self, eid, why):
        v = {"result": "pending", "finding": None, "why": why, "written_at": _now()}
        self.experiments[eid]["verdict"] = v
        return v

    # ---------------- views ----------------
    def inquiry_dag(self) -> str:
        roots = [e for e in self.experiments.values() if not e.get("motivated_by")]
        lines = ["INQUIRY DAG (questions; edges = motivated_by):"]

        def walk(eid, depth):
            e = self.experiments[eid]
            v = e.get("verdict") or {}
            lines.append("  " * depth + f"• {eid}: {e['question']}  ⇒ "
                         f"[{(v.get('result') or 'unrun').upper()}] {v.get('finding') or ''}")
            for c in self.experiments.values():
                if c.get("motivated_by") == eid:
                    walk(c["id"], depth + 1)

        for r in roots:
            walk(r["id"], 0)
        return "\n".join(lines)

    def frontier(self, metric: str = "eval_passrate", top: int = 5) -> list[dict]:
        """Best-scoring *done* checkpoints — the decision support for branch/continue."""
        scored = [
            {"id": c["id"], "parent": c.get("parent"), metric: c["metrics"].get(metric),
             "config": c.get("config", {})}
            for c in self.checkpoints.values()
            if c.get("status") == "done" and c.get("metrics", {}).get(metric) is not None
        ]
        scored.sort(key=lambda r: r[metric], reverse=True)
        return scored[:top]

    def summary(self) -> dict:
        """Compact, model-readable snapshot: goal, checkpoint states, frontier, verdicts."""
        states: dict[str, int] = {}
        for c in self.checkpoints.values():
            states[c["status"]] = states.get(c["status"], 0) + 1
        return {
            "goal": self.goal,
            "checkpoints_total": len(self.checkpoints),
            "checkpoint_states": states,
            "frontier": self.frontier(),
            "experiments": {
                eid: {"question": e["question"], "predicted": e["predicted"],
                      "verdict": (e.get("verdict") or {}).get("result", "unrun"),
                      "finding": (e.get("verdict") or {}).get("finding"),
                      "motivated_by": e.get("motivated_by")}
                for eid, e in self.experiments.items()
            },
        }


# ---------------- CLI: the read/write entrypoint the research agent uses ----------------
# The agent never hand-edits ledger.json. It *reads* via `read`/`show`/`summary`/`frontier`
# and *writes* via `goal --set`, `add-checkpoint`, `register`, and `evaluate` — every write
# goes through the atomic Ledger.save(). Configs/selectors/metrics are passed as JSON so the
# agent can emit arbitrary config dicts (README: "the agent only emits config dicts").

def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Experiment ledger — read/write CLI for the research agent.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="inquiry DAG + verdicts (human view)")

    p_read = sub.add_parser("read", help="dump ledger JSON (for the model to read)")
    p_read.add_argument("--section", choices=["all", "meta", "checkpoints", "experiments", "summary"],
                        default="summary")
    p_read.add_argument("--id", help="restrict to one checkpoint/experiment id")

    p_goal = sub.add_parser("goal", help="print, or with --set, update the research goal")
    p_goal.add_argument("--set", dest="set_to", help="new goal text")

    p_front = sub.add_parser("frontier", help="best-scoring done checkpoints")
    p_front.add_argument("--metric", default="eval_passrate")
    p_front.add_argument("--top", type=int, default=5)

    p_add = sub.add_parser("add-checkpoint", help="append a checkpoint (default queued) for the driver to run")
    p_add.add_argument("--id", required=True)
    p_add.add_argument("--parent", default=None)
    p_add.add_argument("--config", default="{}", help="JSON hyperparameter dict")
    p_add.add_argument("--status", default="queued")
    p_add.add_argument("--metrics", default="{}", help="JSON metrics dict")
    p_add.add_argument("--checkpoint", default=None)

    p_reg = sub.add_parser("register", help="pre-register an experiment (claim over a node-set)")
    p_reg.add_argument("--id", required=True)
    p_reg.add_argument("--question", required=True)
    p_reg.add_argument("--hypothesis", required=True)
    p_reg.add_argument("--predicted", required=True,
                       help="improved|regressed|flat|arm_effect|no_arm_effect")
    p_reg.add_argument("--selector", required=True, help="JSON selector")
    p_reg.add_argument("--metric", required=True, help='JSON, e.g. {"name":"eval_passrate","n":50}')
    p_reg.add_argument("--decision-rule", required=True,
                       help='JSON, e.g. {"type":"trend_vs_anchor","k_sigma":2}')
    p_reg.add_argument("--anchor", default=None)
    p_reg.add_argument("--motivated-by", default=None)

    p_eval = sub.add_parser("evaluate", help="recompute verdict(s) from current metrics")
    p_eval.add_argument("--id", help="experiment id; omit with --all")
    p_eval.add_argument("--all", action="store_true", help="evaluate every experiment")

    args = ap.parse_args()
    L = Ledger()

    if args.cmd == "show":
        print(f"goal: {L.goal}")
        print(f"checkpoints: {len(L.checkpoints)} | experiments: {len(L.experiments)}\n")
        print(L.inquiry_dag())
        return

    if args.cmd == "read":
        if args.section == "summary":
            out = L.summary()
        elif args.section == "meta":
            out = L.meta
        elif args.section == "checkpoints":
            out = L.checkpoints[args.id] if args.id else L.checkpoints
        elif args.section == "experiments":
            out = L.experiments[args.id] if args.id else L.experiments
        else:  # all
            out = {"meta": L.meta, "checkpoints": L.checkpoints, "experiments": L.experiments}
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "goal":
        if args.set_to:
            L.set_goal(args.set_to); L.save()
        print(L.goal)
        return

    if args.cmd == "frontier":
        print(json.dumps(L.frontier(args.metric, args.top), indent=2))
        return

    if args.cmd == "add-checkpoint":
        node = L.add_checkpoint(args.id, parent=args.parent, config=json.loads(args.config),
                                status=args.status, metrics=json.loads(args.metrics),
                                checkpoint=args.checkpoint)
        L.save()
        print(f"added checkpoint {args.id} (status={node['status']}, parent={node['parent']})")
        return

    if args.cmd == "register":
        exp = L.register(
            args.id, question=args.question, hypothesis=args.hypothesis,
            predicted=args.predicted, selector=json.loads(args.selector),
            metric=json.loads(args.metric), decision_rule=json.loads(args.decision_rule),
            anchor=args.anchor, motivated_by=args.motivated_by)
        L.save()
        print(f"registered {args.id}: resolves {exp['resolved_nodes']}; "
              f"power: {exp['power_check']['note']}")
        return

    if args.cmd == "evaluate":
        eids = list(L.experiments) if args.all else [args.id]
        for eid in eids:
            v = L.evaluate(eid)
            print(f"[{eid}] {v['result'].upper()}  finding={v.get('finding')}"
                  + (f"  ({v['why']})" if v.get("why") else ""))
        L.save()
        return


if __name__ == "__main__":
    _main()
