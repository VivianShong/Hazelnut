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

LEDGER = Path(__file__).resolve().parent / "ledger.json"


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
        self.checkpoints = d["checkpoints"]
        self.experiments = d["experiments"]

    # ---------------- persistence ----------------
    def save(self) -> None:
        self.meta["updated_at"] = _now()
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


if __name__ == "__main__":
    import sys
    L = Ledger()
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        print(f"checkpoints: {len(L.checkpoints)} | experiments: {len(L.experiments)}\n")
        print(L.inquiry_dag())
