"""Live dual-tree dashboard for tree.json and ledger.json.

Renders the per-run checkpoint tree (outputs/runs/<run>/tree.json) and the global
ledger's inquiry DAG (outputs/ledger.json) in a single browser UI with card-based
tree visualization, expandable nodes, and live polling.

Usage:
  uv run python dashboard.py
  uv run python dashboard.py --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


STATUS_COLORS = {
    "base": "#455A64",
    "done": "#2E7D32",
    "running": "#1565C0",
    "queued": "#F9A825",
    "failed": "#C62828",
    "skipped": "#6D4C41",
    "held": "#8E24AA",
    "external": "#546E7A",
    "confirmed": "#1B5E20",
    "inconclusive": "#E65100",
    "refuted": "#B71C1C",
    "unknown": "#90A4AE",
}

DEFAULT_TREE = Path("outputs/runs/run-001/tree.json")
DEFAULT_LEDGER = Path("outputs/ledger.json")


@dataclass(frozen=True)
class SourceSpec:
    key: str
    label: str
    path: Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _nodes_for_source(tree: dict, source: str) -> dict[str, dict]:
    """Returns a normalized node dict with parent, status, created_at, and type-specific fields."""
    base_model = tree.get("meta", {}).get("base_model", "base model")

    if source == "tree":
        raw = tree.get("nodes", {})
        nodes: dict[str, dict] = {}
        for nid, n in raw.items():
            metrics = n.get("metrics") or {}
            config = n.get("config") or {}
            nodes[nid] = {
                "id": nid,
                "parent": n.get("parent"),
                "status": n.get("status", "unknown"),
                "type": "checkpoint",
                "created_at": n.get("created_at", ""),
                "label": nid,
                "eval_passrate": metrics.get("eval_passrate"),
                "mean_reward": metrics.get("mean_reward"),
                "train_steps": config.get("train_steps"),
                "kl_coeff": config.get("kl_coeff"),
                "peak_vram_mb": metrics.get("peak_vram_mb"),
            }
    else:
        experiments = tree.get("experiments", {})
        nodes = {}
        for eid, exp in experiments.items():
            verdict = exp.get("verdict") or {}
            evidence = verdict.get("evidence") or {}
            nodes[eid] = {
                "id": eid,
                "parent": exp.get("motivated_by"),
                "status": verdict.get("result") or "queued",
                "type": "experiment",
                "created_at": exp.get("created_at", ""),
                "label": eid,
                "question": exp.get("question", ""),
                "hypothesis": exp.get("hypothesis", ""),
                "finding": verdict.get("finding", ""),
                "anchor": evidence.get("anchor"),
                "best_passrate": evidence.get("best"),
                "delta": evidence.get("delta_vs_anchor"),
                "sigma": evidence.get("sigma"),
                "trajectory": evidence.get("trajectory", []),
            }

    # Insertion order is the chronological order — tree nodes carry no
    # created_at, so this is what replay/layout fall back on for sequencing.
    for seq, node in enumerate(nodes.values()):
        node["seq"] = seq

    nodes["__base__"] = {
        "id": "__base__",
        "parent": None,
        "status": "base",
        "type": "base",
        "created_at": "",
        "seq": -1,
        "label": Path(base_model).name,
    }
    for nid, node in nodes.items():
        if nid != "__base__" and node.get("parent") is None:
            node["parent"] = "__base__"

    return nodes


def _meta_for_source(tree: dict, source: str) -> dict:
    meta = tree.get("meta", {})
    if source == "tree":
        nodes = tree.get("nodes", {})
        experiment = meta.get("experiment") or "Experiment Tree"
        title = _shorten_experiment_title(experiment)
        subtitle = _shorten_experiment_subtitle(experiment)
        counts: dict[str, int] = defaultdict(int)
        for n in nodes.values():
            counts[n.get("status", "unknown")] += 1
        node_count = len(nodes)
    else:
        experiments = tree.get("experiments", {})
        title = "Ledger"
        subtitle = f"{len(experiments)} experiments · inquiry DAG"
        counts = defaultdict(int)
        for exp in experiments.values():
            result = (exp.get("verdict") or {}).get("result") or "queued"
            counts[result] += 1
        node_count = len(experiments)

    counts_text = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "No nodes"
    return {
        "title": title,
        "subtitle": subtitle,
        "node_count": node_count,
        "counts_text": counts_text,
        "extra": "",
        "updated_at": meta.get("updated_at") or "",
    }


def _shorten_experiment_title(text: str) -> str:
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    if " at " in text:
        text = text.split(" at ", 1)[0].strip()
    return text.rstrip("-–: ")


def _shorten_experiment_subtitle(text: str) -> str:
    parts = []
    lower = text.lower()
    if "lr" in lower:
        parts.append("working dose")
    if "steps" in lower:
        parts.append("40 steps")
    if "base anchor" in lower:
        parts.append("base anchor")
    return " · ".join(parts) or "Live experiment tree"


def _html_page(sources: list[SourceSpec], interval_ms: int, initial_source: SourceSpec) -> str:
    tabs_html = "".join(
        f'<button class="tab" data-source="{spec.key}">{spec.label}</button>'
        for spec in sources
    )
    source_json = json.dumps([
        {"key": spec.key, "label": spec.label, "path": str(spec.path)}
        for spec in sources
    ])
    status_colors_json = json.dumps(STATUS_COLORS)
    initial_meta = _meta_for_source(_load_json(initial_source.path), initial_source.key)
    initial_meta_json = json.dumps(initial_meta)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Repurrr Live Dashboard</title>
  <style>
    :root {{
      --bg: #f5f7fb; --panel: #ffffff; --line: #dde5ee;
      --text: #142231; --muted: #617185; --accent: #1565c0; --accent-soft: #eaf2fb;
      --card-w: 280px; --row-h: 152px; --gap-x: 68px; --gap-y: 14px; --pad: 28px;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;
            background: var(--bg); color: var(--text); }}
    /* layout */
    .app {{ display: grid; grid-template-columns: 212px 1fr; height: 100vh; overflow: hidden; }}
    /* sidebar */
    .sidebar {{ border-right: 1px solid var(--line); background: rgba(255,255,255,0.88);
                backdrop-filter: blur(10px); padding: 16px 12px;
                display: flex; flex-direction: column; gap: 14px;
                position: sticky; top: 0; height: 100vh; overflow-y: auto; }}
    .brand h1 {{ font-size: 15px; font-weight: 800; line-height: 1.3; }}
    .brand p  {{ font-size: 11px; color: var(--muted); margin-top: 3px; line-height: 1.4; }}
    .tabs {{ display: flex; flex-direction: column; gap: 5px; }}
    .tab {{ border: 1px solid transparent; border-radius: 9px; padding: 9px 11px;
            text-align: left; font-size: 13px; font-weight: 700;
            background: transparent; color: var(--text); cursor: pointer; transition: 130ms ease; }}
    .tab:hover {{ background: #eef4fb; }}
    .tab.active {{ background: var(--accent-soft); border-color: rgba(21,101,192,0.2); color: var(--accent); }}
    .legend {{ display: flex; flex-direction: column; gap: 5px; }}
    .legend-item {{ display: flex; align-items: center; gap: 7px; font-size: 11px; color: var(--muted); }}
    .legend-dot {{ width: 9px; height: 9px; border-radius: 3px; flex-shrink: 0; }}
    .sidebar-footer {{ margin-top: auto; font-size: 11px; color: var(--muted); line-height: 1.6; }}
    /* main */
    .main {{ padding: 18px 22px; display: flex; flex-direction: column; gap: 10px;
             min-height: 0; min-width: 0; overflow: hidden; }}
    .topbar {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }}
    .headline h2 {{ font-size: 18px; font-weight: 800; }}
    .headline .sub {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
    .topbar-right {{ display: flex; align-items: center; gap: 9px; flex-shrink: 0; padding-top: 2px; }}
    .status {{ color: var(--muted); font-size: 11px; white-space: nowrap; }}
    .replay-btn {{ border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px;
                   font-size: 12px; font-weight: 700; background: var(--panel); color: var(--accent);
                   cursor: pointer; transition: 120ms ease; white-space: nowrap; }}
    .replay-btn:hover {{ background: var(--accent-soft); }}
    .replay-btn:disabled {{ opacity: 0.45; cursor: default; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 6px; min-height: 22px; }}
    .pill {{ display: inline-flex; align-items: center; border-radius: 999px;
             padding: 3px 9px; background: #eef3f8; color: #334659; font-size: 11px; font-weight: 600; }}
    /* frame / canvas */
    .frame {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
              box-shadow: 0 8px 24px rgba(18,33,47,0.05); overflow: hidden;
              flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }}
    .tree-canvas {{ position: relative; overflow: auto; flex: 1 1 auto; min-height: 0; padding: var(--pad); }}
    .tree-inner  {{ position: relative; }}
    .tree-svg    {{ position: absolute; top: 0; left: 0; pointer-events: none; overflow: visible; z-index: 0; }}
    /* cards */
    .node-card {{
      position: absolute; width: var(--card-w); z-index: 1;
      background: var(--panel); border: 1.5px solid var(--line); border-radius: 13px;
      padding: 13px 14px;
      box-shadow: 0 2px 8px rgba(18,33,47,0.06);
      transition: box-shadow 160ms, opacity 380ms, transform 380ms;
    }}
    .node-card:hover {{ box-shadow: 0 6px 22px rgba(18,33,47,0.12); border-color: #c0cfe0; }}
    /* an expanded card overlays its neighbors instead of being hidden behind them */
    .node-card:has(.card-detail[open]) {{
      z-index: 10; box-shadow: 0 12px 32px rgba(18,33,47,0.20); border-color: #c0cfe0;
    }}
    .node-card[data-type="base"] {{ background: #f0f4f8; border-color: #b0bec5; }}
    .node-card.replay-hidden {{ opacity: 0; transform: scale(0.87) translateY(5px); pointer-events: none; }}
    /* replay: the node currently being revealed (the "head") */
    .node-card.replay-head {{
      border-color: var(--accent); box-shadow: 0 0 0 3px rgba(21,101,192,0.22), 0 8px 26px rgba(21,101,192,0.28);
      animation: head-pop 380ms ease;
    }}
    @keyframes head-pop {{ 0% {{ transform: scale(1.06); }} 100% {{ transform: scale(1); }} }}
    /* replay: the earlier branch point we jumped back to */
    .node-card.backtrack-from {{
      border-color: var(--accent); animation: backtrack-pulse 560ms ease-in-out infinite;
    }}
    @keyframes backtrack-pulse {{
      0%, 100% {{ box-shadow: 0 0 0 0 rgba(230,81,0,0.0); }}
      50%      {{ box-shadow: 0 0 0 5px rgba(230,81,0,0.30); }}
    }}
    /* card internals */
    .card-header {{ display: flex; align-items: center; gap: 7px; flex-wrap: wrap; margin-bottom: 6px; }}
    .badge {{
      border-radius: 5px; padding: 2px 7px; font-size: 10px; font-weight: 800;
      text-transform: uppercase; letter-spacing: 0.03em; color: white; flex-shrink: 0;
    }}
    .badge[data-status="queued"] {{ color: #5a3e00; }}
    .card-id {{ font-size: 13px; font-weight: 800; }}
    .card-metric {{ font-size: 11px; color: var(--muted); margin-bottom: 5px; line-height: 1.4; }}
    .card-text {{
      font-size: 12px; color: #334659; line-height: 1.45;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }}
    /* expandable detail */
    .card-detail summary {{
      font-size: 11px; color: var(--accent); cursor: pointer; margin-top: 9px;
      user-select: none; list-style: none; display: flex; align-items: center; gap: 4px;
    }}
    .card-detail summary::-webkit-details-marker {{ display: none; }}
    .card-detail summary::before {{ content: '▶'; font-size: 8px; transition: transform 150ms; }}
    .card-detail[open] summary::before {{ transform: rotate(90deg); }}
    .card-full {{ margin-top: 9px; font-size: 11px; color: #3a4f62; line-height: 1.55; }}
    .card-full .field {{ margin-bottom: 4px; }}
    .card-full .field strong {{ color: var(--text); }}
    .card-full .traj {{ font-family: monospace; font-size: 10px; word-break: break-all; color: #556; }}
    /* SVG edges */
    .edge {{ transition: opacity 380ms; }}
    .edge.backtrack {{
      stroke: #E65100 !important; stroke-width: 2.5px !important;
      stroke-dasharray: 7 5; animation: edge-flow 600ms linear infinite;
    }}
    @keyframes edge-flow {{ to {{ stroke-dashoffset: -24; }} }}
    @media (max-width: 860px) {{
      .app {{ grid-template-columns: 1fr; height: auto; overflow: visible; }}
      .main {{ overflow: visible; }}
      .sidebar {{ position: static; height: auto; border-right: none; border-bottom: 1px solid var(--line); }}
      .tabs {{ flex-direction: row; flex-wrap: wrap; }}
      .tree-canvas {{ min-height: 420px; }}
    }}
  </style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">
      <h1>Repurrr Dashboard</h1>
      <p>Live experiment &amp; checkpoint tree.</p>
    </div>
    <div class="tabs" id="tabs">{tabs_html}</div>
    <div class="legend" id="legend"></div>
    <div class="sidebar-footer">
      <div>Poll: <code>{interval_ms} ms</code></div>
      <div style="margin-top:4px">Click ▶ on a card to expand.</div>
    </div>
  </aside>
  <main class="main">
    <div class="topbar">
      <div class="headline">
        <h2 id="title">loading…</h2>
        <div class="sub" id="sub">—</div>
      </div>
      <div class="topbar-right">
        <div class="status" id="status">Idle</div>
        <button class="replay-btn" id="replay-btn" onclick="replayTree()">&#9654; Replay</button>
      </div>
    </div>
    <div class="pill-row" id="pill-row"></div>
    <div class="frame">
      <div class="tree-canvas" id="tree-canvas">
        <div class="tree-inner" id="tree-inner">
          <svg class="tree-svg" id="tree-svg"></svg>
        </div>
      </div>
    </div>
  </main>
</div>
<script>
  const SOURCES = {source_json};
  const STATUS_COLORS = {status_colors_json};
  const CARD_W = 280, ROW_H = 152, GAP_X = 68, GAP_Y = 14, PAD = 28;

  const titleEl  = document.getElementById('title');
  const subEl    = document.getElementById('sub');
  const statusEl = document.getElementById('status');
  const pillRow  = document.getElementById('pill-row');
  const treeInner = document.getElementById('tree-inner');
  const treeSvg  = document.getElementById('tree-svg');
  const replayBtn = document.getElementById('replay-btn');
  const tabs = Array.from(document.querySelectorAll('.tab'));

  let active = SOURCES[0]?.key || null;
  let currentNodes = null;
  let replayRunning = false;

  // Chronological comparator: created_at, then insertion order (seq), then id.
  // Tree nodes have no created_at, so seq carries the true order there.
  function makeCmp(nodes) {{
    return (a, b) =>
      (nodes[a].created_at || '').localeCompare(nodes[b].created_at || '') ||
      ((nodes[a].seq ?? 0) - (nodes[b].seq ?? 0)) ||
      a.localeCompare(b);
  }}

  // ── Layout ──────────────────────────────────────────────────────────────
  function computeLayout(nodes) {{
    const children = {{}};
    const roots = [];
    for (const [id, node] of Object.entries(nodes)) {{
      const pid = node.parent;
      if (!pid || !(pid in nodes)) {{
        roots.push(id);
      }} else {{
        if (!children[pid]) children[pid] = [];
        children[pid].push(id);
      }}
    }}
    const cmp = makeCmp(nodes);
    for (const list of Object.values(children)) {{
      list.sort(cmp);
    }}
    // BFS depths
    const depths = {{}};
    const queue = [];
    for (const r of roots) {{ depths[r] = 0; queue.push(r); }}
    let qi = 0;
    while (qi < queue.length) {{
      const id = queue[qi++];
      for (const child of (children[id] || [])) {{
        if (!(child in depths)) {{ depths[child] = depths[id] + 1; queue.push(child); }}
      }}
    }}
    // Group by depth
    const byDepth = {{}};
    for (const [id, d] of Object.entries(depths)) {{
      if (!byDepth[d]) byDepth[d] = [];
      byDepth[d].push(id);
    }}
    // Sort within depth by parent's row order then created_at
    for (const [d, ids] of Object.entries(byDepth)) {{
      const prevCol = byDepth[parseInt(d) - 1] || [];
      ids.sort((a, b) => {{
        const pi = prevCol.indexOf(nodes[a]?.parent ?? '');
        const pj = prevCol.indexOf(nodes[b]?.parent ?? '');
        if (pi !== pj) return pi - pj;
        return cmp(a, b);
      }});
    }}
    // Pixel positions
    const pos = {{}};
    for (const [d, ids] of Object.entries(byDepth)) {{
      const col = parseInt(d);
      for (let row = 0; row < ids.length; row++) {{
        pos[ids[row]] = {{ x: col * (CARD_W + GAP_X), y: row * (ROW_H + GAP_Y) }};
      }}
    }}
    return {{ children, pos }};
  }}

  // ── Card rendering ───────────────────────────────────────────────────────
  function fmtN(v, dec) {{ return v == null ? null : v.toFixed(dec); }}

  function buildCardHtml(node) {{
    const color = STATUS_COLORS[node.status] || '#90A4AE';
    const badge = `<span class="badge" data-status="${{node.status}}" style="background:${{color}}">${{node.status}}</span>`;
    const idSpan = `<span class="card-id">${{esc(node.label)}}</span>`;
    let metric = '', bodyText = '', detail = '';

    if (node.type === 'base') {{
      bodyText = `<div class="card-text" style="color:#546e7a">Base model anchor</div>`;
    }} else if (node.type === 'checkpoint') {{
      const parts = [];
      if (node.eval_passrate != null) parts.push(`pass@1 ${{fmtN(node.eval_passrate, 2)}}`);
      if (node.mean_reward    != null) parts.push(`rwd ${{fmtN(node.mean_reward, 3)}}`);
      if (node.train_steps    != null) parts.push(`${{node.train_steps}} steps`);
      if (node.kl_coeff       != null) parts.push(`kl ${{node.kl_coeff}}`);
      metric = parts.length ? `<div class="card-metric">${{parts.join(' · ')}}</div>` : '';
      bodyText = `<div class="card-text">${{esc(node.label)}}</div>`;
      detail = buildCheckpointDetail(node);
    }} else if (node.type === 'experiment') {{
      const parts = [];
      if (node.best_passrate != null) parts.push(`best ${{fmtN(node.best_passrate, 2)}}`);
      if (node.delta != null) parts.push(`Δ${{node.delta >= 0 ? '+' : ''}}${{fmtN(node.delta, 2)}}`);
      if (node.finding) parts.push(node.finding);
      metric = parts.length ? `<div class="card-metric">${{parts.join(' · ')}}</div>` : '';
      bodyText = `<div class="card-text" title="${{esc(node.question)}}">${{esc(node.question)}}</div>`;
      detail = buildExperimentDetail(node);
    }}

    const expandSection = detail
      ? `<details class="card-detail"><summary>Details</summary><div class="card-full">${{detail}}</div></details>`
      : '';

    return `<div class="card-header">${{badge}}${{idSpan}}</div>${{metric}}${{bodyText}}${{expandSection}}`;
  }}

  function buildCheckpointDetail(n) {{
    const rows = [];
    if (n.eval_passrate != null) rows.push(['pass@1', fmtN(n.eval_passrate, 3)]);
    if (n.mean_reward   != null) rows.push(['mean reward', fmtN(n.mean_reward, 4)]);
    if (n.train_steps   != null) rows.push(['train steps', n.train_steps]);
    if (n.kl_coeff      != null) rows.push(['kl coeff', n.kl_coeff]);
    if (n.peak_vram_mb  != null) rows.push(['peak vram', `${{(n.peak_vram_mb / 1024).toFixed(1)}} GB`]);
    return rows.map(([k, v]) => `<div class="field"><strong>${{k}}:</strong> ${{esc(String(v))}}</div>`).join('');
  }}

  function buildExperimentDetail(n) {{
    const rows = [];
    if (n.question)         rows.push(['question',   n.question]);
    if (n.hypothesis)       rows.push(['hypothesis', n.hypothesis]);
    if (n.finding)          rows.push(['finding',    n.finding]);
    if (n.anchor        != null) rows.push(['anchor',      fmtN(n.anchor, 2)]);
    if (n.best_passrate != null) rows.push(['best pass@1', fmtN(n.best_passrate, 2)]);
    if (n.delta         != null) rows.push(['Δ vs anchor', (n.delta >= 0 ? '+' : '') + fmtN(n.delta, 2)]);
    if (n.sigma         != null) rows.push(['σ', fmtN(n.sigma, 3)]);
    let html = rows.map(([k, v]) => `<div class="field"><strong>${{k}}:</strong> ${{esc(String(v))}}</div>`).join('');
    if (n.trajectory?.length) {{
      const traj = n.trajectory.map(([s, p]) => `${{s}}→${{p.toFixed(2)}}`).join('  ');
      html += `<div class="field"><strong>trajectory:</strong><div class="traj">${{esc(traj)}}</div></div>`;
    }}
    return html;
  }}

  function esc(s) {{
    return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }}

  // ── Tree render ──────────────────────────────────────────────────────────
  function renderTree(nodes) {{
    currentNodes = nodes;
    const {{ children, pos }} = computeLayout(nodes);

    // Canvas size
    let maxX = 0, maxY = 0;
    for (const {{ x, y }} of Object.values(pos)) {{
      maxX = Math.max(maxX, x + CARD_W);
      maxY = Math.max(maxY, y + ROW_H);
    }}
    treeInner.style.width  = (maxX + PAD) + 'px';
    treeInner.style.height = (maxY + PAD) + 'px';
    treeSvg.setAttribute('width',  maxX + PAD);
    treeSvg.setAttribute('height', maxY + PAD);

    // Edges
    treeSvg.innerHTML = '';
    for (const [pid, kids] of Object.entries(children)) {{
      if (!(pid in pos)) continue;
      const {{ x: x1, y: y1 }} = pos[pid];
      for (const kid of kids) {{
        if (!(kid in pos)) continue;
        const {{ x: x2, y: y2 }} = pos[kid];
        const ex1 = x1 + CARD_W, ey1 = y1 + ROW_H / 2;
        const ex2 = x2,          ey2 = y2 + ROW_H / 2;
        const cx  = (ex1 + ex2) / 2;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', `M ${{ex1}} ${{ey1}} C ${{cx}} ${{ey1}}, ${{cx}} ${{ey2}}, ${{ex2}} ${{ey2}}`);
        path.setAttribute('stroke', '#B0BEC5');
        path.setAttribute('stroke-width', '2');
        path.setAttribute('fill', 'none');
        path.setAttribute('class', 'edge');
        path.setAttribute('data-to', kid);
        treeSvg.appendChild(path);
      }}
    }}

    // Cards — diff to avoid flicker
    const existing = {{}};
    for (const el of treeInner.querySelectorAll('.node-card')) existing[el.dataset.id] = el;
    const seen = new Set();
    for (const [id, node] of Object.entries(nodes)) {{
      seen.add(id);
      const {{ x, y }} = pos[id] || {{ x: 0, y: 0 }};
      let card = existing[id];
      if (!card) {{
        card = document.createElement('div');
        card.className = 'node-card';
        card.dataset.id   = id;
        card.dataset.type = node.type || '';
        treeInner.appendChild(card);
      }}
      card.style.left = x + 'px';
      card.style.top  = y + 'px';
      // Only rebuild innerHTML when the node data changed — otherwise the poll
      // would recreate the <details> element and collapse an expanded card.
      const sig = JSON.stringify(node);
      if (card.dataset.sig !== sig) {{
        card.innerHTML  = buildCardHtml(node);
        card.dataset.sig = sig;
      }}
    }}
    for (const [id, el] of Object.entries(existing)) {{
      if (!seen.has(id)) el.remove();
    }}
  }}

  // ── Meta / legend ────────────────────────────────────────────────────────
  function renderMeta(payload) {{
    titleEl.textContent = payload.title || '';
    subEl.textContent   = payload.subtitle || '';
    const ts = payload.updated_at
      ? new Intl.DateTimeFormat('en-US', {{ hour: 'numeric', minute: '2-digit', second: '2-digit' }})
          .format(new Date(payload.updated_at))
      : 'now';
    statusEl.textContent = `Updated ${{ts}} · ${{payload.node_count}} nodes`;
    pillRow.innerHTML = '';
    (payload.counts_text || '').split(', ').filter(Boolean).forEach(item => {{
      const span = document.createElement('span');
      span.className   = 'pill';
      span.textContent = item;
      pillRow.appendChild(span);
    }});
  }}

  function buildLegend(nodes) {{
    const statuses = [...new Set(Object.values(nodes).map(n => n.status))];
    document.getElementById('legend').innerHTML = statuses.map(s => {{
      const c = STATUS_COLORS[s] || '#90A4AE';
      return `<div class="legend-item"><div class="legend-dot" style="background:${{c}}"></div>${{s}}</div>`;
    }}).join('');
  }}

  // ── Fetch + tick ─────────────────────────────────────────────────────────
  async function tick() {{
    if (replayRunning || !active) return;
    const source = SOURCES.find(s => s.key === active);
    if (!source) return;
    const t = Date.now();
    try {{
      const [dRes, mRes] = await Promise.all([
        fetch(`/api/data?source=${{encodeURIComponent(active)}}&t=${{t}}`, {{ cache: 'no-store' }}),
        fetch(`/api/meta?source=${{encodeURIComponent(active)}}&t=${{t}}`, {{ cache: 'no-store' }}),
      ]);
      const nodes = await dRes.json();
      const meta  = await mRes.json();
      renderTree(nodes);
      renderMeta(meta);
      buildLegend(nodes);
    }} catch (e) {{
      statusEl.textContent = 'Waiting for file…';
    }}
  }}

  // ── Tab switching ─────────────────────────────────────────────────────────
  function setActiveTab(key) {{
    active = key;
    tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.source === key));
  }}

  tabs.forEach(tab => tab.addEventListener('click', async () => {{
    setActiveTab(tab.dataset.source);
    currentNodes = null;
    treeSvg.innerHTML = '';
    treeInner.querySelectorAll('.node-card').forEach(el => el.remove());
    titleEl.textContent = 'loading…';
    subEl.textContent   = '—';
    await tick();
  }}));

  // ── Replay ────────────────────────────────────────────────────────────────
  async function replayTree() {{
    if (replayRunning || !currentNodes) return;
    replayRunning = true;
    replayBtn.disabled   = true;
    replayBtn.textContent = '⏸ Replaying';

    // Chronological order (created_at, then insertion seq, then id).
    const order = Object.keys(currentNodes)
      .filter(id => id !== '__base__')
      .sort(makeCmp(currentNodes));

    // Hide all non-base cards and edges
    treeInner.querySelectorAll('.node-card:not([data-id="__base__"])').forEach(el => {{
      el.classList.add('replay-hidden');
    }});
    treeSvg.querySelectorAll('.edge').forEach(el => {{
      el.style.opacity = '0';
      el.classList.remove('backtrack');
    }});

    await delay(320);

    // Walk the timeline. A node "backtracks" when its parent is not the node
    // we just revealed — the agent jumped back to an earlier branch point.
    let prevId = '__base__';
    for (let i = 0; i < order.length; i++) {{
      const id     = order[i];
      const parent = currentNodes[id].parent;
      const card   = treeInner.querySelector(`.node-card[data-id="${{id}}"]`);
      const pcard  = parent ? treeInner.querySelector(`.node-card[data-id="${{parent}}"]`) : null;
      const isBacktrack = parent && parent !== prevId && pcard;

      if (isBacktrack) {{
        // Pull focus back to the branch point before sprouting the new node.
        pcard.classList.add('backtrack-from');
        scrollIntoView(pcard);
        const plabel = currentNodes[parent]?.label || parent;
        statusEl.textContent = `↩ backtrack to ${{plabel}}  ·  ${{i + 1}} / ${{order.length}}`;
        await delay(560);
      }}

      if (card) {{
        card.classList.remove('replay-hidden');
        treeInner.querySelectorAll('.node-card.replay-head').forEach(el => el.classList.remove('replay-head'));
        card.classList.add('replay-head');
      }}
      treeSvg.querySelectorAll(`.edge[data-to="${{id}}"]`).forEach(el => {{
        el.style.opacity = '1';
        if (isBacktrack) el.classList.add('backtrack');
      }});
      scrollIntoView(card);
      if (!isBacktrack) statusEl.textContent = `Replaying… ${{i + 1}} / ${{order.length}}`;
      await delay(720);

      if (pcard) pcard.classList.remove('backtrack-from');
      prevId = id;
    }}

    treeInner.querySelectorAll('.node-card.replay-head').forEach(el => el.classList.remove('replay-head'));
    replayRunning = false;
    replayBtn.disabled   = false;
    replayBtn.textContent = '▶ Replay';
    statusEl.textContent  = 'Done';
  }}

  function delay(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

  function scrollIntoView(el) {{
    if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'nearest', inline: 'center' }});
  }}

  // ── Init ──────────────────────────────────────────────────────────────────
  setActiveTab(active);
  renderMeta({initial_meta_json});
  tick();
  setInterval(tick, {interval_ms});
</script>
</body>
</html>"""


class _DashboardHandler(BaseHTTPRequestHandler):
    sources: dict[str, SourceSpec]
    interval_ms: int

    def _source_from_query(self) -> SourceSpec:
        from urllib.parse import parse_qs, urlparse
        query = parse_qs(urlparse(self.path).query)
        key = (query.get("source") or [""])[0]
        if key not in self.sources:
            raise KeyError(key)
        return self.sources[key]

    def _write_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        from urllib.parse import urlparse
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            source_list = list(self.sources.values())
            body = _html_page(source_list, self.interval_ms, source_list[0]).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/data":
            try:
                source = self._source_from_query()
                tree = _load_json(source.path)
                nodes = _nodes_for_source(tree, source.key)
                self._write_json(nodes)
            except Exception as exc:
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(str(exc).encode("utf-8"))
            return

        if path == "/api/meta":
            try:
                source = self._source_from_query()
                tree = _load_json(source.path)
                payload = _meta_for_source(tree, source.key)
                payload.update({"path": str(source.path)})
                self._write_json(payload)
            except Exception as exc:
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(str(exc).encode("utf-8"))
            return

        if path == "/api/replay_count":
            try:
                source = self._source_from_query()
                tree = _load_json(source.path)
                nodes = _nodes_for_source(tree, source.key)
                count = sum(1 for nid in nodes if nid != "__base__")
                self._write_json({"count": count})
            except Exception as exc:
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(str(exc).encode("utf-8"))
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A003
        return


def serve(tree_path: Path, ledger_path: Path, host: str, port: int, interval_ms: int) -> None:
    handler = _DashboardHandler
    handler.sources = {
        "tree":   SourceSpec("tree",   "Experiment Tree", tree_path),
        "ledger": SourceSpec("ledger", "Ledger Tree",     ledger_path),
    }
    handler.interval_ms = interval_ms
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"serving live dashboard at {url}")
    print(f"tree:   {tree_path}")
    print(f"ledger: {ledger_path}")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live tabbed dashboard for tree.json and ledger.json.")
    ap.add_argument("--tree",        type=Path, default=DEFAULT_TREE,   help="path to tree.json")
    ap.add_argument("--ledger",      type=Path, default=DEFAULT_LEDGER, help="path to ledger.json")
    ap.add_argument("--host",        type=str,  default="127.0.0.1",    help="host to bind")
    ap.add_argument("--port",        type=int,  default=8765,           help="port to bind")
    ap.add_argument("--interval-ms", type=int,  default=1500,           help="poll interval")
    args = ap.parse_args()
    serve(args.tree, args.ledger, args.host, args.port, args.interval_ms)


if __name__ == "__main__":
    main()
