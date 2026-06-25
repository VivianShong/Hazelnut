"""Live dual-tree dashboard for tree.json and ledger.json.

Renders the per-run checkpoint tree (outputs/runs/<run>/tree.json) and the global
ledger's inquiry DAG (outputs/ledger.json) in a single browser UI with a left tab
rail and live polling.

Usage:
  uv run python dashboard.py
  uv run python dashboard.py --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_svg import FigureCanvasSVG


STATUS_COLORS = {
    "base": "#455A64",
    "done": "#2E7D32",
    "running": "#1565C0",
    "queued": "#F9A825",
    "failed": "#C62828",
    "skipped": "#6D4C41",
    "held": "#8E24AA",
    "external": "#546E7A",
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


def _build_children(nodes: dict[str, dict]) -> dict[str | None, list[str]]:
    children: dict[str | None, list[str]] = defaultdict(list)
    for node_id, node in nodes.items():
        children[node.get("parent")].append(node_id)
    for ids in children.values():
        ids.sort(key=lambda nid: (nodes[nid].get("created_at", ""), nid))
    return children


def _collect_depths(root_ids: list[str], children: dict[str | None, list[str]]) -> dict[str, int]:
    depths: dict[str, int] = {}

    def visit(node_id: str, depth: int) -> None:
        depths[node_id] = depth
        for child_id in children.get(node_id, []):
            visit(child_id, depth + 1)

    for root_id in root_ids:
        visit(root_id, 0)
    return depths


def _layout(nodes: dict[str, dict]) -> tuple[dict[str, tuple[float, float]], dict[int, list[str]]]:
    children = _build_children(nodes)
    root_ids = children.get(None, [])
    if not root_ids:
        root_ids = sorted(nodes)

    depths = _collect_depths(root_ids, children)
    buckets: dict[int, list[str]] = defaultdict(list)
    for node_id in sorted(nodes, key=lambda nid: (depths.get(nid, 0), nodes[nid].get("created_at", ""), nid)):
        buckets[depths.get(node_id, 0)].append(node_id)

    positions: dict[str, tuple[float, float]] = {}
    y_step = 1.55
    x_step = 2.95
    for depth, ids in buckets.items():
        total = len(ids)
        for index, node_id in enumerate(ids):
            y = (total - 1) * y_step / 2 - index * y_step
            positions[node_id] = (depth * x_step, y)
    return positions, buckets


def _short_label(node_id: str, node: dict) -> str:
    return node_id


def _detail_label(node: dict) -> str:
    status = node.get("status", "unknown")
    if status == "base":
        base_model = node.get("config", {}).get("base_model", "base model")
        return f"start | {Path(base_model).name}"

    metrics = node.get("metrics") or {}
    parts = [status]
    if "eval_passrate" in metrics:
        parts.append(f"pass {metrics['eval_passrate']:.2f}")
    elif "mean_reward" in metrics:
        parts.append(f"rwd {metrics['mean_reward']:.2f}")

    config = node.get("config") or {}
    if "kl_coeff" in config:
        parts.append(f"kl {config['kl_coeff']}")
    if "train_steps" in config:
        parts.append(f"steps {config['train_steps']}")
    return " | ".join(parts)


def _count_statuses(nodes: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for node in nodes.values():
        counts[node.get("status", "unknown")] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _display_nodes(tree: dict, source: str) -> dict[str, dict]:
    if source == "tree":
        nodes = {node_id: dict(node) for node_id, node in tree.get("nodes", {}).items()}
        base_model = tree.get("meta", {}).get("base_model", "base model")
    else:
        nodes = {node_id: dict(node) for node_id, node in tree.get("checkpoints", {}).items()}
        base_model = tree.get("meta", {}).get("base_model", "base model")

    nodes["__base__"] = {
        "id": "__base__",
        "parent": None,
        "status": "base",
        "config": {"base_model": base_model},
        "metrics": {},
        "created_at": "",
    }
    for node_id, node in nodes.items():
        if node_id != "__base__" and node.get("parent") is None:
            node["parent"] = "__base__"
    return nodes


def _meta_for_source(tree: dict, source: str) -> dict:
    if source == "tree":
        meta = tree.get("meta", {})
        nodes = tree.get("nodes", {})
        experiment = meta.get("experiment") or "Experiment Tree"
        return {
            "title": _shorten_experiment_title(experiment),
            "subtitle": _shorten_experiment_subtitle(experiment),
            "node_count": len(nodes),
            "counts_text": ", ".join(f"{k}: {v}" for k, v in _count_statuses(nodes).items()) or "No nodes",
            "extra": "",
            "updated_at": meta.get("updated_at") or "",
        }

    meta = tree.get("meta", {})
    checkpoints = tree.get("checkpoints", {})
    experiments = tree.get("experiments", {})
    return {
        "title": "Ledger Tree",
        "subtitle": f"{len(experiments)} experiments · checkpoint DAG",
        "node_count": len(checkpoints),
        "counts_text": ", ".join(f"{k}: {v}" for k, v in _count_statuses(checkpoints).items()) or "No nodes",
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


def _summary_html(meta: dict) -> str:
    counts = meta.get("counts_text", "")
    pills = "".join(
        f'<span class="pill">{item}</span>'
        for item in counts.split(", ")
        if item
    )
    extra = meta.get("extra", "")
    extra_html = f'<div class="submeta">{extra}</div>' if extra else ""
    return (
        '<div class="summary">'
        f'<div class="summary-title">{meta.get("title", "")}</div>'
        f'<div class="submeta">{meta.get("subtitle", "")}</div>'
        f'<div class="pill-row">{pills}</div>'
        f'{extra_html}'
        '</div>'
    )


def _figure_for_source(tree_path: Path, source: str):
    tree = _load_json(tree_path)
    nodes = _display_nodes(tree, source)
    if not nodes:
        raise ValueError(f"no nodes found in {tree_path}")

    positions, buckets = _layout(nodes)
    children = _build_children(nodes)

    width = max(10.0, (max((depth for depth in buckets.keys()), default=0) + 1) * 3.0)
    height = max(6.0, max((len(ids) for ids in buckets.values()), default=1) * 1.85)
    fig, ax = plt.subplots(figsize=(width, height))

    for parent_id, child_ids in children.items():
        if parent_id is None or parent_id not in positions:
            continue
        x1, y1 = positions[parent_id]
        for child_id in child_ids:
            if child_id not in positions:
                continue
            x2, y2 = positions[child_id]
            ax.plot([x1, x2], [y1, y2], color="#B0BEC5", linewidth=2.0, zorder=1)

    for node_id, (x, y) in positions.items():
        node = nodes[node_id]
        status = node.get("status", "unknown")
        color = STATUS_COLORS.get(status, "#455A64")
        marker = "D" if status == "base" else "o"
        ax.scatter([x], [y], s=760, color=color, marker=marker, edgecolors="white", linewidths=1.8, zorder=3)
        ax.text(x, y + 0.02, _short_label(node_id, node), ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=4)
        ax.text(x, y - 0.52, _detail_label(node), ha="center", va="center",
                color="#263238", fontsize=8.5, zorder=4)

    meta = _meta_for_source(tree, source)
    ax.set_title(meta["title"], fontsize=15, pad=18)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.15)

    legend_items = [
        plt.Line2D([0], [0], marker="o", color="w", label=status, markerfacecolor=color,
                   markeredgecolor="white", markersize=12)
        for status, color in STATUS_COLORS.items()
    ]
    ax.legend(handles=legend_items, loc="upper left", frameon=False, title="Status")

    fig.tight_layout()
    return fig, meta


def _svg_for_source(tree_path: Path, source: str) -> str:
    fig, _ = _figure_for_source(tree_path, source)
    try:
        buffer = io.StringIO()
        FigureCanvasSVG(fig).print_svg(buffer)
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _html_page(sources: list[SourceSpec], interval_ms: int, initial_source: SourceSpec) -> str:
    tabs = "".join(
        f'<button class="tab" data-source="{spec.key}">{spec.label}</button>'
        for spec in sources
    )
    source_json = json.dumps([
        {"key": spec.key, "label": spec.label, "path": str(spec.path)}
        for spec in sources
    ])
    initial_tree = _load_json(initial_source.path)
    initial_meta = _meta_for_source(initial_tree, initial_source.key)
    initial_summary = _summary_html(initial_meta)
    initial_svg_url = f"/api/svg?source={initial_source.key}"

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>GRPO Live Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --line: #dde5ee;
      --text: #142231;
      --muted: #617185;
      --accent: #1565c0;
      --accent-soft: #eaf2fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }}
    .app {{ display: grid; grid-template-columns: 240px 1fr; min-height: 100vh; }}
    .sidebar {{ border-right: 1px solid var(--line); background: rgba(255,255,255,0.8); backdrop-filter: blur(10px); padding: 18px 14px; display: flex; flex-direction: column; gap: 14px; }}
    .brand {{ padding: 6px 8px 10px; }}
    .brand h1 {{ margin: 0; font-size: 18px; line-height: 1.2; }}
    .brand p {{ margin: 6px 0 0; font-size: 13px; color: var(--muted); }}
    .tabs {{ display: flex; flex-direction: column; gap: 10px; }}
    .tab {{ border: 1px solid transparent; border-radius: 14px; padding: 13px 14px; text-align: left; font-size: 14px; font-weight: 700; background: transparent; color: var(--text); cursor: pointer; transition: 160ms ease; }}
    .tab:hover {{ background: #eef4fb; }}
    .tab.active {{ background: var(--accent-soft); border-color: rgba(21,101,192,0.18); color: var(--accent); box-shadow: 0 8px 18px rgba(21,101,192,0.08); }}
    .sidebar-footer {{ margin-top: auto; font-size: 12px; color: var(--muted); line-height: 1.5; }}
    .main {{ padding: 18px; }}
    .topbar {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 14px; }}
    .headline {{ min-width: 0; }}
    .headline h2 {{ margin: 0; font-size: 20px; }}
    .headline .meta {{ margin-top: 5px; color: var(--muted); font-size: 13px; }}
    .status {{ color: var(--muted); font-size: 13px; white-space: nowrap; }}
    .frame {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 12px 30px rgba(18,33,47,0.06); overflow: hidden; }}
    .summary {{ padding: 16px 18px 0; }}
    .summary-title {{ font-size: 18px; font-weight: 800; }}
    .submeta {{ margin-top: 4px; color: var(--muted); font-size: 13px; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 5px 10px; background: #eef3f8; color: #334659; font-size: 12px; }}
    .canvas {{ width: 100%; overflow: auto; padding: 8px 8px 16px; min-height: 320px; }}
    .canvas img {{ display: block; width: 100%; height: auto; min-width: 860px; }}
    code {{ background: #eef3f8; padding: 1px 6px; border-radius: 6px; }}
    @media (max-width: 900px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .tabs {{ flex-direction: row; flex-wrap: wrap; }}
      .tab {{ flex: 1 1 180px; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <h1>Live GRPO Dashboard</h1>
                <p>Track the experiment tree and ledger tree in one live view.</p>
      </div>
      <div class="tabs" id="tabs">{tabs}</div>
      <div class="sidebar-footer">
                <div>Poll interval: <code>{interval_ms}ms</code></div>
                <div>Switch tabs to redraw that source immediately.</div>
      </div>
    </aside>
    <main class="main">
      <div class="topbar">
        <div class="headline">
          <h2 id="title">loading…</h2>
                    <div class="meta" id="meta">Waiting for data…</div>
        </div>
                <div class="status" id="status">Idle</div>
      </div>
      <div class="frame">
                {initial_summary}
                <div class="summary" style="display:none;" id="js-summary">
          <div class="summary-title" id="summary-title"></div>
          <div class="submeta" id="summary-submeta"></div>
          <div class="pill-row" id="summary-pills"></div>
          <div class="submeta" id="summary-extra"></div>
        </div>
                <div class="canvas"><img id="tree" alt="tree visualization" src="{initial_svg_url}" loading="eager" /></div>
      </div>
    </main>
  </div>
  <script>
    const SOURCES = {source_json};
    const tabs = Array.from(document.querySelectorAll('.tab'));
    const img = document.getElementById('tree');
    const title = document.getElementById('title');
    const meta = document.getElementById('meta');
    const status = document.getElementById('status');
    const summaryContainer = document.getElementById('js-summary');
    const summaryTitle = document.getElementById('summary-title');
    const summarySubmeta = document.getElementById('summary-submeta');
    const summaryPills = document.getElementById('summary-pills');
    const summaryExtra = document.getElementById('summary-extra');
    let active = SOURCES[0]?.key || null;

    function setActiveTab(key) {{
      active = key;
      tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.source === key));
    }}

    function renderMeta(payload) {{
        if (summaryContainer) summaryContainer.style.display = 'block';
      title.textContent = payload.title || payload.path || '';
        meta.textContent = payload.subtitle || payload.path || '';
        status.textContent = `Updated ${{new Intl.DateTimeFormat('en-US', {{ hour: 'numeric', minute: '2-digit', second: '2-digit' }}).format(new Date(payload.updated_at || Date.now()))}} • ${{payload.node_count}} nodes`;
      summaryTitle.textContent = payload.title || '';
      summarySubmeta.textContent = payload.subtitle || '';
      summaryPills.innerHTML = '';
      (payload.counts_text || '').split(', ').filter(Boolean).forEach(item => {{
        const span = document.createElement('span');
        span.className = 'pill';
        span.textContent = item;
        summaryPills.appendChild(span);
      }});
      summaryExtra.textContent = payload.extra || '';
    }}

    async function tick() {{
      if (!active) return;
      const source = SOURCES.find(item => item.key === active);
      if (!source) return;
      const stamp = Date.now();
      img.src = `/api/svg?source=${{encodeURIComponent(source.key)}}&t=${{stamp}}`;
      try {{
        const response = await fetch(`/api/meta?source=${{encodeURIComponent(source.key)}}&t=${{stamp}}`, {{ cache: 'no-store' }});
        const payload = await response.json();
        renderMeta(payload);
      }} catch (error) {{
        title.textContent = source.label;
        meta.textContent = source.path;
                status.textContent = 'Waiting for file…';
      }}
    }}

    tabs.forEach(tab => tab.addEventListener('click', async () => {{
      setActiveTab(tab.dataset.source);
      await tick();
    }}));

    setActiveTab(active);
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

        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        key = (query.get("source") or [""])[0]
        if key not in self.sources:
            raise KeyError(key)
        return self.sources[key]

    def _write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
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

        if path == "/api/meta":
            try:
                source = self._source_from_query()
                tree = _load_json(source.path)
                payload = _meta_for_source(tree, source.key)
                payload.update({"path": str(source.path)})
                self._write_json(payload)
            except Exception as exc:  # pragma: no cover - surfaced in browser
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(str(exc).encode("utf-8"))
            return

        if path == "/api/svg":
            try:
                source = self._source_from_query()
                svg = _svg_for_source(source.path, source.key)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(svg.encode("utf-8"))
            except Exception as exc:  # pragma: no cover - surfaced in browser
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
        "tree": SourceSpec("tree", "Experiment Tree", tree_path),
        "ledger": SourceSpec("ledger", "Ledger Tree", ledger_path),
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
    ap.add_argument("--tree", type=Path, default=DEFAULT_TREE, help="path to tree.json")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="path to ledger.json")
    ap.add_argument("--host", type=str, default="127.0.0.1", help="host to bind")
    ap.add_argument("--port", type=int, default=8765, help="port to bind")
    ap.add_argument("--interval-ms", type=int, default=1500, help="poll interval")
    args = ap.parse_args()
    serve(args.tree, args.ledger, args.host, args.port, args.interval_ms)


if __name__ == "__main__":
    main()