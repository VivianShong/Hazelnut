"""
Tiny dependency-free dashboard server for the autoresearch agent.

Run:
    uv run dashboard.py            # then open http://localhost:8000

Exposes:
    GET  /                  -> dashboard UI
    GET  /static/<file>     -> static assets
    GET  /api/status        -> JSON snapshot of the agent workflow
    POST /api/start         -> start the agent loop   (body: {"max_runs": N|null})
    POST /api/stop          -> stop the agent loop
    POST /api/restore_best  -> write best train.py config back to disk
"""

from __future__ import annotations

import json
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import agent

REPO_DIR = Path(__file__).resolve().parent
STATIC_DIR = REPO_DIR / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    # Silence default noisy logging.
    def log_message(self, *args):
        pass

    # ---- helpers --------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _serve_file(self, path: Path):
        if not path.is_file():
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        ctype = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self._send(200, path.read_bytes(), ctype)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    # ---- routes ---------------------------------------------------------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html")
        elif self.path.startswith("/static/"):
            rel = self.path[len("/static/"):].split("?", 1)[0]
            self._serve_file(STATIC_DIR / rel)
        elif self.path == "/api/status":
            self._send_json(agent.snapshot())
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path == "/api/start":
            body = self._read_body()
            max_runs = body.get("max_runs")
            started = agent.start(max_runs=max_runs)
            self._send_json({"ok": started})
        elif self.path == "/api/stop":
            agent.stop()
            self._send_json({"ok": True})
        elif self.path == "/api/restore_best":
            ok = agent.restore_best()
            self._send_json({"ok": ok})
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")


def main():
    parser = argparse.ArgumentParser(description="Autoresearch agent dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        agent.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
