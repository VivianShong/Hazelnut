"""
Autonomous research agent orchestrator.

Wraps the existing `train.py` experiment loop and exposes a live, observable
workflow state so the progress can be visualized in a UI (see dashboard.py).

The agent loops:
    propose -> patch train.py -> train -> evaluate -> compare -> keep/discard

It tracks the best model seen so far (lowest val_bpb) and can restore the
winning `train.py` configuration on demand.

This module has NO third-party dependencies so it can be served from a tiny
stdlib HTTP server.
"""

from __future__ import annotations

import os
import re
import copy
import time
import threading
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
TRAIN_PY = REPO_DIR / "train.py"
RESULTS_TSV = REPO_DIR / "results.tsv"

# Ordered workflow stages shown in the UI.
STAGES = ["propose", "patch", "train", "evaluate", "compare", "commit"]

# Tunable constants in train.py the agent is allowed to perturb.
# name -> (python type used for formatting)
TUNABLE = {
    "DEPTH": int,
    "DEVICE_BATCH_SIZE": int,
    "MATRIX_LR": float,
    "EMBEDDING_LR": float,
    "UNEMBEDDING_LR": float,
    "WEIGHT_DECAY": float,
    "WINDOW_PATTERN": str,
}


# ---------------------------------------------------------------------------
# train.py reading / patching helpers
# ---------------------------------------------------------------------------

def _const_pattern(name: str) -> re.Pattern:
    # Matches:  NAME = <value>   (with optional inline comment)
    return re.compile(rf"^(?P<pre>{name}\s*=\s*)(?P<val>.+?)(?P<post>\s*(#.*)?)$", re.MULTILINE)


def read_config(content: str) -> dict:
    """Extract the current values of the tunable constants from train.py text."""
    cfg = {}
    for name, typ in TUNABLE.items():
        m = _const_pattern(name).search(content)
        if not m:
            continue
        raw = m.group("val").strip()
        try:
            if typ is str:
                cfg[name] = raw.strip("\"'")
            elif typ is int:
                cfg[name] = int(raw)
            else:
                cfg[name] = float(raw)
        except ValueError:
            cfg[name] = raw
    return cfg


def _format_value(value) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def apply_config(content: str, config: dict) -> str:
    """Return new train.py text with the given constants overwritten."""
    new = content
    for name, value in config.items():
        pat = _const_pattern(name)

        def _repl(m, value=value):
            return f"{m.group('pre')}{_format_value(value)}{m.group('post')}"

        new, n = pat.subn(_repl, new)
        if n == 0:
            raise ValueError(f"Could not find constant '{name}' in train.py")
    return new


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_PROGRESS_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\((?P<pct>[\d.]+)%\).*?"
    r"loss:\s*(?P<loss>[\d.]+).*?"
    r"dt:\s*(?P<dt>\d+)ms.*?"
    r"tok/sec:\s*(?P<tps>[\d,]+).*?"
    r"mfu:\s*(?P<mfu>[\d.]+)%.*?"
    r"epoch:\s*(?P<epoch>\d+).*?"
    r"remaining:\s*(?P<remaining>\d+)s"
)

_SUMMARY_KEYS = {
    "val_bpb": float,
    "training_seconds": float,
    "total_seconds": float,
    "peak_vram_mb": float,
    "mfu_percent": float,
    "total_tokens_M": float,
    "num_steps": int,
    "num_params_M": float,
    "depth": int,
}


def parse_progress(line: str) -> dict | None:
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    return {
        "step": int(m.group("step")),
        "pct": float(m.group("pct")),
        "loss": float(m.group("loss")),
        "dt_ms": int(m.group("dt")),
        "tok_per_sec": int(m.group("tps").replace(",", "")),
        "mfu": float(m.group("mfu")),
        "epoch": int(m.group("epoch")),
        "remaining": int(m.group("remaining")),
    }


def parse_summary_line(line: str, into: dict) -> None:
    if ":" not in line:
        return
    key, _, val = line.partition(":")
    key = key.strip()
    val = val.strip()
    if key in _SUMMARY_KEYS:
        try:
            into[key] = _SUMMARY_KEYS[key](val)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_id: int
    variant: str
    config: dict
    status: str = "running"        # running | keep | discard | crash
    started_at: str = ""
    finished_at: str = ""
    val_bpb: float | None = None
    peak_vram_mb: float | None = None
    training_seconds: float | None = None
    num_params_M: float | None = None
    note: str = ""


class ResearchAgent:
    """Thread-safe orchestrator with an observable workflow state."""

    def __init__(self):
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._proc: subprocess.Popen | None = None

        self.running = False
        self.stage = "idle"
        self.message = "Agent idle. Press Start to begin."
        self.current: RunRecord | None = None
        self.live: dict = {}
        self.history: list[RunRecord] = []
        self.best: RunRecord | None = None
        self._best_train_py: str | None = None
        self.log_tail: list[str] = []
        self._run_counter = 0

        self._ensure_results_header()

    # ---- snapshot for the UI -------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "stage": self.stage,
                "stages": STAGES,
                "message": self.message,
                "current": asdict(self.current) if self.current else None,
                "live": dict(self.live),
                "best": asdict(self.best) if self.best else None,
                "history": [asdict(r) for r in self.history],
                "log_tail": list(self.log_tail)[-40:],
            }

    # ---- lifecycle ------------------------------------------------------
    def start(self, max_runs: int | None = None) -> bool:
        with self._lock:
            if self.running:
                return False
            self._stop_flag.clear()
            self.running = True
            self._thread = threading.Thread(
                target=self._loop, args=(max_runs,), daemon=True
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop_flag.set()
        with self._lock:
            proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()

    def restore_best(self) -> bool:
        """Write the best-performing train.py configuration back to disk."""
        with self._lock:
            if self._best_train_py is None:
                return False
            content = self._best_train_py
        TRAIN_PY.write_text(content, encoding="utf-8")
        self._set(message=f"Restored best config (val_bpb={self.best.val_bpb:.6f}).")
        return True

    # ---- internal state setter -----------------------------------------
    def _set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _log(self, line: str):
        with self._lock:
            self.log_tail.append(line)
            if len(self.log_tail) > 200:
                self.log_tail = self.log_tail[-200:]

    # ---- main loop ------------------------------------------------------
    def _loop(self, max_runs: int | None):
        try:
            runs_done = 0
            while not self._stop_flag.is_set():
                if max_runs is not None and runs_done >= max_runs:
                    break
                self._run_one(is_baseline=(self._run_counter == 0))
                runs_done += 1
        finally:
            self._set(running=False, stage="idle",
                      message="Agent stopped." if self._stop_flag.is_set()
                      else "Agent finished requested runs.")

    def _run_one(self, is_baseline: bool):
        original = TRAIN_PY.read_text(encoding="utf-8")

        # 1) PROPOSE
        self._set(stage="propose")
        if is_baseline:
            config = read_config(original)
            variant = "baseline"
            self._set(message="Establishing baseline (unmodified train.py).")
        else:
            config, variant = self._propose(original)
            self._set(message=f"Proposing variant: {variant}")
        time.sleep(0.3)

        self._run_counter += 1
        record = RunRecord(
            run_id=self._run_counter,
            variant=variant,
            config=config,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        with self._lock:
            self.current = record
            self.live = {}

        # 2) PATCH
        self._set(stage="patch")
        if not is_baseline:
            try:
                patched = apply_config(original, config)
                TRAIN_PY.write_text(patched, encoding="utf-8")
            except ValueError as e:
                record.status = "crash"
                record.note = f"patch failed: {e}"
                self._finish_record(record, original, improved=False)
                return
        time.sleep(0.2)

        # 3) TRAIN + 4) EVALUATE (handled inside train.py)
        self._set(stage="train", message=f"Training: {variant}")
        summary, crashed = self._run_training()

        if crashed or "val_bpb" not in summary:
            record.status = "crash"
            record.note = "training crashed or no val_bpb"
            self._set(stage="compare", message=f"{variant} crashed — reverting.")
            self._finish_record(record, original, improved=False)
            return

        record.val_bpb = summary.get("val_bpb")
        record.peak_vram_mb = summary.get("peak_vram_mb")
        record.training_seconds = summary.get("training_seconds")
        record.num_params_M = summary.get("num_params_M")

        # 5) COMPARE
        self._set(stage="compare")
        improved = self.best is None or record.val_bpb < self.best.val_bpb
        if improved:
            record.status = "keep"
            record.note = "new best" if self.best else "baseline"
            self._set(message=f"{variant}: val_bpb={record.val_bpb:.6f} — NEW BEST!")
        else:
            record.status = "discard"
            record.note = f"worse than best ({self.best.val_bpb:.6f})"
            self._set(message=f"{variant}: val_bpb={record.val_bpb:.6f} — discarded.")
        time.sleep(0.3)

        # 6) COMMIT (keep or revert the file)
        self._set(stage="commit")
        self._finish_record(record, original, improved)

    def _finish_record(self, record: RunRecord, original_train_py: str, improved: bool):
        record.finished_at = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            if improved and record.status == "keep":
                self.best = copy.deepcopy(record)
                self._best_train_py = TRAIN_PY.read_text(encoding="utf-8")
            else:
                # revert train.py to the state before this experiment
                TRAIN_PY.write_text(original_train_py, encoding="utf-8")
            self.history.append(record)
            self.current = None
            self.live = {}
        self._append_results(record)

    # ---- proposal strategy ---------------------------------------------
    def _propose(self, content: str):
        """Perturb the current-best config along one knob."""
        base = read_config(self._best_train_py or content)
        cfg = dict(base)
        knobs = [
            ("DEPTH", lambda v: min(v + 2, 20)),
            ("DEPTH", lambda v: max(v - 2, 2)),
            ("MATRIX_LR", lambda v: round(v * 1.5, 5)),
            ("MATRIX_LR", lambda v: round(v * 0.66, 5)),
            ("EMBEDDING_LR", lambda v: round(v * 1.5, 5)),
            ("WEIGHT_DECAY", lambda v: round(v * 1.5, 5)),
            ("WEIGHT_DECAY", lambda v: round(v * 0.5, 5)),
            ("DEVICE_BATCH_SIZE", lambda v: v * 2),
            ("WINDOW_PATTERN", lambda v: "L" if v != "L" else "SSSL"),
        ]
        idx = (self._run_counter - 1) % len(knobs)
        name, fn = knobs[idx]
        if name in cfg:
            old = cfg[name]
            cfg[name] = fn(old)
            variant = f"{name.lower()}: {old} -> {cfg[name]}"
        else:
            variant = "noop"
        return cfg, variant

    # ---- training subprocess -------------------------------------------
    def _run_training(self):
        summary: dict = {}
        crashed = False
        try:
            proc = subprocess.Popen(
                ["uv", "run", "train.py"],
                cwd=str(REPO_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
        except FileNotFoundError:
            self._log("ERROR: 'uv' not found on PATH.")
            return summary, True

        with self._lock:
            self._proc = proc

        in_summary = False
        for raw in self._iter_chunks(proc.stdout):
            if self._stop_flag.is_set():
                proc.terminate()
                break
            text = raw.strip()
            if not text:
                continue
            prog = parse_progress(text)
            if prog is not None:
                with self._lock:
                    self.live = prog
                    self.stage = "train"
                continue
            if text == "---":
                in_summary = True
                self._set(stage="evaluate", message="Evaluating val_bpb...")
                continue
            if in_summary:
                parse_summary_line(text, summary)
            self._log(text)

        proc.wait()
        with self._lock:
            self._proc = None
        if proc.returncode not in (0, None) and not summary:
            crashed = True
        return summary, crashed

    @staticmethod
    def _iter_chunks(stream):
        """Yield logical lines, splitting on both \\n and \\r (train.py uses \\r)."""
        buf = b""
        while True:
            chunk = stream.read(256)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for p in parts:
                yield p.decode("utf-8", errors="replace")
        if buf:
            yield buf.decode("utf-8", errors="replace")

    # ---- results.tsv ----------------------------------------------------
    def _ensure_results_header(self):
        if not RESULTS_TSV.exists():
            RESULTS_TSV.write_text(
                "run_id\tvariant\tval_bpb\tpeak_vram_mb\tstatus\tnote\n",
                encoding="utf-8",
            )

    def _append_results(self, r: RunRecord):
        line = "\t".join([
            str(r.run_id),
            r.variant.replace("\t", " "),
            f"{r.val_bpb:.6f}" if r.val_bpb is not None else "0.000000",
            f"{r.peak_vram_mb:.1f}" if r.peak_vram_mb is not None else "0.0",
            r.status,
            r.note.replace("\t", " "),
        ])
        with open(RESULTS_TSV, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# A module-level singleton the dashboard imports.
agent = ResearchAgent()
