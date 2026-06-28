"""Deterministic reward harness for the "write good Python code" task.

A completion is scored by a deterministic function (never an LLM), and the
*component* scores are returned alongside the scalar so the research agent can
reason about WHY a config improved, not just that it did.

Components
- compiles:    1.0 if the extracted code parses + byte-compiles, else 0.0
- correctness: fraction of the problem's test cases that pass (sandboxed)
- style:       1 / (1 + ruff_violations)        (1.0 if ruff unavailable)
- security:    severity-weighted bandit penalty  (1.0 if bandit unavailable)

scalar reward = weighted sum of the components (weights configurable).

Two grading modes (a problem carries exactly one):
- "tests": list[str] of standalone assert statements (the dummy/MBPP style).
- "io":    {"inputs": [...], "outputs": [...], "fn_name": str | None} — the APPS
           style. With fn_name -> call-based (call fn(*args), compare return).
           Without -> stdin/stdout (feed input to a script, compare its stdout).

The code runs in a subprocess with a wall-clock timeout. This is a hackathon
sandbox, not hardened isolation — don't run untrusted models on a shared host
without OS-level sandboxing.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

# Default component weights -> scalar reward. Correctness dominates.
DEFAULT_WEIGHTS = {
    "compiles": 0.15,
    "correctness": 0.65,
    "style": 0.10,
    "security": 0.10,
}

# APPS problems carry hundreds of test cases; grading them all per completion is
# far too slow for an RL loop. Cap to the first N (still a strong reward signal).
DEFAULT_MAX_CASES = 8

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class RewardBreakdown:
    reward: float
    compiles: float
    correctness: float
    style: float
    security: float
    n_tests: int
    n_passed: int
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def extract_code(completion: str) -> str:
    """Pull a code block out of a model completion.

    Prefers the first fenced ```python block; falls back to the whole string
    (stripped of a stray leading 'python' line) when no fence is present.
    """
    blocks = _FENCE_RE.findall(completion)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks).strip()
    text = completion.strip()
    if text.startswith("python\n"):
        text = text[len("python\n"):]
    return text


def _compiles(code: str) -> bool:
    try:
        ast.parse(code)
        compile(code, "<candidate>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


# --------------------------------------------------------------------------
# Grading mode 1: assert-style tests (dummy / MBPP)
# --------------------------------------------------------------------------

# Harnesses read their payload from a sibling payload.json (json.load), never
# from source-embedded strings — embedding code via str literals re-interprets
# \n/\t escapes and corrupts the payload. They print a single JSON result line.

_ASSERT_HARNESS = """
import json
with open("payload.json") as f:
    p = json.load(f)
tests = p["tests"]
ns = {}
try:
    if p.get("setup"):
        exec(p["setup"], ns)
    exec(p["code"], ns)
except Exception as e:
    print(json.dumps({"passed": 0, "total": len(tests), "error": f"exec failed: {e}"}))
    raise SystemExit(0)
passed = 0
err = None
for t in tests:
    try:
        exec(t, ns)
        passed += 1
    except Exception as e:
        if err is None:
            err = f"{type(e).__name__}: {e}"
print(json.dumps({"passed": passed, "total": len(tests), "error": err}))
"""


def _run_harness(td: Path, harness: str, payload: dict, timeout: float, total: int):
    (td / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    (td / "harness.py").write_text(harness, encoding="utf-8")
    try:
        proc = subprocess.run([sys.executable, "harness.py"],
                              capture_output=True, text=True, timeout=timeout, cwd=td)
    except subprocess.TimeoutExpired:
        return 0, total, "timeout"
    try:
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        return r["passed"], r["total"], r.get("error")
    except (json.JSONDecodeError, IndexError, KeyError):
        return 0, total, (proc.stderr.strip()[-300:] or "no output")


def _run_assert_tests(code: str, tests: list[str], timeout: float, setup: str = "") -> tuple[int, int, str | None]:
    if not tests:
        return 0, 0, None
    with tempfile.TemporaryDirectory() as td:
        return _run_harness(Path(td), _ASSERT_HARNESS,
                            {"setup": setup or "", "code": code, "tests": tests},
                            timeout, len(tests))


# --------------------------------------------------------------------------
# Grading mode 2: APPS input/output
# --------------------------------------------------------------------------

def _norm_stdout(s: str) -> str:
    """APPS-style output normalisation: rstrip each line, drop trailing blanks."""
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _expected_str(expected) -> str:
    if isinstance(expected, list):
        expected = "\n".join(str(x) for x in expected)
    return _norm_stdout(str(expected))


def _run_stdin_tests(code, inputs, outputs, timeout, max_cases) -> tuple[int, int, str | None]:
    """Run the candidate as a script, feeding each input to stdin."""
    n = min(len(inputs), len(outputs), max_cases)
    passed, err = 0, None
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "sol.py"
        script.write_text(code, encoding="utf-8")
        for i in range(n):
            stdin = inputs[i] if isinstance(inputs[i], str) else "\n".join(map(str, inputs[i]))
            try:
                proc = subprocess.run([sys.executable, str(script)], input=stdin,
                                      capture_output=True, text=True, timeout=timeout, cwd=td)
            except subprocess.TimeoutExpired:
                err = err or "timeout"
                continue
            if proc.returncode != 0:
                err = err or (proc.stderr.strip()[-200:] or "nonzero exit")
                continue
            if _norm_stdout(proc.stdout) == _expected_str(outputs[i]):
                passed += 1
    return passed, n, err


_CALL_HARNESS = """
import json
with open("payload.json") as f:
    p = json.load(f)
inputs, outputs, fn_name = p["inputs"], p["outputs"], p["fn_name"]
ns = {}
try:
    exec(p["code"], ns)
except Exception as e:
    print(json.dumps({"passed": 0, "total": len(inputs), "error": f"exec failed: {e}"}))
    raise SystemExit(0)
fn = ns.get(fn_name)
if fn is None:
    print(json.dumps({"passed": 0, "total": len(inputs), "error": f"missing fn {fn_name}"}))
    raise SystemExit(0)
passed = 0
err = None
for args, exp in zip(inputs, outputs):
    if not isinstance(args, list):
        args = [args]
    # APPS often wraps the single expected return in a 1-element list
    expected = exp[0] if (isinstance(exp, list) and len(exp) == 1) else exp
    try:
        got = fn(*args)
        if got == expected or got == exp:
            passed += 1
        elif err is None:
            err = "wrong answer"
    except Exception as e:
        if err is None:
            err = f"{type(e).__name__}: {e}"
print(json.dumps({"passed": passed, "total": len(inputs), "error": err}))
"""


def _run_call_tests(code, inputs, outputs, fn_name, timeout, max_cases) -> tuple[int, int, str | None]:
    n = min(len(inputs), len(outputs), max_cases)
    with tempfile.TemporaryDirectory() as td:
        return _run_harness(
            Path(td), _CALL_HARNESS,
            {"code": code, "fn_name": fn_name, "inputs": inputs[:n], "outputs": outputs[:n]},
            timeout * n, n)


def _run_io_tests(code, io, timeout, max_cases) -> tuple[int, int, str | None]:
    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    if not inputs or not outputs:
        return 0, 0, "no io cases"
    fn_name = io.get("fn_name")
    if fn_name:
        return _run_call_tests(code, inputs, outputs, fn_name, timeout, max_cases)
    return _run_stdin_tests(code, inputs, outputs, timeout, max_cases)


# --------------------------------------------------------------------------
# Optional style / security components
# --------------------------------------------------------------------------

def _style_score(code: str, timeout: float) -> float:
    """1 / (1 + ruff_violations). Returns 1.0 if ruff is unavailable."""
    if shutil.which("ruff") is None:
        return 1.0
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "c.py"
        f.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(["ruff", "check", "--output-format=json", str(f)],
                                  capture_output=True, text=True, timeout=timeout)
            violations = len(json.loads(proc.stdout or "[]"))
        except Exception:
            return 1.0
    return 1.0 / (1.0 + violations)


def _security_score(code: str, timeout: float) -> float:
    """Severity-weighted bandit penalty in [0,1]. Returns 1.0 if bandit unavailable."""
    if shutil.which("bandit") is None:
        return 1.0
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "c.py"
        f.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(["bandit", "-f", "json", "-q", str(f)],
                                  capture_output=True, text=True, timeout=timeout)
            report = json.loads(proc.stdout or "{}")
        except Exception:
            return 1.0
    weights = {"LOW": 1, "MEDIUM": 3, "HIGH": 8}
    penalty = sum(weights.get(r.get("issue_severity", "LOW"), 1)
                  for r in report.get("results", []))
    return 1.0 / (1.0 + penalty)


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------

def score_completion(
    completion: str,
    problem: dict,
    weights: dict | None = None,
    timeout: float = 6.0,
    max_cases: int = DEFAULT_MAX_CASES,
) -> RewardBreakdown:
    """Score one model completion against a problem. Deterministic.

    `problem` carries either "tests" (list of assert strings) or "io"
    (APPS-style inputs/outputs, optionally with fn_name).
    """
    weights = weights or DEFAULT_WEIGHTS
    code = extract_code(completion)

    compiles = 1.0 if _compiles(code) else 0.0
    if compiles == 0.0:
        return RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, error="does not compile")

    if problem.get("io"):
        n_passed, n_tests, err = _run_io_tests(code, problem["io"], timeout, max_cases)
    else:
        n_passed, n_tests, err = _run_assert_tests(
            code, problem.get("tests", []), timeout, problem.get("setup", ""))
    correctness = (n_passed / n_tests) if n_tests else 0.0
    style = _style_score(code, timeout)
    security = _security_score(code, timeout)

    reward = (
        weights["compiles"] * compiles
        + weights["correctness"] * correctness
        + weights["style"] * style
        + weights["security"] * security
    )
    return RewardBreakdown(reward, compiles, correctness, style, security,
                           n_tests, n_passed, err)
