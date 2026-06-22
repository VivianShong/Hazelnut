"""
Code-quality reward for the GRPO "write good Python" environment.

A single generated sample is scored on three axes, each mapped into [0, 1]:

1. Compilation   — Python's built-in ``compile()`` (the built-in compiler).
                   Counts hard syntax errors (fatal) and ``SyntaxWarning`` /
                   ``DeprecationWarning`` raised during compilation.
2. Coding standard — ``pyflakes`` (logical issues: undefined names, unused
                   imports, redefinitions, ...) + ``pycodestyle`` (PEP 8 style).
3. Security      — ``bandit`` static analysis, weighted by issue severity.

The three component scores are combined into a single scalar reward in
``[-1, 1]`` via a weighted sum. Code that does not compile short-circuits to a
large negative reward, because the other analyzers cannot run on it.

This module has no torch / model dependencies, so it can be unit-tested and
run on its own:

    python -m grpo.code_reward
"""

from __future__ import annotations

import io
import os
import re
import math
import tempfile
import warnings
import contextlib
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Tunable weights / decay constants
# ---------------------------------------------------------------------------

# How the three component scores are weighted in the final reward.
W_COMPILE = 0.40
W_STANDARD = 0.35
W_SECURITY = 0.25

# Reward returned when the code fails to compile at all (worst case).
SYNTAX_ERROR_REWARD = -1.0

# Reward returned when no Python code could be extracted from the sample.
NO_CODE_REWARD = -1.0

# Exponential-decay sharpness: score = exp(-k * weighted_issue_count).
# Larger k => issues are punished more aggressively.
K_COMPILE_WARNING = 0.50
K_STANDARD = 0.08
K_SECURITY = 0.60

# Bandit severity weights.
BANDIT_SEVERITY_WEIGHT = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0}


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull Python source out of a model completion.

    Prefers the contents of the first fenced ``` ```python ``` ``` block. If no
    fenced block is present, the raw text is returned (the model may have
    emitted bare code). Returns an empty string only if there is nothing.
    """
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# 1. Compilation (built-in compiler)
# ---------------------------------------------------------------------------


def check_compile(code: str) -> tuple[bool, int]:
    """Compile ``code`` with the built-in ``compile()``.

    Returns ``(ok, num_warnings)`` where ``ok`` is False on a fatal
    ``SyntaxError`` and ``num_warnings`` counts warnings (e.g.
    ``SyntaxWarning``) emitted while compiling.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError:
            return False, 0
        except ValueError:
            # e.g. source contains null bytes — treat as a fatal failure.
            return False, 0
        return True, len(caught)


# ---------------------------------------------------------------------------
# 2. Coding standard (pyflakes + pycodestyle)
# ---------------------------------------------------------------------------


def count_pyflakes(code: str) -> int:
    """Number of pyflakes findings (undefined names, unused imports, ...)."""
    try:
        from pyflakes import api as pyflakes_api
        from pyflakes.reporter import Reporter
    except ImportError:
        return 0

    sink = io.StringIO()
    reporter = Reporter(sink, sink)
    # api.check returns the number of reported messages.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        count = pyflakes_api.check(code, "<generated>", reporter=reporter)
    return int(count)


def count_pycodestyle(code: str) -> int:
    """Number of PEP 8 violations reported by pycodestyle."""
    try:
        import pycodestyle
    except ImportError:
        return 0

    lines = code.splitlines(keepends=True)
    if not lines:
        return 0
    style = pycodestyle.StyleGuide(quiet=True)
    checker = pycodestyle.Checker(lines=lines, options=style.options)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        checker.check_all()
    return int(checker.report.total_errors)


# ---------------------------------------------------------------------------
# 3. Security (bandit)
# ---------------------------------------------------------------------------


def count_bandit(code: str) -> float:
    """Severity-weighted count of bandit security issues.

    Bandit scans files, so the snippet is written to a temp file and analyzed
    in-process. Degrades gracefully (returns 0.0) if bandit is unavailable or
    its API changes.
    """
    try:
        from bandit.core import config as b_config
        from bandit.core import manager as b_manager
    except ImportError:
        return 0.0

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        conf = b_config.BanditConfig()
        mgr = b_manager.BanditManager(conf, "file", quiet=True)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            mgr.discover_files([tmp_path], recursive=False)
            mgr.run_tests()
        issues = mgr.get_issue_list()
        weighted = 0.0
        for issue in issues:
            severity = str(getattr(issue, "severity", "LOW")).upper()
            weighted += BANDIT_SEVERITY_WEIGHT.get(severity, 1.0)
        return weighted
    except Exception:
        # Any internal bandit failure should not crash training.
        return 0.0
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class CodeEval:
    """Full breakdown of a single sample's evaluation."""

    has_code: bool
    compiles: bool
    compile_warnings: int
    pyflakes_issues: int
    pycodestyle_issues: int
    bandit_weighted: float
    score_compile: float
    score_standard: float
    score_security: float
    reward: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_code(completion: str) -> CodeEval:
    """Evaluate a single model completion and return a full breakdown."""
    code = extract_code(completion)

    if not code:
        return CodeEval(
            has_code=False,
            compiles=False,
            compile_warnings=0,
            pyflakes_issues=0,
            pycodestyle_issues=0,
            bandit_weighted=0.0,
            score_compile=0.0,
            score_standard=0.0,
            score_security=0.0,
            reward=NO_CODE_REWARD,
        )

    compiles, n_warnings = check_compile(code)

    if not compiles:
        # Cannot run the other analyzers on non-compiling code.
        return CodeEval(
            has_code=True,
            compiles=False,
            compile_warnings=0,
            pyflakes_issues=0,
            pycodestyle_issues=0,
            bandit_weighted=0.0,
            score_compile=0.0,
            score_standard=0.0,
            score_security=0.0,
            reward=SYNTAX_ERROR_REWARD,
        )

    pyflakes_issues = count_pyflakes(code)
    pycodestyle_issues = count_pycodestyle(code)
    bandit_weighted = count_bandit(code)

    # Component scores in [0, 1] via exponential decay on issue counts.
    score_compile = math.exp(-K_COMPILE_WARNING * n_warnings)
    standard_count = pyflakes_issues + 0.25 * pycodestyle_issues
    score_standard = math.exp(-K_STANDARD * standard_count)
    score_security = math.exp(-K_SECURITY * bandit_weighted)

    # Weighted sum in [0, 1], then mapped to [-1, 1] so that a flawless sample
    # scores +1 and a barely-compiling, insecure mess approaches the floor.
    quality = (
        W_COMPILE * score_compile
        + W_STANDARD * score_standard
        + W_SECURITY * score_security
    )
    reward = 2.0 * quality - 1.0

    return CodeEval(
        has_code=True,
        compiles=True,
        compile_warnings=n_warnings,
        pyflakes_issues=pyflakes_issues,
        pycodestyle_issues=pycodestyle_issues,
        bandit_weighted=bandit_weighted,
        score_compile=score_compile,
        score_standard=score_standard,
        score_security=score_security,
        reward=reward,
    )


def reward_fn(completion: str) -> float:
    """Scalar reward in ``[-1, 1]`` for a single completion (GRPO entry point)."""
    return evaluate_code(completion).reward


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_SAMPLES = {
    "clean": '''```python
def add(a, b):
    """Return the sum of two numbers."""
    return a + b
''' + "```",
    "syntax_error": "```python\ndef broken(:\n    return 1\n```",
    "unused_import_and_style": "```python\nimport os\nx=1\ny =2\n```",
    "insecure": (
        "```python\n"
        "import subprocess\n"
        "def run(cmd):\n"
        "    return subprocess.call(cmd, shell=True)\n"
        "```"
    ),
}


def _selftest() -> None:
    for name, sample in _SAMPLES.items():
        result = evaluate_code(sample)
        print(f"=== {name} ===")
        for key, value in result.as_dict().items():
            print(f"  {key}: {value}")
        print()


if __name__ == "__main__":
    _selftest()
