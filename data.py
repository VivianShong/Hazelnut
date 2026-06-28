"""Dataset of Python coding problems for GRPO.

Default source is the local **APPS** dataset (HF arrow format) at
`<repo>/data/apps`. APPS does not ship assert-style unit tests; each problem
provides `input_output` (stdin/stdout examples, or call-based args+returns when
`fn_name` is set) plus reference `solutions`. We adapt that into the reward
harness's "io" grading mode (see rewards.py) rather than assert strings.

Schema produced by `load_dataset()`:

    problem = {
        "prompt": str,            # instruction + problem statement shown to model
        "io": {                   # APPS grading spec
            "inputs":  list,      # stdin strings, or arg-lists (call-based)
            "outputs": list,      # expected stdout strings, or return values
            "fn_name": str | None # set -> call-based; None -> stdin/stdout
        },
    }

The dummy hand-written set (assert-style `"tests"`) is kept as a fallback for
offline smoke tests: `load_dataset(split, source="dummy")`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Absolute path: stable whether this file lives in the worktree or repo root after merge.
DATA_ROOT = Path("/opt/Hazelnut/data")
APPS_DIR = DATA_ROOT / "apps"
MBPP_DIR = DATA_ROOT / "mbpp"  # cached HF dataset (created on first load)

# --------------------------------------------------------------------------
# Chain-of-thought prompt toggle (experiment knob, not a code change to the
# GRPO loop / reward harness). When env var GRPO_COT is truthy, prompts ask the
# model to reason step-by-step in plain text *before* emitting the single final
# ```python``` block. rewards.extract_code() takes the FIRST python fence, so
# the reasoning text is free and is reinforced only when it yields correct code.
# --------------------------------------------------------------------------

_COT_INSTRUCTION = (
    "First, think step by step about the approach in plain text "
    "(edge cases, algorithm, the function signature). Do NOT write any code "
    "during this reasoning. Then, on a new line, give the complete final "
    "solution in a single ```python code block."
)


def _cot_enabled() -> bool:
    return os.environ.get("GRPO_COT", "").strip().lower() in ("1", "true", "yes", "on")


def _maybe_cot(prompt: str) -> str:
    """Append the chain-of-thought reasoning instruction when GRPO_COT is set."""
    if _cot_enabled():
        return prompt + "\n\n" + _COT_INSTRUCTION
    return prompt

# --------------------------------------------------------------------------
# Dummy fallback set (assert-style)
# --------------------------------------------------------------------------

_DUMMY: list[dict] = [
    {"prompt": "Write a Python function `add(a, b)` that returns the sum of two numbers.",
     "tests": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]},
    {"prompt": "Write a Python function `is_even(n)` that returns True if n is even, else False.",
     "tests": ["assert is_even(4) is True", "assert is_even(7) is False", "assert is_even(0) is True"]},
    {"prompt": "Write a Python function `reverse_string(s)` that returns the reversed string.",
     "tests": ["assert reverse_string('abc') == 'cba'", "assert reverse_string('') == ''", "assert reverse_string('a') == 'a'"]},
    {"prompt": "Write a Python function `factorial(n)` that returns n! for non-negative n.",
     "tests": ["assert factorial(0) == 1", "assert factorial(5) == 120", "assert factorial(1) == 1"]},
    {"prompt": "Write a Python function `max_of_list(xs)` that returns the largest element of a non-empty list.",
     "tests": ["assert max_of_list([1, 2, 3]) == 3", "assert max_of_list([-5, -2, -9]) == -2", "assert max_of_list([42]) == 42"]},
    {"prompt": "Write a Python function `count_vowels(s)` that returns the number of vowels (aeiou) in s.",
     "tests": ["assert count_vowels('hello') == 2", "assert count_vowels('xyz') == 0", "assert count_vowels('aeiou') == 5"]},
    {"prompt": "Write a Python function `fib(n)` that returns the nth Fibonacci number (fib(0)=0, fib(1)=1).",
     "tests": ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(10) == 55"]},
    {"prompt": "Write a Python function `is_palindrome(s)` that returns True if s reads the same forwards and backwards.",
     "tests": ["assert is_palindrome('racecar') is True", "assert is_palindrome('hello') is False", "assert is_palindrome('') is True"]},
]


# --------------------------------------------------------------------------
# APPS loader
# --------------------------------------------------------------------------

def _build_prompt(question: str, starter_code: str, fn_name: str | None) -> str:
    if fn_name:
        instr = (f"Solve the following programming problem in Python by implementing "
                 f"the function `{fn_name}`.")
    else:
        instr = ("Solve the following programming problem in Python. Read the input from "
                 "standard input (stdin) and print the answer to standard output (stdout).")
    parts = [instr, "\nProblem:\n" + question.strip()]
    if starter_code and starter_code.strip():
        parts.append("\nUse this starter code:\n```python\n" + starter_code.strip() + "\n```")
    parts.append("\nProvide a complete solution in a single ```python code block.")
    return _maybe_cot("\n".join(parts))


def _row_to_problem(row: dict) -> dict | None:
    """Convert one APPS row into a {prompt, io} problem, or None if unusable."""
    raw = row.get("input_output") or ""
    try:
        io = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    inputs, outputs = io.get("inputs") or [], io.get("outputs") or []
    if not inputs or not outputs:
        return None
    fn_name = io.get("fn_name")
    return {
        "prompt": _build_prompt(row.get("question", ""), row.get("starter_code") or "", fn_name),
        "io": {"inputs": inputs, "outputs": outputs, "fn_name": fn_name},
    }


def _load_apps(split: str, limit: int | None, difficulty: str | None) -> list[dict]:
    from datasets import load_from_disk  # added dep; only imported when needed

    hf_split = "train" if split == "train" else "test"
    ds = load_from_disk(str(APPS_DIR))[hf_split]
    problems: list[dict] = []
    for row in ds:
        if difficulty and row.get("difficulty") != difficulty:
            continue
        p = _row_to_problem(row)
        if p is None:
            continue
        problems.append(p)
        if limit and len(problems) >= limit:
            break
    return problems


# --------------------------------------------------------------------------
# MBPP loader (training set: simple function-writing tasks with assert tests)
# --------------------------------------------------------------------------

def _build_mbpp_prompt(text: str, test_list: list[str]) -> str:
    # Showing the asserts is the MBPP convention: it reveals the expected
    # function name/signature the solution must define.
    tests = "\n".join(test_list)
    return _maybe_cot(
        "Write Python code to solve the following task.\n\n"
        f"Task:\n{text.strip()}\n\n"
        f"Your code must pass these tests:\n{tests}\n\n"
        "Provide the solution in a single ```python code block."
    )


def _load_mbpp(split: str, limit: int | None) -> list[dict]:
    from datasets import load_dataset as hf_load_dataset, load_from_disk

    if MBPP_DIR.exists():
        ds = load_from_disk(str(MBPP_DIR))
    else:
        ds = hf_load_dataset("mbpp", "full")
        try:
            ds.save_to_disk(str(MBPP_DIR))  # cache locally for offline reuse
        except Exception:
            pass

    hf_split = "train" if split == "train" else "test"
    rows = ds[hf_split]
    problems: list[dict] = []
    for r in rows:
        tests = r.get("test_list") or []
        if not tests:
            continue
        problems.append({
            "prompt": _build_mbpp_prompt(r.get("text", ""), tests),
            "tests": tests,
            "setup": r.get("test_setup_code", "") or "",
        })
        if limit and len(problems) >= limit:
            break
    return problems


def load_dataset(
    split: str = "train",
    source: str = "mbpp",
    limit: int | None = 256,
    difficulty: str | None = None,
    train_frac: float = 0.75,
) -> list[dict]:
    """Return a list of coding problems for a split.

    source="mbpp"  -> MBPP (default training set; assert-style, achievable for a 2B model).
    source="apps"  -> local APPS dataset (the hard benchmark); `difficulty` filters
                      to introductory/interview/competition.
    source="dummy" -> the hand-written assert-style set (offline smoke).
    Falls back to the dummy set if the requested source is unavailable.
    `limit` caps how many problems are materialised.
    """
    assert split in ("train", "eval"), split

    try:
        if source == "mbpp":
            problems = _load_mbpp(split, limit)
        elif source == "apps":
            problems = _load_apps(split, limit, difficulty)
        else:
            problems = []
        if problems:
            return problems
        if source in ("mbpp", "apps"):
            print(f"data: {source} yielded no usable problems; falling back to dummy set")
    except Exception as exc:  # missing dataset / datasets pkg / offline
        print(f"data: {source} unavailable ({exc}); falling back to dummy set")

    n_train = max(1, int(len(_DUMMY) * train_frac))
    return _DUMMY[:n_train] if split == "train" else _DUMMY[n_train:]
