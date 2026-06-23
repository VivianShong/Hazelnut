#!/usr/bin/env python3
"""Evaluate LoRA-trained coding models on APPS with deterministic heuristics.

Dataset: https://huggingface.co/datasets/codeparrot/apps

What this file is for
---------------------
This script is the project's coding evaluation harness. After a small model has
been fine-tuned with LoRA, this script checks whether the trained adapter is
actually better than the original base model on LeetCode-like programming tasks.

The evaluator is intentionally deterministic and execution-based:

1. Load a fixed subset of APPS problems from Hugging Face.
2. Ask a model to generate Python code for each problem, or read pre-generated
   candidate code from a JSONL file.
3. Extract executable Python from the model answer.
4. Run the candidate against APPS test inputs/outputs in a subprocess.
5. Mark each problem as pass, wrong answer, compile error, runtime error, or
   timeout.
6. Aggregate intuitive metrics:
   - strict_accuracy: fraction of problems where every test passed
   - mean_test_pass_rate: average fraction of tests passed per problem
   - compile_error_rate
   - timeout_problem_rate
   - runtime_error_problem_rate
7. In --compare-adapter mode, run the same fixed problem subset twice. By
   default this compares:
   - base model only, also called v0
   - base model + LoRA adapter, e.g. v1
   If --baseline-adapter is supplied, it instead compares:
   - base model + baseline LoRA adapter, e.g. v1
   - base model + candidate LoRA adapter, e.g. v1.1
   Then write a comparison verdict: improved, regressed, or same.

Why this matters for the self-improvement loop
----------------------------------------------
The Research Agent needs a clear, repeatable signal to decide whether a LoRA
training run should be kept or discarded. This script produces that signal. It
turns "the model feels better" into a deterministic before/after comparison:

  base Qwen score  vs.  base Qwen + LoRA adapter score

or, for tree branches:

  parent adapter score  vs.  child adapter score

The resulting comparison file can be used by the agent to add a node to the
experiment tree, update the best frontier, or roll back a bad config.

The APPS rows include:
  - question: natural language programming problem
  - starter_code: optional starter code
  - input_output: JSON with inputs/outputs and sometimes fn_name
  - solutions: JSON list of reference Python solutions

This evaluator supports three modes:

1. Evaluate existing predictions:
   uv run python scripts/evaluate_apps.py \
     --predictions results/apps_predictions.jsonl --split test --limit 50

2. Generate predictions with a base/adapter model, then evaluate.
   The --model value can be either a Hugging Face id or a local VM path:
   uv run python scripts/evaluate_apps.py \
     --model /opt/Hazelnut/models/qwen3.5-2b-Base \
     --adapter results/runs/exp_000/adapter \
     --split test --limit 20

3. Compare base model vs base+LoRA adapter on the same problem subset:
   uv run python scripts/evaluate_apps.py \
     --model /opt/Hazelnut/models/qwen3.5-2b-Base \
     --adapter results/runs/exp_000/adapter \
     --compare-adapter \
     --split test --limit 20

4. Compare a parent LoRA adapter vs a child LoRA adapter:
   uv run python scripts/evaluate_apps.py \
     --model /opt/Hazelnut/models/qwen3.5-2b-Base \
     --baseline-adapter results/runs/v1/adapter \
     --adapter results/runs/v1_1/adapter \
     --compare-adapter \
     --split test --limit 20

5. Smoke-test the harness with reference solutions:
   uv run python scripts/evaluate_apps.py --use-reference-solution --split test --limit 20

WARNING: this runs model-generated Python in a subprocess with timeouts. That is useful
for a hackathon VM, but it is not a complete security sandbox. For untrusted public
service usage, run candidates inside Docker/firejail/gVisor with network disabled.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class AppsProblem:
    problem_id: int
    question: str
    starter_code: str
    difficulty: str
    input_output: dict[str, Any]
    solutions: list[str]


@dataclass
class TestResult:
    status: str
    passed: bool
    stdout: str = ""
    stderr: str = ""
    expected: Any = None
    actual: Any = None
    seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated code on APPS")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--difficulty",
        choices=["introductory", "interview", "competition"],
        help="Optional APPS difficulty filter.",
    )
    parser.add_argument("--max-tests-per-problem", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=3.0, help="Timeout per test case in seconds.")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "apps_eval.jsonl")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="JSONL with problem_id and code/answer.")
    source.add_argument(
        "--model",
        help=(
            "Base causal LM to generate answers. Accepts either a Hugging Face model id "
            "or a local VM path such as /opt/Hazelnut/models/qwen3.5-2b-Base."
        ),
    )
    source.add_argument("--use-reference-solution", action="store_true")

    parser.add_argument("--adapter", type=Path, help="Optional PEFT LoRA adapter path for --model.")
    parser.add_argument(
        "--baseline-adapter",
        type=Path,
        help=(
            "Optional parent PEFT LoRA adapter for --compare-adapter. If omitted, "
            "the baseline is the raw base model. If supplied, comparison is "
            "base+baseline-adapter vs base+adapter."
        ),
    )
    parser.add_argument(
        "--compare-adapter",
        action="store_true",
        help=(
            "With --model and --adapter, evaluate baseline first, then candidate, "
            "and write a delta summary."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def load_apps_dataset(split: str, offset: int, limit: int, difficulty: str | None) -> list[AppsProblem]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run `uv sync` to install datasets.") from exc

    try:
        dataset = load_dataset("codeparrot/apps", split=split, trust_remote_code=True)
    except TypeError:
        dataset = load_dataset("codeparrot/apps", split=split)

    problems: list[AppsProblem] = []
    for row in dataset:
        if difficulty and str(row.get("difficulty", "")).lower() != difficulty:
            continue
        input_output = parse_json_field(row.get("input_output", "{}"), default={})
        inputs = input_output.get("inputs") or []
        outputs = input_output.get("outputs") or []
        if not inputs or not outputs:
            continue
        solutions = parse_json_field(row.get("solutions", "[]"), default=[])
        problems.append(
            AppsProblem(
                problem_id=int(row["problem_id"]),
                question=row["question"],
                starter_code=row.get("starter_code") or "",
                difficulty=str(row.get("difficulty", "")),
                input_output=input_output,
                solutions=solutions if isinstance(solutions, list) else [],
            )
        )
        if len(problems) >= offset + limit:
            break
    return problems[offset : offset + limit]


def parse_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def load_predictions(path: Path) -> dict[int, str]:
    predictions: dict[int, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "problem_id" not in row:
            raise ValueError(f"Missing problem_id in {path}:{line_number}")
        code = row.get("code") or row.get("answer") or row.get("completion")
        if not isinstance(code, str):
            raise ValueError(f"Missing code/answer/completion in {path}:{line_number}")
        predictions[int(row["problem_id"])] = code
    return predictions


def resolve_model_ref(model_name: str) -> str:
    """Return a local model path when it exists, otherwise keep the HF model id."""
    path = Path(model_name).expanduser()
    if path.exists():
        return str(path)

    looks_like_local_path = (
        model_name.startswith("/")
        or model_name.startswith("./")
        or model_name.startswith("../")
        or model_name.startswith("models/")
    )
    if looks_like_local_path:
        raise SystemExit(
            f"Local model path does not exist: {path}\n"
            "On the VM, pass the absolute path, for example:\n"
            "  --model /opt/Hazelnut/models/qwen3.5-2b-Base"
        )
    return model_name


def load_generator(model_name: str, adapter: Path | None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run `uv sync` to install transformers.") from exc

    model_ref = resolve_model_ref(model_name)
    print(f"loading_model: {model_ref}")
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    if adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("Missing dependency. Run `uv sync` to install peft.") from exc
        model = PeftModel.from_pretrained(model, adapter).to(device)

    model.eval()
    return model, tokenizer, device


def unload_generator(generator: tuple[Any, Any, str]) -> None:
    model, tokenizer, _device = generator
    del model
    del tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def build_prompt(problem: AppsProblem) -> str:
    parts = [
        "Solve this programming problem in Python.",
        "Return only one complete Python program or function. Do not include Markdown.",
        "",
        problem.question.strip(),
    ]
    if problem.starter_code.strip():
        parts.extend(["", "Starter code:", problem.starter_code.strip()])
    return "\n".join(parts)


def generate_code(model: Any, tokenizer: Any, device: str, problem: AppsProblem, args: argparse.Namespace) -> str:
    import torch

    prompt = build_prompt(problem)
    messages = [
        {"role": "system", "content": "You are a precise competitive-programming Python solver."},
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant:\n"

    encoded = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        output = model.generate(**encoded, **generation_kwargs)
    completion_ids = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def normalize_output(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def parse_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return text


def values_equal(actual: Any, expected: Any) -> bool:
    parsed_expected = parse_value(expected)
    if actual == parsed_expected:
        return True
    return normalize_output(actual) == normalize_output(expected)


def syntax_check(code: str) -> TestResult | None:
    try:
        compile(code, "candidate.py", "exec")
    except SyntaxError as exc:
        return TestResult(status="compile_error", passed=False, stderr=str(exc))
    return None


def run_stdio_test(code: str, test_input: Any, expected_output: Any, timeout: float) -> TestResult:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        candidate = tmp_path / "candidate.py"
        candidate.write_text(code + "\n", encoding="utf-8")
        start = time.time()
        try:
            completed = subprocess.run(
                [sys.executable, str(candidate)],
                input=str(test_input),
                text=True,
                cwd=tmp_path,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return TestResult(
                status="timeout",
                passed=False,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                expected=expected_output,
                seconds=timeout,
            )
        seconds = time.time() - start

    if completed.returncode != 0:
        return TestResult(
            status="runtime_error",
            passed=False,
            stdout=completed.stdout,
            stderr=completed.stderr,
            expected=expected_output,
            actual=completed.stdout,
            seconds=seconds,
        )
    passed = normalize_output(completed.stdout) == normalize_output(expected_output)
    return TestResult(
        status="pass" if passed else "wrong_answer",
        passed=passed,
        stdout=completed.stdout,
        stderr=completed.stderr,
        expected=expected_output,
        actual=completed.stdout,
        seconds=seconds,
    )


def run_call_test(code: str, fn_name: str, test_input: Any, expected_output: Any, timeout: float) -> TestResult:
    runner = textwrap.dedent(
        """
        import ast
        import importlib.util
        import json
        import sys

        def parse_value(value):
            if not isinstance(value, str):
                return value
            text = value.strip()
            for parser in (json.loads, ast.literal_eval):
                try:
                    return parser(text)
                except Exception:
                    pass
            return text

        payload = json.loads(sys.stdin.read())
        spec = importlib.util.spec_from_file_location("candidate", "candidate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, payload["fn_name"])
        args = parse_value(payload["input"])
        if not isinstance(args, (list, tuple)):
            args = [args]
        result = fn(*args)
        print(json.dumps(result, ensure_ascii=False, default=repr))
        """
    ).strip()

    payload = json.dumps({"fn_name": fn_name, "input": test_input})
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "candidate.py").write_text(code + "\n", encoding="utf-8")
        (tmp_path / "runner.py").write_text(runner + "\n", encoding="utf-8")
        start = time.time()
        try:
            completed = subprocess.run(
                [sys.executable, "runner.py"],
                input=payload,
                text=True,
                cwd=tmp_path,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return TestResult(
                status="timeout",
                passed=False,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                expected=expected_output,
                seconds=timeout,
            )
        seconds = time.time() - start

    if completed.returncode != 0:
        return TestResult(
            status="runtime_error",
            passed=False,
            stdout=completed.stdout,
            stderr=completed.stderr,
            expected=expected_output,
            seconds=seconds,
        )
    actual = parse_value(completed.stdout)
    passed = values_equal(actual, expected_output)
    return TestResult(
        status="pass" if passed else "wrong_answer",
        passed=passed,
        stdout=completed.stdout,
        stderr=completed.stderr,
        expected=expected_output,
        actual=actual,
        seconds=seconds,
    )


def evaluate_code(problem: AppsProblem, code: str, max_tests: int, timeout: float) -> dict[str, Any]:
    syntax_error = syntax_check(code)
    if syntax_error:
        return {
            "problem_id": problem.problem_id,
            "passed": False,
            "passed_tests": 0,
            "total_tests": 0,
            "pass_rate": 0.0,
            "status_counts": {"compile_error": 1},
            "first_error": syntax_error.stderr,
        }

    inputs = problem.input_output.get("inputs") or []
    outputs = problem.input_output.get("outputs") or []
    fn_name = problem.input_output.get("fn_name")
    total = min(len(inputs), len(outputs), max_tests)
    results: list[TestResult] = []

    for test_input, expected_output in zip(inputs[:total], outputs[:total]):
        if fn_name:
            result = run_call_test(code, fn_name, test_input, expected_output, timeout)
        else:
            result = run_stdio_test(code, test_input, expected_output, timeout)
        results.append(result)

    passed_tests = sum(1 for result in results if result.passed)
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    first_failure = next((result for result in results if not result.passed), None)
    return {
        "problem_id": problem.problem_id,
        "passed": total > 0 and passed_tests == total,
        "passed_tests": passed_tests,
        "total_tests": total,
        "pass_rate": passed_tests / total if total else 0.0,
        "status_counts": status_counts,
        "first_error": first_failure.stderr[:2000] if first_failure else "",
        "first_expected": first_failure.expected if first_failure else None,
        "first_actual": first_failure.actual if first_failure else None,
    }


def reference_code(problem: AppsProblem) -> str:
    if not problem.solutions:
        raise ValueError(f"Problem {problem.problem_id} has no reference solution")
    return str(problem.solutions[0])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"num_problems": 0}
    return {
        "num_problems": total,
        "strict_accuracy": sum(1 for row in rows if row["passed"]) / total,
        "mean_test_pass_rate": sum(float(row["pass_rate"]) for row in rows) / total,
        "compile_error_rate": sum(1 for row in rows if "compile_error" in row["status_counts"]) / total,
        "timeout_problem_rate": sum(1 for row in rows if "timeout" in row["status_counts"]) / total,
        "runtime_error_problem_rate": sum(
            1 for row in rows if "runtime_error" in row["status_counts"]
        )
        / total,
    }


def write_eval_outputs(
    rows: list[dict[str, Any]],
    out_path: Path,
    label: str,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"--- {label}")
    print(json.dumps(summary, indent=2))
    print(f"details: {out_path}")
    print(f"summary: {summary_path}")
    return summary


def evaluate_problem_set(
    problems: list[AppsProblem],
    args: argparse.Namespace,
    predictions: dict[int, str] | None = None,
    generator: tuple[Any, Any, str] | None = None,
    use_reference_solution: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    predictions = predictions or {}
    for index, problem in enumerate(problems, start=1):
        if use_reference_solution:
            raw_code = reference_code(problem)
        elif generator:
            model, tokenizer, device = generator
            raw_code = generate_code(model, tokenizer, device, problem, args)
        else:
            raw_code = predictions.get(problem.problem_id, "")

        code = extract_code(raw_code)
        result = evaluate_code(problem, code, args.max_tests_per_problem, args.timeout)
        row = {
            **result,
            "difficulty": problem.difficulty,
            "has_fn_name": bool(problem.input_output.get("fn_name")),
            "code_chars": len(code),
        }
        rows.append(row)
        print(
            f"[{index}/{len(problems)}] problem_id={problem.problem_id} "
            f"pass_rate={row['pass_rate']:.3f} passed={row['passed']} "
            f"statuses={row['status_counts']}"
        )
    return rows


def compare_summaries(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    higher_is_better = ["strict_accuracy", "mean_test_pass_rate"]
    lower_is_better = ["compile_error_rate", "timeout_problem_rate", "runtime_error_problem_rate"]
    deltas: dict[str, float] = {}
    for key in higher_is_better + lower_is_better:
        deltas[key] = float(candidate_summary.get(key, 0.0)) - float(baseline_summary.get(key, 0.0))

    primary_delta = deltas["mean_test_pass_rate"]
    strict_delta = deltas["strict_accuracy"]
    error_delta = (
        deltas["compile_error_rate"]
        + deltas["timeout_problem_rate"]
        + deltas["runtime_error_problem_rate"]
    )
    verdict = "same"
    if primary_delta > 0.005 or strict_delta > 0.005:
        verdict = "improved"
    if primary_delta < -0.005 or strict_delta < -0.005 or error_delta > 0.02:
        verdict = "regressed"

    return {
        "verdict": verdict,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": deltas,
        "interpretation": (
            f"{candidate_label} is better than {baseline_label} on this APPS subset"
            if verdict == "improved"
            else f"{candidate_label} is worse than {baseline_label} on this APPS subset"
            if verdict == "regressed"
            else f"{candidate_label} is roughly tied with {baseline_label} on this APPS subset"
        ),
    }


def run_compare_adapter(args: argparse.Namespace, problems: list[AppsProblem]) -> None:
    if not args.model or not args.adapter:
        raise SystemExit("--compare-adapter requires both --model and --adapter")
    if args.baseline_adapter and args.baseline_adapter == args.adapter:
        raise SystemExit("--baseline-adapter and --adapter must point to different adapter directories")

    baseline_out = args.out.with_suffix(".baseline.jsonl")
    candidate_out = args.out.with_suffix(".candidate.jsonl")
    compare_out = args.out.with_suffix(".compare.json")
    baseline_label = "base_v0" if args.baseline_adapter is None else f"parent:{args.baseline_adapter}"
    candidate_label = f"candidate:{args.adapter}"

    print(f"=== Evaluating baseline ({baseline_label}) ===")
    baseline_generator = load_generator(args.model, adapter=args.baseline_adapter)
    baseline_rows = evaluate_problem_set(problems, args, generator=baseline_generator)
    unload_generator(baseline_generator)
    baseline_summary = write_eval_outputs(baseline_rows, baseline_out, "baseline")

    print(f"=== Evaluating candidate ({candidate_label}) ===")
    candidate_generator = load_generator(args.model, adapter=args.adapter)
    candidate_rows = evaluate_problem_set(problems, args, generator=candidate_generator)
    unload_generator(candidate_generator)
    candidate_summary = write_eval_outputs(candidate_rows, candidate_out, "candidate")

    comparison = compare_summaries(
        baseline_summary,
        candidate_summary,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    compare_out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("--- comparison")
    print(json.dumps(comparison, indent=2))
    print(f"comparison: {compare_out}")


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    problems = load_apps_dataset(args.split, args.offset, args.limit, args.difficulty)
    if not problems:
        raise SystemExit("No APPS problems found for the requested filters.")

    if args.compare_adapter:
        run_compare_adapter(args, problems)
        return

    predictions = load_predictions(args.predictions) if args.predictions else {}
    generator = load_generator(args.model, args.adapter) if args.model else None
    rows = evaluate_problem_set(
        problems,
        args,
        predictions=predictions,
        generator=generator,
        use_reference_solution=args.use_reference_solution,
    )
    if generator:
        unload_generator(generator)

    write_eval_outputs(rows, args.out, "evaluation")


if __name__ == "__main__":
    main()
