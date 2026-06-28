# main.py
# Launch the GitHub Copilot CLI to drive the /repur GRPO loop unattended.
#
# Before launching, the skill is synced into the Copilot CLI's global skills
# folder (~/.copilot/skills/repur) so /repur is also available outside this repo.
# The live dashboard (dashboard.py) is started as a background process.
#
# Every run resets ledger.json and tree.json back to a known starting state
# (see reset_state): the current files are first snapshotted once into seeds/
# (the "starting state"), then archived into backups/<timestamp>/ before being
# restored from the seed (or an empty structure if no seed exists). tree.json is
# left read-only afterwards so the agent can ONLY write ledger.json — the ledger
# is the single system of record the research loop is allowed to mutate.

import contextlib
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = str(Path(__file__).resolve().parent)
SKILL_NAME = "repur"
SKILL_SOURCE = Path(REPO) / ".github" / "skills" / SKILL_NAME
CLI_SKILL_DIR = Path.home() / ".copilot" / "skills" / SKILL_NAME

# Output layout (see README "Repo structure"): the ledger is global; the tree is
# per-run under outputs/runs/<run>/.
RUN = "run-001"
LEDGER_PATH = Path(REPO) / "outputs" / "ledger.json"
TREE_PATH = Path(REPO) / "outputs" / "runs" / RUN / "tree.json"
BACKUP_DIR = Path(REPO) / "outputs" / "backups"

# Fallback starting states used only when no seed has been captured yet.
EMPTY_LEDGER = {
    "meta": {"schema_version": 2, "headline": "inquiry-DAG over a checkpoint substrate"},
    "checkpoints": {},
    "experiments": {},
}
EMPTY_TREE = {
    "meta": {"schema_version": 1, "task": "grpo-qwen3.5-2b-code", "root_id": None},
    "nodes": {},
}

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8765

PROMPT = (
    "/repur your goal is to train chain of thought on this Qwen-3.5-2B-base, "
    "check the repository setting and use the current available experiment frameworks"
)


def _write_fresh(path: Path, text: str, *, read_only: bool) -> None:
    """Replace path with fresh content we own, then set its write policy.

    Unlinks any existing file first so a file that is read-only or owned by
    another user (e.g. written by a prior `sudo` session) can still be replaced
    — we own the parent directory even when we don't own the file. chmod is then
    safe because the new file is owned by us.
    """
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        # Couldn't remove it; fall back to a best-effort in-place chmod.
        with contextlib.suppress(PermissionError):
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    path.write_text(text)
    mode = path.stat().st_mode
    if read_only:
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    else:
        path.chmod(mode | stat.S_IWUSR)


def reset_state() -> None:
    # Store the current ledger.json and tree.json into a timestamped backup folder.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = BACKUP_DIR / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in (LEDGER_PATH, TREE_PATH):
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
    # Ensure the outputs/ layout exists before writing the fresh blank state.
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TREE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Reset Ledger and Tree to blank
    _write_fresh(LEDGER_PATH, json.dumps(EMPTY_LEDGER, indent=2), read_only=False)
    _write_fresh(TREE_PATH, json.dumps(EMPTY_TREE, indent=2), read_only=True)

    print("Reset ledger.json (writable) and tree.json (read-only)")


def install_skill() -> None:
    """Copy (or refresh) the repo skill into the Copilot CLI skills folder.

    Mirrors the whole skill directory so SKILL.md and the bundled tools/ scripts
    ship together. Re-running updates an existing install in place.
    """
    if not (SKILL_SOURCE / "SKILL.md").is_file():
        raise FileNotFoundError(f"Skill not found at: {SKILL_SOURCE / 'SKILL.md'}")

    CLI_SKILL_DIR.parent.mkdir(parents=True, exist_ok=True)
    # Replace any prior install so removed files don't linger.
    if CLI_SKILL_DIR.exists():
        shutil.rmtree(CLI_SKILL_DIR)
    shutil.copytree(SKILL_SOURCE, CLI_SKILL_DIR)
    print(f"Installed /{SKILL_NAME} (CLI skill) -> {CLI_SKILL_DIR / 'SKILL.md'}")


def start_dashboard() -> subprocess.Popen:
    """Launch the live dashboard server in the background.

    Returns the Popen handle so the caller can terminate it on exit.
    """
    proc = subprocess.Popen(
        [
            "uv", "run", "python", "dashboard.py",
            "--host", DASHBOARD_HOST,
            "--port", str(DASHBOARD_PORT),
        ],
        cwd=REPO,
    )
    print(f"Live dashboard -> http://{DASHBOARD_HOST}:{DASHBOARD_PORT}/ (pid {proc.pid})")
    return proc


def launch(*, mode: str, prompt: str = PROMPT) -> None:
    """Launch the Copilot research loop.

    mode="run"  — the full autonomous loop: reset only if REPURRR_RESET=1, start
        the live dashboard, run Copilot in --autopilot/--allow-all.

    mode="demo" — cold-open for recording. Injects the same prompt into Copilot so
        the agent visibly spins up, but is engineered to NOT interfere with the
        experiments already running:
          • never resets the ledger/tree,
          • does NOT start a second dashboard (reuse the one already on :8765),
          • runs Copilot interactively WITHOUT --autopilot/--allow-all, so the
            agent cannot autonomously write the ledger or queue checkpoints the
            driver would pick up — every mutating action would need your approval.
        Purely cosmetic: nothing the current run depends on is touched.
    """
    demo = mode == "demo"

    if not demo and os.environ.get("REPURRR_RESET") == "1":
        reset_state()
    else:
        print("reset DISABLED: preserving current ledger.json and tree.json")

    install_skill()  # only touches ~/.copilot/skills; never the ledger/run data
    dashboard = None if demo else start_dashboard()

    cmd = ["copilot", "-C", REPO]            # run in-repo so the repur skill auto-loads
    if not demo:
        cmd += ["--allow-all", "--autopilot"]  # autonomous, approve-all (run mode only)
    cmd += ["-i", prompt]                     # inject the prompt
    print(f"injecting prompt: {prompt!r}")

    if demo:
        print("DEMO MODE: interactive cold-open. The agent will NOT auto-modify the "
              "ledger or driver — approve no tool calls to keep the running "
              "experiments untouched. (Dashboard reused on :8765; no reset.)")

    try:
        subprocess.run(cmd, cwd=REPO)
    finally:
        if dashboard is not None:
            # Tear down the dashboard we started (run mode only).
            dashboard.terminate()
            try:
                dashboard.wait(timeout=10)
            except subprocess.TimeoutExpired:
                dashboard.kill()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Launch the Repurrr Copilot research loop.")
    ap.add_argument("mode", nargs="?", choices=["run", "demo"], default="run",
                    help="run: full autonomous loop (reset only if REPURRR_RESET=1); "
                         "demo: cold-open visual only, does not touch running experiments")
    ap.add_argument("-p", "--prompt", default=PROMPT,
                    help="custom prompt to inject into Copilot (default: the built-in goal prompt)")
    args = ap.parse_args()
    launch(mode=args.mode, prompt=args.prompt)
