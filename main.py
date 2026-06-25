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


if __name__ == "__main__":
    reset_state()
    install_skill()
    dashboard = start_dashboard()
    try:
        subprocess.run(
            [
                "copilot",
                "-C", REPO,          # run inside the repo so the repur skill auto-loads
                "--allow-all",       # approve-all: no confirmation prompts (tools/shell/paths/urls)
                "--autopilot",       # autonomous execution mode
                "-i", PROMPT,        # interactive: auto-execute the prompt, then keep the loop running
            ],
            cwd=REPO,
        )
    finally:
        # Tear down the dashboard when the CLI session ends.
        dashboard.terminate()
        try:
            dashboard.wait(timeout=10)
        except subprocess.TimeoutExpired:
            dashboard.kill()
