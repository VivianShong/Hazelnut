"""Export a trained leaf-node student model with its provenance metadata.

In the autoresearch version tree every node trains from the *same fixed base
model* (nodes branch on the config, not on a continued checkpoint). Only the
trained models at the leaf / frontier nodes are worth keeping, so this script is
called once a leaf is final: it copies the trained checkpoint to an exports dir
and writes a `metadata.json` capturing the config, score, base model, and git
SHA so the export is reproducible and traceable.

Examples:
  uv run python .github/skills/self-improve/tools/export_model.py \
      --checkpoint run.log.d/ckpt --name best-coder

  uv run python .github/skills/self-improve/tools/export_model.py \
      --checkpoint out/model --val-bpb 0.83 --base-model Qwen/Qwen2.5-0.5B-Instruct \
      --config '{"lr": 3e-4, "n_layer": 12}'

By default exports land in ~/.cache/autoresearch/exports/<name>. Pass --out to override.
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".cache" / "autoresearch" / "exports"


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def main(argv):
    ap = argparse.ArgumentParser(description="Export a trained leaf-node model.")
    ap.add_argument("--checkpoint", required=True,
                    help="path to the trained checkpoint file or directory")
    ap.add_argument("--name", default=None,
                    help="export name (default: derived from timestamp)")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--base-model", default=None,
                    help="HF id of the fixed base model this leaf trained from")
    ap.add_argument("--config", default=None,
                    help="hyperparameter config as a JSON string")
    ap.add_argument("--val-bpb", type=float, default=None,
                    help="validation bits-per-byte score of this node")
    ap.add_argument("--note", default=None, help="free-form note")
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    name = args.name or f"export-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out = Path(args.out) if args.out else DEFAULT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    dest = out / ckpt.name
    try:
        if ckpt.is_dir():
            shutil.copytree(ckpt, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(ckpt, dest)
    except Exception as exc:
        print(f"error: copy failed: {exc}", file=sys.stderr)
        return 2

    config = None
    if args.config:
        try:
            config = json.loads(args.config)
        except json.JSONDecodeError as exc:
            print(f"error: --config is not valid JSON: {exc}", file=sys.stderr)
            return 1

    metadata = {
        "name": name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(dest),
        "base_model": args.base_model,
        "config": config,
        "val_bpb": args.val_bpb,
        "git_sha": git_sha(),
        "note": args.note,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"exported '{name}' to:\n  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
