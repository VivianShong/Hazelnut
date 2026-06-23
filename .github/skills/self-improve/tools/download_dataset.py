"""Download a training dataset from the Hugging Face Hub.

The Research agent first *searches* (web) for a dataset suited to the task and
hardware, then calls this script with the chosen repo id to fetch it locally.

Examples:
  # whole dataset repo
  uv run python .github/skills/self-improve/tools/download_dataset.py \
      --repo karpathy/tinystories-gpt4-clean

  # only specific files (faster)
  uv run python .github/skills/self-improve/tools/download_dataset.py \
      --repo karpathy/tinystories-gpt4-clean --files data/train.parquet data/val.parquet

By default files land in ~/.cache/autoresearch/datasets/<repo>. Pass --out to override.
"""
import argparse
import sys
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".cache" / "autoresearch" / "datasets"


def main(argv):
    ap = argparse.ArgumentParser(description="Download a HF dataset.")
    ap.add_argument("--repo", required=True, help="HF dataset repo id, e.g. user/name")
    ap.add_argument("--files", nargs="*", default=None,
                    help="specific files to fetch; omit to download the whole repo")
    ap.add_argument("--revision", default=None, help="branch, tag, or commit")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print(
            "error: huggingface_hub is not installed.\n"
            "Install it into the environment (e.g. add it to the project deps) and retry.",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out) if args.out else DEFAULT_ROOT / args.repo.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)

    try:
        if args.files:
            paths = [
                hf_hub_download(
                    repo_id=args.repo, filename=f, revision=args.revision,
                    repo_type="dataset", local_dir=str(out),
                )
                for f in args.files
            ]
        else:
            paths = [snapshot_download(
                repo_id=args.repo, revision=args.revision,
                repo_type="dataset", local_dir=str(out),
            )]
    except Exception as exc:  # network / auth / not-found
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 2

    print(f"downloaded dataset '{args.repo}' to:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
