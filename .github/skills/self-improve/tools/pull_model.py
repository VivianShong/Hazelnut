"""Pull the chosen small Qwen student model from the Hugging Face Hub.

After reading the hardware (read_hardware.py), the Research agent picks a Qwen
size that fits VRAM and calls this script to download the weights locally so the
student model can be fine-tuned / served.

Examples:
  uv run python .github/skills/self-improve/tools/pull_model.py \
      --model Qwen/Qwen2.5-0.5B-Instruct

  uv run python .github/skills/self-improve/tools/pull_model.py \
      --model Qwen/Qwen2.5-1.5B-Instruct --out models/qwen-1.5b

By default weights land in ~/.cache/autoresearch/models/<model>. Pass --out to override.
Only `*.safetensors` and config/tokenizer files are fetched (no `.bin` duplicates).
"""
import argparse
import sys
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".cache" / "autoresearch" / "models"

# Skip PyTorch .bin shards when safetensors are present, and other large extras.
IGNORE_PATTERNS = ["*.bin", "*.pth", "*.h5", "*.msgpack", "*.gguf", "*.onnx"]


def main(argv):
    ap = argparse.ArgumentParser(description="Pull a small Qwen model from HF.")
    ap.add_argument("--model", required=True,
                    help="HF model repo id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--revision", default=None, help="branch, tag, or commit")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--include-bin", action="store_true",
                    help="also download .bin shards (default: safetensors only)")
    args = ap.parse_args(argv)

    if "qwen" not in args.model.lower():
        print(f"warning: '{args.model}' does not look like a Qwen model", file=sys.stderr)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "error: huggingface_hub is not installed.\n"
            "Install it into the environment (e.g. add it to the project deps) and retry.",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out) if args.out else DEFAULT_ROOT / args.model.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            local_dir=str(out),
            ignore_patterns=None if args.include_bin else IGNORE_PATTERNS,
        )
    except Exception as exc:  # network / auth / not-found
        print(f"error: model pull failed: {exc}", file=sys.stderr)
        return 2

    print(f"pulled model '{args.model}' to:\n  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
