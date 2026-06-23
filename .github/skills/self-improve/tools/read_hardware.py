"""Report hardware constraints so the model can be sized to the GPU.

Run on setup before the first experiment. Uses torch for accurate total VRAM
and falls back to `nvidia-smi` for live utilization. No extra dependencies.

Usage:
  uv run python .github/skills/self-improve/tools/read_hardware.py [--json]
"""
import json
import os
import shutil
import subprocess
import sys


def gpu_info():
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": p.name,
                    "total_vram_mb": round(p.total_memory / (1024 ** 2), 1),
                    "capability": f"{p.major}.{p.minor}",
                })
    except Exception:
        pass
    return gpus


def nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        return None


def main(argv):
    data = {
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb(),
        "gpus": gpu_info(),
        "nvidia_smi": nvidia_smi(),
    }
    if "--json" in argv:
        print(json.dumps(data, indent=2))
        return 0

    print(f"cpu_count: {data['cpu_count']}")
    print(f"ram_gb: {data['ram_gb']}")
    for g in data["gpus"]:
        print(f"gpu{g['index']}: {g['name']} | {g['total_vram_mb']} MB | sm_{g['capability']}")
    if data["nvidia_smi"]:
        print("nvidia-smi:")
        print(data["nvidia_smi"])
    if not data["gpus"] and not data["nvidia_smi"]:
        print("no NVIDIA GPU detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
