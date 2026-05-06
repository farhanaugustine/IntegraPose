from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .autolabel_runtime import load_job_from_path, run_autolabel_job


def _emit(message: str) -> None:
    print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AutoLabel Forge autolabeling in a clean subprocess.")
    parser.add_argument("--job", required=True, help="Path to a JSON job payload.")
    args = parser.parse_args(argv)

    job_path = Path(args.job)
    try:
        job = load_job_from_path(job_path)
        dataset_dir = run_autolabel_job(job, _emit)
        _emit(f"Autolabeling complete. Dataset folder: {dataset_dir}")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
