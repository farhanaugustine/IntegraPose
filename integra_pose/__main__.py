"""Executable module for ``python -m integra_pose``."""
from __future__ import annotations

import argparse
from typing import Sequence

from . import bootstrap


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="integrapose",
        description="Launch the IntegraPose GUI.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mirror application logs to the terminal for troubleshooting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Launch the IntegraPose GUI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    bootstrap.launch_gui(debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
