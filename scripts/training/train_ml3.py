"""Future Colab entrypoint for ML3 Within and LOSO training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train configured ML3 tree ensembles.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    build_arg_parser().parse_args(argv)
    raise NotImplementedError(
        "Training policy is intentionally pending; configure estimators and folds first."
    )


if __name__ == "__main__":
    raise SystemExit(main())
