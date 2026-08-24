"""Print a FlowPi metrics JSON file as compact human-readable terminal tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import openpi.policies.flowpi_runtime as flowpi_runtime  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="policy_runtime.json or a runtime metrics JSON file")
    parser.add_argument(
        "--title",
        default="FlowPi Metrics / 推理指标",
        help="title printed above the tables",
    )
    args = parser.parse_args()

    metrics_path = args.metrics.expanduser().resolve()
    if not metrics_path.is_file():
        parser.error(f"metrics file does not exist: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    print(flowpi_runtime.format_metrics_table(metrics, title=args.title))


if __name__ == "__main__":
    main()
