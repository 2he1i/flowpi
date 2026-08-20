"""Fit the slow-channel delay distribution for training from runtime freshness telemetry.

The training transform `DelaySlowImage` samples `d_vlm ~ U{0..min(vlm_delay_max, frame_index)}`
by default. The runtime's slow-channel delay (`current_tick - prefix_source_tick`, which includes
the VLM compute latency) is NOT uniform in deployment: the VLM service rate and the refresh
interval concentrate the delay around a few values. Training with a uniform delay over-samples
delays that never occur and under-samples the delays the policy will actually see.

This script fits a histogram of the runtime delay distribution from the telemetry dump of a
realtime replay (`scripts/flowpi_infer.py --realtime --telemetry-json out.json`), and prints:

1. The recommended `vlm_delay_max`: the P99 of the observed delay in ticks (a hard cap well above
   the operating point, so the delay embedding keeps training mass where the runtime actually is).
2. The smoothed histogram `0.8 * P̂ + 0.2 * U` over `[0, vlm_delay_max]` to paste into
   `FlowConfig.vlm_delay_distribution`. The 20% uniform floor keeps a non-zero training mass on
   every reachable delay (the runtime clamps at `vlm_delay_max`, and the fast Action Expert must
   not see an untrained delay embedding when the VLM stalls).

Usage:
    uv run python scripts/flowpi_infer.py --config-name flowpi_aloha --checkpoint ... \\
        --dataset data/adjust_bottle_ep0 --realtime --telemetry-json telemetry.json
    uv run python scripts/fit_vlm_delay.py telemetry.json [--alpha 0.8]
"""

import argparse
import json
from pathlib import Path

import numpy as np

_SMOOTHING_ALPHA = 0.8  # P̂ weight; uniform floor is (1 - alpha)


def _load_delays(path: Path) -> tuple[np.ndarray, int]:
    """Reads raw delay ticks, falling back to legacy clamped telemetry when necessary."""
    with open(path) as f:
        payload = json.load(f)
    vlm_delay_max = int(payload.get("vlm_delay_max", 0))
    telemetry = payload.get("telemetry") or []
    if telemetry and "delay_ticks_raw" in telemetry[0]:
        delays = np.asarray([t["delay_ticks_raw"] for t in telemetry], dtype=np.int64)
    elif telemetry and "delay_ticks" in telemetry[0]:
        delays = np.asarray([t["delay_ticks"] for t in telemetry], dtype=np.int64)
    else:
        stats = payload.get("stats") or {}
        delays = np.asarray(
            stats.get("prefix_age_at_install_raw", stats.get("prefix_age_at_install", [])), dtype=np.int64
        )
    if delays.size == 0:
        raise ValueError(f"No delay samples in {path}. Run a realtime replay with --telemetry-json first.")
    return delays, vlm_delay_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry_json", type=Path, help="Telemetry dump from flowpi_infer.py")
    parser.add_argument("--alpha", type=float, default=_SMOOTHING_ALPHA, help="P̂ weight in [0, 1]")
    parser.add_argument(
        "--delay-max",
        type=int,
        default=None,
        help="Override vlm_delay_max (default: P99 of the observed delays, at least 1).",
    )
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}")

    delays, recorded_max = _load_delays(args.telemetry_json)
    p99 = int(np.percentile(delays, 99))
    delay_max = args.delay_max if args.delay_max is not None else max(p99, 1)
    n = delay_max + 1

    hist = np.bincount(delays.astype(np.int64), minlength=n)[:n].astype(np.float64)
    if hist.sum() == 0:
        raise ValueError(f"All observed delays are beyond vlm_delay_max={delay_max}; raise --delay-max.")
    fitted = hist / hist.sum()
    uniform = np.full(n, 1.0 / n)
    smoothed = args.alpha * fitted + (1.0 - args.alpha) * uniform
    smoothed = smoothed / smoothed.sum()

    print(f"samples: {len(delays)}  recorded vlm_delay_max: {recorded_max}")
    print(f"delay histogram (P̂): {', '.join(f'{i}: {100 * p:.1f}%' for i, p in enumerate(fitted))}")
    print(f"P99(delay) = {p99}  ->  recommended vlm_delay_max = {delay_max}")
    print(f"smoothed distribution ({args.alpha:.2f} * P̂ + {1 - args.alpha:.2f} * U) over [0, {delay_max}]:")
    print("  " + ", ".join(f"{p:.6f}" for p in smoothed))
    print("\nPaste into the model config:")
    print(f"  flow: pi0_config.FlowConfig(vlm_delay_max={delay_max}, vlm_delay_distribution=(")
    print("      " + ", ".join(f"{p:.4f}" for p in smoothed))
    print("  ))")
    print(
        "\nNote: fit from a --realtime replay of a representative workload (same refresh interval, "
        "same VLM hardware); the delay distribution is workload-specific."
    )


if __name__ == "__main__":
    main()
