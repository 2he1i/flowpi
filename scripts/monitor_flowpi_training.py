"""Display live FlowPI training progress from the durable shared logs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import pathlib
import re
import time

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_STEP_RE = re.compile(r"Step\s+(?P<step>\d+):\s*(?P<metrics>.*)")
_KV_RE = re.compile(rf"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>{_NUMBER}|nan|inf|-inf)")
_PROGRESS_RE = re.compile(
    rf"(?P<time>\d{{2}}:\d{{2}}:\d{{2}}\.\d{{3}}).*?"
    rf"(?P<iteration>{_NUMBER})it/.*?"
    rf"rate:(?P<rate>{_NUMBER})(?P<unit>[sm])/it\s+remaining:(?P<remaining>\S+)"
)
_HEARTBEAT_RE = re.compile(r"=== heartbeat (?P<time>[^=]+) ===")
_EXIT_RE = re.compile(r"exit_status=(?P<status>-?\d+)")


@dataclass(frozen=True)
class ProgressSample:
    time_seconds: float
    iteration: float
    reported_seconds_per_step: float | None
    remaining: str


@dataclass(frozen=True)
class GpuSummary:
    timestamp: str
    count: int
    utilization_min: float
    utilization_max: float
    utilization_mean: float
    memory_min_mib: float
    memory_max_mib: float
    memory_total_mib: float
    temperature_max_c: float | None


@dataclass(frozen=True)
class TrainingSnapshot:
    run_dir: pathlib.Path
    status: str
    step: int | None
    total_steps: int | None
    batch_size: int | None
    metrics: dict[str, float]
    speed_steps_per_second: float | None
    speed_seconds_per_step: float | None
    eta_seconds: float | None
    progress_remaining: str | None
    latest_heartbeat: str | None
    heartbeat_age_seconds: float | None
    checkpoint_steps: tuple[int, ...]
    next_checkpoint: int | None
    checkpoint_dir: str | None
    gpu: GpuSummary | None
    critical_messages: tuple[str, ...]
    log_mtime_age_seconds: float | None


def _read_tail(path: pathlib.Path, max_bytes: int = 4_000_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as file:
        file.seek(0, 2)
        size = file.tell()
        file.seek(max(0, size - max_bytes))
        return file.read().decode("utf-8", errors="replace")


def _parse_number(value: str) -> float:
    if value == "nan":
        return math.nan
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return float(value)


def _clock_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _monotonic_clock_samples(samples: list[ProgressSample]) -> list[ProgressSample]:
    """Make time-of-day log timestamps monotonic across midnight."""
    result: list[ProgressSample] = []
    offset = 0.0
    previous = None
    for sample in samples:
        current = sample.time_seconds + offset
        if previous is not None and current < previous:
            offset += 24 * 3600
            current = sample.time_seconds + offset
        result.append(
            ProgressSample(
                time_seconds=current,
                iteration=sample.iteration,
                reported_seconds_per_step=sample.reported_seconds_per_step,
                remaining=sample.remaining,
            )
        )
        previous = current
    return result


def _parse_progress(text: str) -> list[ProgressSample]:
    samples = []
    for line in text.splitlines():
        match = _PROGRESS_RE.search(line)
        if match is None:
            continue
        rate = float(match.group("rate"))
        if match.group("unit") == "m":
            rate *= 60
        samples.append(
            ProgressSample(
                time_seconds=_clock_seconds(match.group("time")),
                iteration=float(match.group("iteration")),
                reported_seconds_per_step=rate,
                remaining=match.group("remaining"),
            )
        )
    return _monotonic_clock_samples(samples)


def _parse_latest_step(text: str) -> tuple[int | None, dict[str, float]]:
    latest_step = None
    latest_metrics: dict[str, float] = {}
    for line in text.splitlines():
        match = _STEP_RE.search(line)
        if match is None:
            continue
        latest_step = int(match.group("step"))
        latest_metrics = {
            item.group("key"): _parse_number(item.group("value")) for item in _KV_RE.finditer(match.group("metrics"))
        }
    return latest_step, latest_metrics


def _parse_argument(text: str, name: str) -> int | None:
    match = re.search(rf"--{re.escape(name)}\s+(\d+)", text)
    return int(match.group(1)) if match else None


def _parse_checkpoint_steps(text: str) -> tuple[int, ...]:
    sections = text.split("checkpoint_steps:")
    if len(sections) < 2:
        return ()
    latest_section = sections[-1].split("GPU snapshot:", maxsplit=1)[0]
    steps = {int(line.strip()) for line in latest_section.splitlines() if line.strip().isdigit()}
    return tuple(sorted(steps))


def _parse_checkpoint_dir(text: str) -> str | None:
    match = re.findall(r"Checkpoint directory:\s*(.+)", text)
    return match[-1].strip() if match else None


def _checkpoint_steps_from_dir(checkpoint_dir: str | None) -> tuple[int, ...]:
    if checkpoint_dir is None:
        return ()
    root = pathlib.Path(checkpoint_dir)
    try:
        return tuple(sorted(int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit()))
    except (FileNotFoundError, OSError, ValueError):
        return ()


def _parse_exit_status(*paths: pathlib.Path) -> str | None:
    for path in paths:
        text = _read_tail(path, max_bytes=32_000)
        if not text:
            continue
        match = _EXIT_RE.search(text)
        if match is None:
            continue
        status = int(match.group("status"))
        return "COMPLETED" if status == 0 else f"FAILED (exit={status})"
    return None


def _parse_heartbeat(text: str) -> tuple[str | None, float | None]:
    matches = list(_HEARTBEAT_RE.finditer(text))
    if not matches:
        return None, None
    latest = matches[-1].group("time").strip()
    heartbeat_time = _clock_seconds(latest.split("T")[-1].rstrip("Z"))
    now = time.time()
    now_time = now % (24 * 3600)
    age = now_time - heartbeat_time
    if age < 0:
        age += 24 * 3600
    return latest, age


def _number_from_cell(cell: str) -> float | None:
    match = re.search(_NUMBER, cell)
    return float(match.group(0)) if match else None


def _parse_gpu_summary(path: pathlib.Path) -> GpuSummary | None:
    text = _read_tail(path, max_bytes=256_000)
    records_by_timestamp: dict[str, list[list[str]]] = {}
    for row in csv.reader(text.splitlines()):
        if len(row) < 9 or not row[0].strip()[:4].isdigit():
            continue
        timestamp = row[0].strip()
        # nvidia-smi can give each GPU a slightly different millisecond timestamp for one
        # sampling pass; group by second so a complete 8-GPU sample is recognized.
        sample_key = timestamp.rsplit(".", maxsplit=1)[0]
        records_by_timestamp.setdefault(sample_key, []).append([cell.strip() for cell in row])
    if not records_by_timestamp:
        return None
    # nvidia-smi appends one GPU row at a time. During a refresh the newest timestamp can be
    # incomplete, so prefer the newest sample with the largest number of GPU rows.
    timestamp, latest = max(records_by_timestamp.items(), key=lambda item: (len(item[1]), item[0]))
    utilization = [_number_from_cell(row[3]) for row in latest]
    memory_used = [_number_from_cell(row[5]) for row in latest]
    memory_total = [_number_from_cell(row[6]) for row in latest]
    temperature = [_number_from_cell(row[8]) for row in latest]
    utilization_values = [value for value in utilization if value is not None]
    memory_values = [value for value in memory_used if value is not None]
    total_values = [value for value in memory_total if value is not None]
    temperature_values = [value for value in temperature if value is not None]
    if not utilization_values or not memory_values or not total_values:
        return None
    return GpuSummary(
        timestamp=timestamp,
        count=len(latest),
        utilization_min=min(utilization_values),
        utilization_max=max(utilization_values),
        utilization_mean=sum(utilization_values) / len(utilization_values),
        memory_min_mib=min(memory_values),
        memory_max_mib=max(memory_values),
        memory_total_mib=max(total_values),
        temperature_max_c=max(temperature_values) if temperature_values else None,
    )


def _critical_messages(*texts: str) -> tuple[str, ...]:
    pattern = re.compile(
        r"Traceback|RESOURCE_EXHAUSTED|out of memory|\bOOM\b|Training exited with status [1-9]|\[ERROR\]", re.IGNORECASE
    )
    messages = []
    for text in texts:
        for line in text.splitlines():
            if pattern.search(line):
                messages.extend([line.strip()])
    return tuple(messages[-4:])


def _latest_run(run_root: pathlib.Path) -> pathlib.Path:
    candidates = [path for path in run_root.iterdir() if path.is_dir() and (path / "train").is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No FlowPI run directories found below {run_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _speed(progress: list[ProgressSample]) -> tuple[float | None, float | None, float | None, str | None]:
    if not progress:
        return None, None, None, None
    latest = progress[-1]
    window = progress[-8:]
    if len(window) < 2:
        return None, latest.reported_seconds_per_step, None, latest.remaining
    elapsed = window[-1].time_seconds - window[0].time_seconds
    iterations = window[-1].iteration - window[0].iteration
    if elapsed <= 0 or iterations <= 0:
        return None, latest.reported_seconds_per_step, None, latest.remaining
    seconds_per_step = elapsed / iterations
    return 1 / seconds_per_step, seconds_per_step, None, latest.remaining


def _snapshot(run_dir: pathlib.Path) -> TrainingSnapshot:
    train_dir = run_dir / "train"
    train_text = _read_tail(train_dir / "run.log")
    heartbeat_text = _read_tail(train_dir / "heartbeat.log")
    launcher_text = _read_tail(run_dir / "train_launcher.log")
    command_text = _read_tail(train_dir / "command.txt", max_bytes=64_000)

    step, metrics = _parse_latest_step(train_text)
    progress = _parse_progress(train_text)
    steps_per_second, seconds_per_step, _, remaining = _speed(progress)
    total_steps = _parse_argument(command_text, "num-train-steps")
    batch_size = _parse_argument(command_text, "batch-size")
    if steps_per_second and step is not None and total_steps is not None:
        eta_seconds = max(0, total_steps - step) / steps_per_second
    else:
        eta_seconds = None

    status = _parse_exit_status(train_dir / "exit_status.txt", run_dir / "exit_status.txt") or "RUNNING"
    heartbeat, heartbeat_age = _parse_heartbeat(heartbeat_text)
    save_interval = _parse_argument(command_text, "save-interval")
    next_checkpoint = ((step // save_interval) + 1) * save_interval if step is not None and save_interval else None
    checkpoint_dir = _parse_checkpoint_dir(launcher_text)
    checkpoint_steps = tuple(
        sorted(set(_parse_checkpoint_steps(heartbeat_text)) | set(_checkpoint_steps_from_dir(checkpoint_dir)))
    )
    gpu = _parse_gpu_summary(train_dir / "gpu_utilization.csv")
    critical = _critical_messages(train_text, launcher_text, _read_tail(run_dir / "run.log"))

    try:
        log_age = time.time() - (train_dir / "run.log").stat().st_mtime
    except FileNotFoundError:
        log_age = None

    return TrainingSnapshot(
        run_dir=run_dir,
        status=status,
        step=step,
        total_steps=total_steps,
        batch_size=batch_size,
        metrics=metrics,
        speed_steps_per_second=steps_per_second,
        speed_seconds_per_step=seconds_per_step,
        eta_seconds=eta_seconds,
        progress_remaining=remaining,
        latest_heartbeat=heartbeat,
        heartbeat_age_seconds=heartbeat_age,
        checkpoint_steps=checkpoint_steps,
        next_checkpoint=next_checkpoint,
        checkpoint_dir=checkpoint_dir,
        gpu=gpu,
        critical_messages=critical,
        log_mtime_age_seconds=log_age,
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "-"
    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_number(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _render(snapshot: TrainingSnapshot) -> str:
    lines = [
        "FlowPI training monitor (read-only)",
        f"Run:    {snapshot.run_dir}",
        f"Status: {snapshot.status}    log_age={_format_duration(snapshot.log_mtime_age_seconds)}    heartbeat_age={_format_duration(snapshot.heartbeat_age_seconds)}",
        "",
    ]
    if snapshot.step is None:
        lines.append("Progress: waiting for the first training step")
    else:
        total = str(snapshot.total_steps) if snapshot.total_steps is not None else "?"
        lines.append(
            f"Progress: {snapshot.step}/{total}    speed={_format_number(snapshot.speed_steps_per_second, 3)} step/s ({_format_number(snapshot.speed_seconds_per_step, 2)} s/step)"
        )
        lines.append(
            f"          throughput={_format_number(snapshot.speed_steps_per_second * snapshot.batch_size if snapshot.speed_steps_per_second and snapshot.batch_size else None, 1)} samples/s    ETA={_format_duration(snapshot.eta_seconds)}"
        )
    lines.extend(
        [
            f"Loss:     {_format_number(snapshot.metrics.get('loss'))}    grad_norm={_format_number(snapshot.metrics.get('grad_norm'))}    param_norm={_format_number(snapshot.metrics.get('param_norm'), 2)}",
            f"Flow:     mean_delay={_format_number(snapshot.metrics.get('mean_flow_delay'), 3)}    frac_delay_0={_format_number(snapshot.metrics.get('frac_flow_delay_0'), 3)}    age_emb_norm={_format_number(snapshot.metrics.get('flow_age_emb_norm'), 4)}",
            f"VLM:      mean_delay={_format_number(snapshot.metrics.get('mean_vlm_delay'), 3)}    frac_delay_max={_format_number(snapshot.metrics.get('frac_vlm_delay_max'), 3)}",
            f"Flow CA:  gate={_format_number(snapshot.metrics.get('flow_gate_tanh_abs_layer7'), 4)}/{_format_number(snapshot.metrics.get('flow_gate_tanh_abs_layer12'), 4)}/{_format_number(snapshot.metrics.get('flow_gate_tanh_abs_layer16'), 4)}    residual={_format_number(snapshot.metrics.get('flow_ca_residual_ratio_layer7'), 6)}/{_format_number(snapshot.metrics.get('flow_ca_residual_ratio_layer12'), 6)}/{_format_number(snapshot.metrics.get('flow_ca_residual_ratio_layer16'), 6)}",
            f"πR²:      frac_pir2={_format_number(snapshot.metrics.get('frac_pir2'), 3)}",
        ]
    )
    if snapshot.gpu is not None:
        gpu = snapshot.gpu
        memory_gib = gpu.memory_max_mib / 1024
        total_gib = gpu.memory_total_mib / 1024
        lines.append(
            f"GPU:      {gpu.count} cards    util={gpu.utilization_min:.0f}-{gpu.utilization_max:.0f}% (avg {gpu.utilization_mean:.0f}%)    "
            f"memory={memory_gib:.2f}/{total_gib:.2f} GiB    max_temp={gpu.temperature_max_c:.0f}°C"
            if gpu.temperature_max_c is not None
            else f"GPU:      {gpu.count} cards    util={gpu.utilization_min:.0f}-{gpu.utilization_max:.0f}% (avg {gpu.utilization_mean:.0f}%)    memory={memory_gib:.2f}/{total_gib:.2f} GiB"
        )
        lines.append(f"          sample={gpu.timestamp}")
    checkpoints = (
        ", ".join(str(step) for step in snapshot.checkpoint_steps) if snapshot.checkpoint_steps else "none yet"
    )
    lines.append(f"Ckpt:     completed={checkpoints}    next={snapshot.next_checkpoint or '-'}")
    if snapshot.checkpoint_dir:
        lines.append(f"          dir={snapshot.checkpoint_dir}")
    if snapshot.latest_heartbeat:
        lines.append(f"Heartbeat: {snapshot.latest_heartbeat}")
    if snapshot.critical_messages:
        lines.append("")
        lines.append("Recent critical messages:")
        lines.extend(f"  {message}" for message in snapshot.critical_messages)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=pathlib.Path,
        default=pathlib.Path("logs/flowpi_cache_train/flowpi_8xh200"),
        help="Root containing timestamped FlowPI runs; the newest run is followed.",
    )
    parser.add_argument(
        "--run-dir", type=pathlib.Path, help="Follow this exact run instead of auto-selecting the newest run."
    )
    parser.add_argument("--refresh", type=float, default=30.0, help="Refresh interval in seconds (default: 30).")
    parser.add_argument("--once", action="store_true", help="Render once and exit; useful for scripted checks.")
    args = parser.parse_args()
    if args.refresh <= 0:
        parser.error("--refresh must be positive")

    while True:
        try:
            run_dir = args.run_dir if args.run_dir is not None else _latest_run(args.run_root)
            snapshot = _snapshot(run_dir)
            print("\033[2J\033[H", end="")
            print(_render(snapshot), flush=True)
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"FlowPI monitor waiting: {error}", flush=True)
        if args.once:
            return
        try:
            time.sleep(args.refresh)
        except KeyboardInterrupt:
            print("\nFlowPI monitor stopped.")
            return


if __name__ == "__main__":
    main()
