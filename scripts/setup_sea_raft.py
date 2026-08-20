"""Initialize and validate the pinned SEA-RAFT dependency used by FlowPi."""

from __future__ import annotations

from pathlib import Path
import subprocess

_ROOT = Path(__file__).resolve().parents[1]
_SEA_RAFT_DIR = _ROOT / "SEA-RAFT"
_PATCH = _ROOT / "third_party" / "sea_raft" / "flowpi_return_low_res.patch"
_PINNED_COMMIT = "9137517ba24e628442aec097d3afe71d03503b75"


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(_SEA_RAFT_DIR), *args],
        check=check,
        text=True,
        capture_output=True,
    )


def _check_patch(*, reverse: bool = False) -> bool:
    args = ["apply", "--check", "--unidiff-zero"]
    if reverse:
        args.append("--reverse")
    args.append(str(_PATCH))
    return _git(*args, check=False).returncode == 0


def main() -> None:
    if not _SEA_RAFT_DIR.is_dir() or not (_SEA_RAFT_DIR / "core" / "raft.py").is_file():
        raise SystemExit(
            "SEA-RAFT submodule is missing. Run `git submodule update --init SEA-RAFT`, then rerun "
            "`uv run python scripts/setup_sea_raft.py`."
        )
    if not _PATCH.is_file():
        raise SystemExit(f"FlowPi SEA-RAFT patch is missing: {_PATCH}")

    try:
        commit = _git("rev-parse", "HEAD").stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit("SEA-RAFT is not a usable git submodule. Run `git submodule update --init SEA-RAFT`.") from exc
    if commit != _PINNED_COMMIT:
        raise SystemExit(
            f"SEA-RAFT commit mismatch: expected {_PINNED_COMMIT}, got {commit}. "
            "Use the pinned submodule revision from the FlowPi repository."
        )

    if _check_patch():
        _git("apply", "--unidiff-zero", str(_PATCH))
        print("Applied FlowPi SEA-RAFT return_low_res patch.")
    elif _check_patch(reverse=True):
        print("FlowPi SEA-RAFT return_low_res patch is already applied.")
    else:
        raise SystemExit(
            "SEA-RAFT is at the pinned commit but is neither clean nor patched with the expected "
            "FlowPi low-resolution API patch."
        )

    raft_source = (_SEA_RAFT_DIR / "core" / "raft.py").read_text()
    if "return_low_res=False" not in raft_source or "flow_8x" not in raft_source:
        raise SystemExit("SEA-RAFT setup completed without the required return_low_res/flow_8x API.")
    print(f"SEA-RAFT ready at {commit}.")


if __name__ == "__main__":
    main()
