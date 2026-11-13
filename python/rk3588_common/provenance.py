"""Provenance fields written into every result file (STANDARDS.md).

`repos` holds the commit and dirty flag of every repo whose code
produced the file. The board has no git checkout: the PC passes its repo
states as `--repo NAME:COMMIT:DIRTY` arguments (repo_arg,
add_provenance_args).
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import subprocess
from pathlib import Path

from rk3588_common.model_io import PROTOCOL_VERSION


def repo_state(path: str | Path, pathspec: tuple[str, ...] = ()) -> dict:
    """Return {"commit", "dirty"} of the git repo at `path`.

    dirty is True when tracked or new files under `pathspec` (whole repo
    when empty) differ from the commit. Outside a git checkout: commit
    "unknown", dirty True.

    Cached per process: a long-running process (training for hours)
    keeps executing the code it loaded at start regardless of later
    edits on disk, so every file it writes reports the state from its
    own start.
    """
    commit, dirty = _repo_state(str(Path(path).resolve()), tuple(pathspec))
    return {"commit": commit, "dirty": dirty}


@functools.cache
def _repo_state(path: str, pathspec: tuple[str, ...]) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        status = subprocess.check_output(
            ["git", "-C", path, "status", "--porcelain", "--", *pathspec],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return commit.strip(), bool(status.strip())


def common_state() -> dict:
    """Return repo_state of rk3588-common for its python/ directory."""
    return repo_state(Path(__file__).resolve().parents[2], ("python",))


def file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(repos: dict[str, dict], **extra: object) -> dict:
    """Build provenance dictionary with repos, timestamp, and extras."""
    return {
        "repos": repos,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **extra,
    }


def repo_arg(name: str, state: dict) -> str:
    """Format a repo state as a CLI argument for add_provenance_args."""
    return f"{name}:{state['commit']}:{str(state['dirty']).lower()}"


def _parse_repo_arg(text: str) -> tuple[str, dict]:
    parts = text.split(":")
    if len(parts) != 3 or parts[2] not in ("true", "false"):
        raise argparse.ArgumentTypeError(
            f"expected NAME:COMMIT:true|false, got {text!r}"
        )
    return parts[0], {"commit": parts[1], "dirty": parts[2] == "true"}


def add_provenance_args(parser: argparse.ArgumentParser) -> None:
    """Register --run-id and --repo arguments on an ArgumentParser."""
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--repo",
        dest="repos",
        action="append",
        type=_parse_repo_arg,
        required=True,
        help=(
            "NAME:COMMIT:true|false, once per repo whose code runs "
            "(see repo_arg)"
        ),
    )


def board_provenance(args: argparse.Namespace, model_path: Path) -> dict:
    """Provenance of a tier-2 (board) result file."""
    return provenance(
        dict(args.repos),
        run_id=args.run_id,
        tier=2,
        protocol_version=PROTOCOL_VERSION,
        model_file=str(model_path),
        model_sha256=file_sha256(model_path),
    )
