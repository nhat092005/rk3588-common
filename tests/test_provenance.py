"""rk3588_common.provenance: repo state, file hash, board arguments."""

import argparse
import hashlib
import subprocess

import pytest

from rk3588_common import model_io
from rk3588_common import provenance as prov


def git(path, *args):
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "a.py").write_text("x = 1\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    prov._repo_state.cache_clear()
    return tmp_path


def head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def test_repo_state_clean(repo):
    assert prov.repo_state(repo) == {"commit": head(repo), "dirty": False}


def test_repo_state_dirty_only_inside_pathspec(repo):
    (repo / "results.json").write_text("{}")  # generated output outside the code paths
    assert prov.repo_state(repo, ("code",))["dirty"] is False
    assert prov.repo_state(repo)["dirty"] is True
    (repo / "code" / "new.py").write_text("")
    assert prov.repo_state(repo, ("code",))["dirty"] is False  # cached: state from the first call of the process
    prov._repo_state.cache_clear()
    assert prov.repo_state(repo, ("code",))["dirty"] is True


def test_repo_state_outside_git(tmp_path):
    prov._repo_state.cache_clear()
    assert prov.repo_state(tmp_path) == {"commit": "unknown", "dirty": True}


def test_file_sha256(tmp_path):
    data = bytes(range(256)) * 9000  # over two 1 MiB chunks
    (tmp_path / "m.rknn").write_bytes(data)
    assert prov.file_sha256(tmp_path / "m.rknn") == hashlib.sha256(data).hexdigest()


def test_board_provenance_round_trip(tmp_path):
    model = tmp_path / "best.rknn"
    model.write_bytes(b"model")
    states = {"ai": {"commit": "a" * 40, "dirty": False}, "common": {"commit": "c" * 40, "dirty": True}}
    parser = argparse.ArgumentParser()
    prov.add_provenance_args(parser)
    argv = ["--run-id", "r0"] + [x for name, s in states.items() for x in ("--repo", prov.repo_arg(name, s))]
    result = prov.board_provenance(parser.parse_args(argv), model)
    assert result["repos"] == states
    assert result["run_id"] == "r0" and result["tier"] == 2
    assert result["protocol_version"] == model_io.PROTOCOL_VERSION
    assert result["model_file"] == str(model) and result["model_sha256"] == hashlib.sha256(b"model").hexdigest()
    assert list(result)[:2] == ["repos", "created_at"]


@pytest.mark.parametrize("bad", ["ai:abc", "ai:abc:yes", "ai:abc:true:x"])
def test_bad_repo_arg(bad):
    parser = argparse.ArgumentParser()
    prov.add_provenance_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-id", "r0", "--repo", bad])
