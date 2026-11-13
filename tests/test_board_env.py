"""Tests for rk3588_common.board_env against legacy-final fixtures."""

import json
import time

import cases
import pytest

from rk3588_common import board_env as env


@pytest.mark.parametrize("i", range(len(cases.latency_lists())))
def test_summarize(i):
    expected = json.loads(cases.BOARD_ENV_EXPECTED.read_text())["summarize"][i]
    assert env.summarize(cases.latency_lists()[i]) == expected


def test_read_text_and_temperature(tmp_path):
    (tmp_path / "temp").write_text("48125\n")
    assert env.read_text(str(tmp_path / "temp")) == "48125"
    assert env.read_text(str(tmp_path / "missing")) is None
    assert env.temperature_c(tmp_path) == 48.1
    assert env.temperature_c(None) is None
    assert env.thermal_zone("no-such-zone") is None


def test_frequency_state_shape():
    state = env.frequency_state()
    assert {"npu", "ddr", "locked"} <= state.keys()
    assert all(
        key.startswith("policy")
        for key in state.keys() - {"npu", "ddr", "locked"}
    )
    assert set(state["npu"]) == {"governor", "cur_hz"}
    assert isinstance(state["locked"], bool)


def test_system_info_shape():
    info = env.system_info()
    assert set(info) == {
        "kernel",
        "os",
        "python",
        "ram_total_mib",
        "cpu_count",
    }
    assert info["ram_total_mib"] > 0
    assert info["cpu_count"] > 0


def test_resource_monitor():
    with env.ResourceMonitor(interval=0.02) as mon:
        time.sleep(0.2)
    result = mon.result()
    assert result["samples"] > 0
    assert result["peak_rss_mib"] > 0
    assert set(result) == {
        "peak_rss_mib",
        "avg_process_cpu_pct",
        "avg_system_cpu_pct",
        "thermal_zone",
        "temp_start_c",
        "temp_end_c",
        "temp_max_c",
        "duration_s",
        "samples",
    }
