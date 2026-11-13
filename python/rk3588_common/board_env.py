"""Board telemetry for result files: frequencies, temperatures, system info, resource use, latency summary.

Reads Linux sysfs and procfs. On the PC the RK3588 nodes are missing and the values are None.
"""

from __future__ import annotations

import platform
import threading
import time
from array import array
from pathlib import Path

import numpy as np
import psutil


def summarize(values_ms: list[float]) -> dict:
    a = np.asarray(values_ms)
    return {
        "mean": round(float(a.mean()), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "std": round(float(a.std()), 3),
        "n": int(a.size),
    }


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def thermal_zone(name: str = "npu-thermal") -> Path | None:
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        if read_text(str(zone / "type")) == name:
            return zone
    return None


def temperature_c(zone: Path | None) -> float | None:
    raw = read_text(str(zone / "temp")) if zone else None
    return round(int(raw) / 1000, 1) if raw else None


def frequency_state() -> dict:
    """Governor and current frequency of CPU clusters, NPU and DDR (RKNPU2 User Guide p.72-73)."""
    state = {}
    for policy in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        state[policy.name] = {
            "governor": read_text(str(policy / "scaling_governor")),
            "cur_khz": read_text(str(policy / "scaling_cur_freq")),
        }
    for name, node in (("npu", "fdab0000.npu"), ("ddr", "dmc")):
        base = Path("/sys/class/devfreq") / node
        state[name] = {"governor": read_text(str(base / "governor")), "cur_hz": read_text(str(base / "cur_freq"))}
    state["locked"] = all(v["governor"] in ("performance", "userspace") for v in state.values())
    return state


def system_info() -> dict:
    os_release = dict(line.split("=", 1) for line in (read_text("/etc/os-release") or "").splitlines() if "=" in line)
    return {
        "kernel": platform.release(),
        "os": os_release.get("PRETTY_NAME", "").strip('"'),
        "python": platform.python_version(),
        "ram_total_mib": round(psutil.virtual_memory().total / 2**20),
        "cpu_count": psutil.cpu_count(),
    }


class ResourceMonitor:
    """Samples this process' VmRSS and CPU %, system CPU % and NPU temperature every `interval` s."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.proc = psutil.Process()
        self.zone = thermal_zone()
        # array, not list: the samples of a long run must not be walked by the cyclic collector
        # (the legacy camera harness measured 40 ns per list item)
        self.rss_mib, self.proc_cpu, self.sys_cpu, self.temps = array("d"), array("d"), array("d"), array("d")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)
        while not self._stop.wait(self.interval):
            self.rss_mib.append(self.proc.memory_info().rss / 2**20)  # = VmRSS in /proc/self/status
            self.proc_cpu.append(self.proc.cpu_percent(None))  # 100 = one full core
            self.sys_cpu.append(psutil.cpu_percent(None))  # 100 = all cores
            temp = temperature_c(self.zone)
            if temp is not None:
                self.temps.append(temp)

    def __enter__(self):
        self.temp_start = temperature_c(self.zone)
        self.t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.duration_s = time.perf_counter() - self.t0
        self.temp_end = temperature_c(self.zone)

    def result(self) -> dict:
        return {
            "peak_rss_mib": round(max(self.rss_mib), 1) if self.rss_mib else None,
            "avg_process_cpu_pct": round(float(np.mean(self.proc_cpu)), 1) if self.proc_cpu else None,
            "avg_system_cpu_pct": round(float(np.mean(self.sys_cpu)), 1) if self.sys_cpu else None,
            "thermal_zone": self.zone.name if self.zone else None,
            "temp_start_c": self.temp_start,
            "temp_end_c": self.temp_end,
            "temp_max_c": max(self.temps) if self.temps else None,
            "duration_s": round(self.duration_s, 1),
            "samples": len(self.rss_mib),
        }
