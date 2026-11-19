"""MCC 网络环境模拟器。

传输模型（与 issue #4 一致）::

    T_net = RTT / 2 + Payload * 8 / Bandwidth

支持两种用法：
1. ``estimate_ms(payload)`` —— 纯数学估算，用于快速扫参数；
2. ``inject(payload)`` —— 真实 ``time.sleep`` 注入延迟，用于端到端实测。
"""

from __future__ import annotations

import random
import time
from typing import Dict, List

from .config import NETWORK_PRESETS, DEFAULT_NETWORK, NetworkProfile


class NetworkSimulator:
    """可切换的网络环境模拟器。"""

    def __init__(self, profile=DEFAULT_NETWORK, enable_sleep: bool = True, seed: int = 42):
        self.profile = profile if isinstance(profile, NetworkProfile) else NETWORK_PRESETS[profile]
        self.enable_sleep = enable_sleep
        self._rng = random.Random(seed)

    def switch(self, profile) -> NetworkProfile:
        """运行时切换网络（Part C 动态划分会用到）。"""
        self.profile = profile if isinstance(profile, NetworkProfile) else NETWORK_PRESETS[profile]
        return self.profile

    def estimate_ms(self, payload_bytes: int) -> float:
        return self.profile.transmission_ms(payload_bytes)

    def inject(self, payload_bytes: int, with_jitter: bool = True) -> float:
        """注入一次传输延迟，返回实测耗时（毫秒）。"""
        delay = self.estimate_ms(payload_bytes)
        if with_jitter and self.profile.jitter_ms:
            delay += self._rng.uniform(-self.profile.jitter_ms, self.profile.jitter_ms)
        delay = max(delay, 0.0)
        if self.enable_sleep:
            started = time.perf_counter()
            time.sleep(delay / 1000.0)
            return (time.perf_counter() - started) * 1000.0
        return delay

    def describe(self) -> str:
        return (f"{self.profile.name}: {self.profile.bandwidth_mbps:g} Mbps / "
                f"RTT {self.profile.rtt_ms:g} ms")

    @staticmethod
    def available_profiles() -> List[str]:
        return list(NETWORK_PRESETS.keys())

    @staticmethod
    def profile_table() -> Dict[str, Dict[str, float]]:
        return {
            name: {"bandwidth_mbps": p.bandwidth_mbps, "rtt_ms": p.rtt_ms, "jitter_ms": p.jitter_ms}
            for name, p in NETWORK_PRESETS.items()
        }
