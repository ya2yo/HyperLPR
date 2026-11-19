"""端云计算划分的静态配置：划分策略、网络剖面、云端地址。

本模块只放"配置"，不放任何计算逻辑，方便在实验里整体替换。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class PartitionMode(Enum):
    """四种静态划分策略。

    P0: 全端侧执行，零网络流量。
    P1: 全云端卸载，上传完整原图（流量最大）。
    P2: 端侧跑 Detect + Align，只上传矫正后的车牌切片（黄金切分点）。
    P3: 端侧只跑 Detect，上传未矫正的矩形裁剪区，云端做 Align + Rec + Cls。
    """

    P0_ALL_LOCAL = "P0_ALL_LOCAL"
    P1_ALL_CLOUD = "P1_ALL_CLOUD"
    P2_EDGE_DET_CLOUD_REC = "P2_EDGE_DET_CLOUD_REC"
    P3_EDGE_DET_CLOUD_ALIGN_REC = "P3_EDGE_DET_CLOUD_ALIGN_REC"

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        key = str(value).strip().upper()
        if not key.startswith("P"):
            key = "P" + key
        for item in cls:
            if item.value == key or item.name == key:
                return item
        for item in cls:  # 允许简写 P0 / P1 / P2 / P3
            if item.value.startswith(key + "_"):
                return item
        raise ValueError(f"未知的划分策略: {value}，可选 {[m.value for m in cls]}")

    @property
    def short(self) -> str:
        return self.value.split("_")[0]


@dataclass(frozen=True)
class NetworkProfile:
    """一个典型移动云计算网络剖面。

    bandwidth_mbps: 带宽（兆比特每秒）
    rtt_ms: 往返时延（毫秒）
    """

    name: str
    bandwidth_mbps: float
    rtt_ms: float
    jitter_ms: float = 0.0

    @property
    def bandwidth_bps(self) -> float:
        return self.bandwidth_mbps * 1_000_000.0

    def transmission_ms(self, payload_bytes: int) -> float:
        """标准 MCC 传输模型: T_net = RTT/2 + Payload * 8 / Bandwidth."""
        bits = float(payload_bytes) * 8.0
        return self.rtt_ms / 2.0 + bits / self.bandwidth_bps * 1000.0

    def throughput_mbps(self) -> float:
        return self.bandwidth_mbps


NETWORK_PRESETS = {
    "5G": NetworkProfile("5G", 100.0, 10.0, jitter_ms=2.0),
    "WIFI": NetworkProfile("WIFI", 40.0, 20.0, jitter_ms=3.0),
    "4G": NetworkProfile("4G", 12.0, 50.0, jitter_ms=8.0),
    "3G_WEAK": NetworkProfile("3G_WEAK", 1.5, 150.0, jitter_ms=30.0),
    "POOR": NetworkProfile("POOR", 0.5, 300.0, jitter_ms=60.0),
}

DEFAULT_NETWORK = "WIFI"

# 云端微服务地址（队员 3 负责的 FastAPI 服务）
CLOUD_ENDPOINT = os.environ.get("LPR_CLOUD_ENDPOINT", "http://127.0.0.1:8000")

# 切片上传的 JPEG 编码质量：95 时 48x160 切片约 9~10 KB
JPEG_QUALITY = int(os.environ.get("LPR_JPEG_QUALITY", "95"))
