"""HyperLPR 端云计算划分运行时（队员 2 负责的端侧部分）。

对外最常用的是 ``PartitionedPipeline``：切策略、切网络、跑一次就拿到三段式耗时。
"""

from .config import PartitionMode, NetworkProfile, NETWORK_PRESETS, CLOUD_ENDPOINT
from .netsim import NetworkSimulator
from .pipeline import PartitionedPipeline, StageMetrics

__all__ = [
    "PartitionMode",
    "NetworkProfile",
    "NETWORK_PRESETS",
    "CLOUD_ENDPOINT",
    "NetworkSimulator",
    "PartitionedPipeline",
    "StageMetrics",
]

__version__ = "0.1.0"
