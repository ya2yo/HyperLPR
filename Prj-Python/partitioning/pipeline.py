"""可插拔的端云划分流水线（issue #4 的核心交付物）。

同一张输入图，切换 ``PartitionMode`` 就走不同的端云分工，并且每次运行都会
输出三段式分解日志：端侧耗时 / 传输耗时 / 云侧耗时。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

import numpy as np

from .config import PartitionMode, NETWORK_PRESETS, DEFAULT_NETWORK
from .netsim import NetworkSimulator
from .transport import encode_jpeg
from .ops import detect_plates, align_crop, box_crop, run_recognition, run_classification, \
    resolve_plate_type, full_recognize


@dataclass
class StageMetrics:
    """一次识别的全链路耗时分解（单位毫秒，载荷单位字节）。"""

    mode: str = ""
    network: str = ""
    det_ms: float = 0.0
    align_ms: float = 0.0
    rec_ms: float = 0.0
    cls_ms: float = 0.0
    edge_ms: float = 0.0
    net_up_ms: float = 0.0
    cloud_ms: float = 0.0
    net_down_ms: float = 0.0
    total_ms: float = 0.0
    payload_up: int = 0
    payload_down: int = 0
    plate_count: int = 0

    def finalize(self) -> "StageMetrics":
        self.edge_ms = self.det_ms + self.align_ms + self.rec_ms + self.cls_ms
        self.total_ms = self.edge_ms + self.net_up_ms + self.cloud_ms + self.net_down_ms
        return self

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"[{self.mode} @ {self.network}] 端侧 {self.edge_ms:.2f} ms | "
                f"上行 {self.net_up_ms:.2f} ms ({self.payload_up / 1024:.2f} KB) | "
                f"云侧 {self.cloud_ms:.2f} ms | 下行 {self.net_down_ms:.2f} ms | "
                f"合计 {self.total_ms:.2f} ms")


class PartitionedPipeline:
    """按静态策略在端侧与云端之间切分 HyperLPR 流水线。"""

    def __init__(self,
                 mode=PartitionMode.P2_EDGE_DET_CLOUD_REC,
                 network: str = DEFAULT_NETWORK,
                 cloud=None,
                 engine=None,
                 simulate: bool = True):
        from .engine import get_engine
        from .transport import LocalCloudStub

        self.mode = PartitionMode.parse(mode)
        self.engine = engine or get_engine()
        self.cloud = cloud or LocalCloudStub(self.engine)
        self.netsim = NetworkSimulator(network, enable_sleep=simulate)

    # ---- 运行时切换（Part C 动态决策引擎会调用） -------------------------
    def switch_mode(self, mode) -> PartitionMode:
        self.mode = PartitionMode.parse(mode)
        return self.mode

    def switch_network(self, network: str):
        return self.netsim.switch(network)

    # ---- 主入口 ---------------------------------------------------------
    def run(self, image: np.ndarray) -> Tuple[List[dict], StageMetrics]:
        m = StageMetrics(mode=self.mode.value, network=self.netsim.profile.name)
        if self.mode == PartitionMode.P0_ALL_LOCAL:
            plates = self._run_all_local(image, m)
        elif self.mode == PartitionMode.P1_ALL_CLOUD:
            plates = self._run_all_cloud(image, m)
        elif self.mode == PartitionMode.P2_EDGE_DET_CLOUD_REC:
            plates = self._run_edge_det_align(image, m)
        elif self.mode == PartitionMode.P3_EDGE_DET_CLOUD_ALIGN_REC:
            plates = self._run_edge_det_only(image, m)
        else:
            raise NotImplementedError(self.mode)
        m.plate_count = len(plates)
        m.finalize()
        return plates, m

    def __call__(self, image: np.ndarray):
        return self.run(image)

    # ---- 四种策略 -------------------------------------------------------
    def _run_all_local(self, image: np.ndarray, m: StageMetrics) -> List[dict]:
        t0 = time.perf_counter()
        items = detect_plates(self.engine.detector, image)
        m.det_ms = (time.perf_counter() - t0) * 1000.0

        plates = []
        for item in items:
            t0 = time.perf_counter()
            pad = align_crop(image, item["landmarks"])
            m.align_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            code, conf = run_recognition(self.engine.recognizer, pad, item["layer_num"])
            m.rec_ms += (time.perf_counter() - t0) * 1000.0
            if not code:
                continue

            plate_type = resolve_plate_type(code, item["layer_num"])
            if plate_type == 0:
                t0 = time.perf_counter()
                cls_idx = run_classification(self.engine.classifier, pad)
                m.cls_ms += (time.perf_counter() - t0) * 1000.0
                plate_type = resolve_plate_type(code, item["layer_num"], cls_idx)

            plates.append({
                "plate_code": code,
                "rec_confidence": float(conf),
                "plate_type": int(plate_type),
                "det_bound_box": item["rect"].tolist(),
                "det_confidence": item["score"],
            })
        return plates

    def _run_all_cloud(self, image: np.ndarray, m: StageMetrics) -> List[dict]:
        payload = encode_jpeg(image)
        m.payload_up = len(payload)

        m.net_up_ms = self.netsim.inject(m.payload_up)
        t0 = time.perf_counter()
        resp = self.cloud.infer_full(payload)
        m.cloud_ms = (time.perf_counter() - t0) * 1000.0

        down = self._dump_payload(resp)
        m.payload_down = len(down)
        m.net_down_ms = self.netsim.inject(m.payload_down)
        return resp.get("plates", [])

    def _run_edge_det_align(self, image: np.ndarray, m: StageMetrics) -> List[dict]:
        t0 = time.perf_counter()
        items = detect_plates(self.engine.detector, image)
        m.det_ms = (time.perf_counter() - t0) * 1000.0

        plates = []
        for item in items:
            t0 = time.perf_counter()
            pad = align_crop(image, item["landmarks"])
            m.align_ms += (time.perf_counter() - t0) * 1000.0

            payload = encode_jpeg(pad)
            m.payload_up += len(payload)
            m.net_up_ms += self.netsim.inject(len(payload))

            t0 = time.perf_counter()
            resp = self.cloud.infer_rec_cls(payload, layer_num=item["layer_num"])
            m.cloud_ms += (time.perf_counter() - t0) * 1000.0

            down = self._dump_payload(resp)
            m.payload_down += len(down)
            m.net_down_ms += self.netsim.inject(len(down))

            for plate in resp.get("plates", []):
                plate.setdefault("det_bound_box", item["rect"].tolist())
                plate.setdefault("det_confidence", item["score"])
                plates.append(plate)
        return plates

    def _run_edge_det_only(self, image: np.ndarray, m: StageMetrics) -> List[dict]:
        t0 = time.perf_counter()
        items = detect_plates(self.engine.detector, image)
        m.det_ms = (time.perf_counter() - t0) * 1000.0

        plates = []
        for item in items:
            pad = box_crop(image, item["rect"])
            payload = encode_jpeg(pad)
            m.payload_up += len(payload)
            m.net_up_ms += self.netsim.inject(len(payload))

            t0 = time.perf_counter()
            resp = self.cloud.infer_rec_cls(payload, layer_num=item["layer_num"])
            m.cloud_ms += (time.perf_counter() - t0) * 1000.0

            down = self._dump_payload(resp)
            m.payload_down += len(down)
            m.net_down_ms += self.netsim.inject(len(down))

            for plate in resp.get("plates", []):
                plate.setdefault("det_bound_box", item["rect"].tolist())
                plate.setdefault("det_confidence", item["score"])
                plates.append(plate)
        return plates

    @staticmethod
    def _dump_payload(resp: dict) -> bytes:
        """粗略估计下行载荷大小（真实场景是 JSON 响应体）。"""
        import json

        def _default(obj):
            # numpy 标量/数组不是 JSON 原生类型，测量结果里混着 float32，统一降级
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        return json.dumps(resp, ensure_ascii=False, default=_default).encode("utf-8")
