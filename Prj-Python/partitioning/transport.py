"""端云传输层：图像编解码 + 云端 REST 客户端 + 本地等价桩。

对外只暴露三个动作，端侧与云端共用同一套语义：

* ``ping(payload)``         —— 测速探针
* ``infer_full(image)``     —— P1：上传整图，云端跑全流程
* ``infer_rec_cls(patch)``  —— P2/P3：上传切片，云端跑识别 + 分类

``LocalCloudStub`` 在队员 3 的 FastAPI 服务还没就绪时给出等价的本地实现，
让端侧代码可以先跑通、先出数据。
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import requests

from .config import CLOUD_ENDPOINT, JPEG_QUALITY


def encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """把 ndarray 编码成 JPEG 字节流（上传载荷）。"""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG 编码失败")
    return buf.tobytes()


def decode_jpeg(payload: bytes) -> np.ndarray:
    """把 JPEG 字节流还原成 BGR 图像。"""
    arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG 解码失败")
    return image


class CloudClient:
    """真实云端客户端（对接队员 3 的 FastAPI 服务）。"""

    def __init__(self, endpoint: str = CLOUD_ENDPOINT, timeout: float = 10.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def ping(self, payload_bytes: int = 1024) -> float:
        started = time.perf_counter()
        resp = self.session.get(f"{self.endpoint}/ping", params={"n": payload_bytes}, timeout=self.timeout)
        resp.raise_for_status()
        return (time.perf_counter() - started) * 1000.0

    def infer_full(self, image_bytes: bytes) -> dict:
        files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
        resp = self.session.post(f"{self.endpoint}/infer/full", files=files, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def infer_rec_cls(self, patch_bytes: bytes, layer_num: int = 0) -> dict:
        files = {"file": ("patch.jpg", patch_bytes, "image/jpeg")}
        data = {"layer_num": str(int(layer_num))}
        resp = self.session.post(f"{self.endpoint}/infer/rec_cls", files=files, data=data,
                                 timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def __repr__(self) -> str:
        return f"CloudClient({self.endpoint})"


class LocalCloudStub:
    """本地云端桩：接口完全一致，但推理发生在本地（用于离线联调与基线对照）。"""

    def __init__(self, engine=None):
        from .engine import get_engine
        self.engine = engine or get_engine()

    def ping(self, payload_bytes: int = 1024) -> float:
        return 0.0

    def infer_full(self, image_bytes: bytes) -> dict:
        from .ops import detect_plates, align_crop, full_recognize

        image = decode_jpeg(image_bytes)
        plates = []
        for item in detect_plates(self.engine.detector, image):
            pad = align_crop(image, item["landmarks"])
            code, conf, plate_type = full_recognize(self.engine.recognizer, self.engine.classifier,
                                                    pad, item["layer_num"])
            if not code:
                continue
            plates.append({
                "plate_code": code,
                "rec_confidence": conf,
                "plate_type": plate_type,
                "det_bound_box": item["rect"].tolist(),
                "det_confidence": item["score"],
            })
        return {"plates": plates}

    def infer_rec_cls(self, patch_bytes: bytes, layer_num: int = 0) -> dict:
        from .ops import full_recognize

        pad = decode_jpeg(patch_bytes)
        code, conf, plate_type = full_recognize(self.engine.recognizer, self.engine.classifier,
                                                pad, int(layer_num))
        if not code:
            return {"plates": []}
        return {"plates": [{
            "plate_code": code,
            "rec_confidence": conf,
            "plate_type": plate_type,
            "det_bound_box": [],
            "det_confidence": -1.0,
        }]}

    def __repr__(self) -> str:
        return "LocalCloudStub(local)"


def build_cloud(endpoint: Optional[str] = None, offline: bool = False):
    """工厂：给了地址且服务可用就走真实云端，否则退回本地桩。

    探测是"尽力而为"的：ping 不通说明队员 3 的 FastAPI 还没起，
    这时不能让端侧实验卡死，直接切到本地等价桩继续出数据。
    """
    if offline or not endpoint:
        return LocalCloudStub()
    client = CloudClient(endpoint)
    try:
        client.ping()
        return client
    except Exception as exc:  # noqa: BLE001 - 任何网络异常都退化为本地桩
        print(f"[warn] 云端 {endpoint} 不可达（{type(exc).__name__}），自动退回 LocalCloudStub",
              file=sys.stderr)
        return LocalCloudStub()
