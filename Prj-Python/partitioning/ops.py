"""流水线的四个原子算子：Detect / Align / Recognition / Classification。

把 hyperlpr3 原本写在 ``LPRMultiTaskPipeline.run`` 里的逻辑拆成可单独调用的算子，
这样端侧和云端才能各自只跑其中一段——这是可插拔流水线（issue #4）的基础。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from hyperlpr3.common.typedef import (
    DOUBLE, UNKNOWN,
    PLATE_TYPE_YELLOW, PLATE_TYPE_BLUE, PLATE_TYPE_GREEN,
    BLUE, GREEN, YELLOW_SINGLE, YELLOW_DOUBLE,
    code_filter,
)
from hyperlpr3.common.tools_process import get_rotate_crop_image


def detect_plates(detector, image: np.ndarray) -> List[dict]:
    """Stage 1: 多任务检测，输出框、置信度、四角点与单/双层标志。"""
    outputs = detector(image)
    plates = []
    for out in outputs:
        plates.append({
            "rect": out[:4].astype(int),
            "score": float(out[4]),
            "landmarks": out[5:13].reshape(4, 2).astype(int),
            "layer_num": int(out[13]),
        })
    return plates


def align_crop(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Stage 2: 透视变换对齐，得到 48x160 级别的车牌切片。"""
    return get_rotate_crop_image(image, landmarks)


def box_crop(image: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """P3 专用：不做透视矫正，只按检测框直裁（云端再做精算）。"""
    x1, y1, x2, y2 = [int(v) for v in rect]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2].copy()


def run_recognition(recognizer, pad: np.ndarray, layer_num: int = 0) -> Tuple[str, float]:
    """Stage 3: CRNN + CTC 字符识别，双层牌分段识别后拼接。"""
    if layer_num == DOUBLE:
        h = pad.shape[0]
        line = int(h * 0.4)
        top_code, top_conf = recognizer(pad[:line, :, ])
        bottom_code, bottom_conf = recognizer(pad[line:, :])
        return top_code + bottom_code, (top_conf + bottom_conf) / 2.0
    code, conf = recognizer(pad)
    return code, float(conf)


def run_classification(classifier, pad: np.ndarray) -> int:
    """Stage 4: 车牌属性分类，返回类别索引。"""
    cls = classifier(pad)
    return int(np.argmax(cls))


def resolve_plate_type(plate_code: str, layer_num: int, cls_index: int = -1) -> int:
    """先用号码规则判类型，判不出来再查分类结果。"""
    plate_type = code_filter(plate_code)
    if plate_type == UNKNOWN and cls_index >= 0:
        if cls_index == PLATE_TYPE_YELLOW:
            plate_type = YELLOW_DOUBLE if layer_num == DOUBLE else YELLOW_SINGLE
        elif cls_index == PLATE_TYPE_BLUE:
            plate_type = BLUE
        elif cls_index == PLATE_TYPE_GREEN:
            plate_type = GREEN
    return int(plate_type)


def full_recognize(recognizer, classifier, pad: np.ndarray, layer_num: int = 0) -> Tuple[str, float, int]:
    """一次跑完 Stage 3 + Stage 4（云端 P2 接口内部就用这个）。"""
    code, conf = run_recognition(recognizer, pad, layer_num)
    cls_index = run_classification(classifier, pad)
    return code, conf, resolve_plate_type(code, layer_num, cls_index)
