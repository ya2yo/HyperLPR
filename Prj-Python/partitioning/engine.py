"""端侧本地推理引擎：按需加载 Detect / Rec / Cls 三个 ONNX 模型。

模型默认来自 ``~/.hyperlpr3``（``pip install hyperlpr3`` 时自带），
仓库里 ``resource/models/r2_mobile`` 下的 .mnn 是 C++ 侧的 MNN 模型，Python 用不到。
"""

from __future__ import annotations

import os
from typing import Optional

from hyperlpr3.common.typedef import DETECT_LEVEL_LOW, DETECT_LEVEL_HIGH


class LocalEngine:
    """把三个推理算子打包成一个可复用的端侧引擎。"""

    def __init__(self, detect_level: int = DETECT_LEVEL_LOW, folder: Optional[str] = None):
        import hyperlpr3  # noqa: F401  触发 configuration.initialization()
        from hyperlpr3.config.settings import onnx_runtime_config as ort_cfg, _DEFAULT_FOLDER_
        from hyperlpr3.inference.multitask_detect import MultiTaskDetectorORT
        from hyperlpr3.inference.recognition import PPRCNNRecognitionORT
        from hyperlpr3.inference.classification import ClassificationORT

        folder = folder or _DEFAULT_FOLDER_
        if detect_level == DETECT_LEVEL_LOW:
            det_key, det_size = "det_model_path_320x", (320, 320)
        elif detect_level == DETECT_LEVEL_HIGH:
            det_key, det_size = "det_model_path_640x", (640, 640)
        else:
            raise NotImplementedError(f"不支持的检测等级: {detect_level}")

        self.detect_level = detect_level
        self.detector = MultiTaskDetectorORT(os.path.join(folder, ort_cfg[det_key]), input_size=det_size)
        self.recognizer = PPRCNNRecognitionORT(os.path.join(folder, ort_cfg["rec_model_path"]),
                                               input_size=(48, 160))
        self.classifier = ClassificationORT(os.path.join(folder, ort_cfg["cls_model_path"]),
                                            input_size=(96, 96))


_ENGINE_CACHE = {}


def get_engine(detect_level: int = DETECT_LEVEL_LOW, folder: Optional[str] = None) -> LocalEngine:
    """模型加载很慢，全局复用一个实例。"""
    key = (detect_level, folder)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = LocalEngine(detect_level=detect_level, folder=folder)
    return _ENGINE_CACHE[key]
