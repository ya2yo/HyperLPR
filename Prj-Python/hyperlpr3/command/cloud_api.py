# -*- coding: utf-8 -*-
import time
from threading import Lock
from typing import List

import click
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import hyperlpr3 as lpr3
from hyperlpr3.common.typedef import DOUBLE, MONO


TYPE_NAMES = ["蓝牌", "黄牌单层", "白牌单层", "绿牌新能源", "黑牌港澳",
              "香港单层", "香港双层", "澳门单层", "澳门双层", "黄牌双层"]
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class RecognitionService:

    def __init__(self):
        self.catcher = None
        self.lock = Lock()

    def load(self):
        if self.catcher is not None:
            return self.catcher
        with self.lock:
            if self.catcher is None:
                self.catcher = lpr3.LicensePlateCatcher(
                    detect_level=lpr3.DETECT_LEVEL_HIGH,
                    full_result=True
                )
        return self.catcher

    def recognizeFull(self, image: np.ndarray) -> list:
        return self.load()(image)

    def recognizePatch(self, image: np.ndarray, layerNum: int) -> tuple:
        return self.load().pipeline.recognizePatch(image, layerNum)


service = RecognitionService()
app = FastAPI(
    title="HyperLPR3 Cloud API",
    version="1.0.0",
    docs_url="/api/v1/docs",
    description="P1 full-cloud and P2 edge-cloud plate recognition service"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def decodeImage(content: bytes, contentType: str) -> np.ndarray:
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    if contentType and contentType not in IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG and WebP images are supported.")

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Image data cannot be decoded.")
    return image


def typeName(plateType: int) -> str:
    if 0 <= plateType < len(TYPE_NAMES):
        return TYPE_NAMES[plateType]
    return "未知车牌"


def fullPlateResult(result: list) -> dict:
    code, conf, plateType, box = result[:4]
    plate = {
        "code": code,
        "confidence": float(conf),
        "type": int(plateType),
        "typeName": typeName(int(plateType)),
        "box": box,
    }
    if len(result) > 4:
        plate["vertices"] = result[4]
    return plate


def buildResponse(mode: str, content: bytes, image: np.ndarray, plates: list,
                  startTime: float, inferStart: float) -> dict:
    finishTime = time.perf_counter()
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "mode": mode,
            "plates": plates,
            "metrics": {
                "payloadBytes": len(content),
                "imageWidth": int(image.shape[1]),
                "imageHeight": int(image.shape[0]),
                "decodeMs": round((inferStart - startTime) * 1000, 3),
                "inferenceMs": round((finishTime - inferStart) * 1000, 3),
                "serverMs": round((finishTime - startTime) * 1000, 3),
            },
        },
    }


@app.get("/")
async def running():
    return {"service": "HyperLPR3 Cloud API", "status": "running"}


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "modelLoaded": service.catcher is not None}


@app.post("/api/v1/recognize/full", tags=["P1 full cloud"])
async def recognizeFull(file: UploadFile = File(...)):
    startTime = time.perf_counter()
    content = await file.read()
    image = decodeImage(content, file.content_type or "")
    inferStart = time.perf_counter()
    results = service.recognizeFull(image)
    plates = [fullPlateResult(result) for result in results if not np.isnan(float(result[1]))]
    return buildResponse("P1", content, image, plates, startTime, inferStart)


@app.post("/api/v1/recognize/patch", tags=["P2 edge cloud"])
async def recognizePatch(file: UploadFile = File(...), layerNum: int = Form(MONO)):
    if layerNum not in (MONO, DOUBLE):
        raise HTTPException(status_code=422, detail="layerNum must be 0 (single) or 1 (double).")

    startTime = time.perf_counter()
    content = await file.read()
    image = decodeImage(content, file.content_type or "")
    inferStart = time.perf_counter()
    code, conf, plateType = service.recognizePatch(image, layerNum)
    plates = [] if not code or np.isnan(conf) else [{
        "code": code,
        "confidence": conf,
        "type": plateType,
        "typeName": typeName(plateType),
    }]
    return buildResponse("P2", content, image, plates, startTime, inferStart)


@app.post("/api/v1/rec", tags=["Legacy compatibility"])
async def legacyRecognize(file: List[UploadFile] = File(...)):
    if len(file) != 1:
        raise HTTPException(status_code=400, detail="Exactly one image is required.")
    result = await recognizeFull(file[0])
    plates = [{
        "code": plate["code"],
        "conf": plate["confidence"],
        "plate_type": plate["typeName"],
        "box": plate["box"],
    } for plate in result["data"]["plates"]]
    return {
        "result": {"plate_list": plates},
        "code": 5000,
        "msg": "请求成功",
    }


def get_application():
    return app


@click.command(help="Exec HyperLPR3 cloud recognition server.")
@click.option("-host", "--host", default="0.0.0.0", type=str)
@click.option("-port", "--port", default=8715, type=int)
@click.option("-workers", "--workers", default=1, type=int)
def rest(host, port, workers):
    uvicorn.run(app="hyperlpr3.command.cloud_api:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    rest()
