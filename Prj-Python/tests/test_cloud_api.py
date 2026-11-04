import asyncio
from io import BytesIO

import cv2
import numpy as np
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from hyperlpr3.command import cloud_api
from hyperlpr3.inference.pipeline import LPRMultiTaskPipeline


class FakeService:

    def recognizeFull(self, image):
        assert image.shape == (20, 40, 3)
        return [["粤B12345", 0.96, 0, [1, 2, 30, 18], [[1, 2], [30, 2], [30, 18], [1, 18]]]]

    def recognizePatch(self, image, layerNum):
        assert image.shape == (20, 40, 3)
        return "粤B12345", 0.95, 0


def imageFile():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return UploadFile(
        filename="plate.jpg",
        file=BytesIO(encoded.tobytes()),
        headers=Headers({"content-type": "image/jpeg"})
    )


def testRecognizeFull():
    originalService = cloud_api.service
    cloud_api.service = FakeService()
    try:
        result = asyncio.run(cloud_api.recognizeFull(imageFile()))
    finally:
        cloud_api.service = originalService

    assert result["code"] == 0
    assert result["data"]["mode"] == "P1"
    assert result["data"]["plates"][0]["code"] == "粤B12345"
    assert result["data"]["metrics"]["imageWidth"] == 40


def testRecognizePatch():
    originalService = cloud_api.service
    cloud_api.service = FakeService()
    try:
        result = asyncio.run(cloud_api.recognizePatch(imageFile(), 0))
    finally:
        cloud_api.service = originalService

    assert result["data"]["mode"] == "P2"
    assert result["data"]["plates"][0]["confidence"] == 0.95


def testRejectInvalidImage():
    try:
        cloud_api.decodeImage(b"not an image", "image/jpeg")
        assert False
    except HTTPException as exception:
        assert exception.status_code == 400


def testPipelinePatchRecognition():
    recognizer = lambda image: ("粤B12345", 0.9)
    classifier = lambda image: np.array([1.0, 0.0, 0.0])
    pipeline = LPRMultiTaskPipeline(None, recognizer, classifier)

    code, conf, plateType = pipeline.recognizePatch(np.zeros((20, 40, 3), dtype=np.uint8), 0)
    assert code == "粤B12345"
    assert conf == 0.9
    assert plateType == 0
