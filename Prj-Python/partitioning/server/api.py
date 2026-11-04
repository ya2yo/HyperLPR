import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from time import perf_counter

import cv2
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, UploadFile
from hyperlpr3.command.cloud_api import MAX_IMAGE_BYTES, decodeImage, service
from partitioning.ops import align_crop, detect_plates, resolve_plate_type, run_classification, run_recognition

app = FastAPI()
router = APIRouter()
workers = ThreadPoolExecutor(max_workers=2)
classifiers = ThreadPoolExecutor(max_workers=2)
slots = BoundedSemaphore(6)
TIMEOUT = 10


def infer(content, mime, layer):
    times = dict(decode=0, detect=0, align=0, recognize=0, classify=0)

    def measure(stage, fn, *args):
        start = perf_counter()
        result = fn(*args)
        times[stage] += (perf_counter() - start) * 1000
        return result

    try:
        image = measure("decode", decodeImage, content, mime)
    except cv2.error as exc:
        raise HTTPException(400, "Invalid image") from exc
    if layer == 1 and image.shape[0] < 3:
        raise HTTPException(422, "Double-layer patch too short")
    pipeline = service.load().pipeline
    detections = measure("detect", detect_plates, pipeline.detector, image) if layer is None else [None]
    plates = []
    for detection in detections:
        patch = measure("align", align_crop, image, detection["landmarks"]) if detection else image
        layers = detection["layer_num"] if detection else layer
        classification = classifiers.submit(measure, "classify", run_classification, pipeline.classifier, patch)
        try:
            code, confidence = measure("recognize", run_recognition, pipeline.recognizer, patch, layers)
        finally:
            cls = classification.result()
        if code and math.isfinite(confidence):
            plates.append(dict(plate_code=code, rec_confidence=float(confidence),
                               plate_type=resolve_plate_type(code, layers, cls),
                               det_bound_box=detection["rect"].tolist() if detection else [],
                               det_confidence=detection["score"] if detection else -1.0))
    return dict(plates=plates, stage_ms=times, payload_bytes=len(content))


async def dispatch(file, layer):
    content = await file.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large")
    if not slots.acquire(blocking=False):
        raise HTTPException(503, "Inference queue full")
    try:
        job = workers.submit(infer, content, file.content_type, layer)
    except BaseException:
        slots.release()
        raise
    job.add_done_callback(lambda _: slots.release())
    try:
        return await asyncio.wait_for(asyncio.wrap_future(job), TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(504, "Inference timed out")


@router.get("/ping")
def ping(n: int = Query(1024, ge=0)):
    return {"echo_bytes": n, "server_time_ms": perf_counter() * 1000}


@router.post("/infer/full")
async def full(file: UploadFile = File(...)):
    return await dispatch(file, None)


@router.post("/infer/rec_cls")
async def rec_cls(file: UploadFile = File(...), layer_num: int = Form(0)):
    if layer_num not in (0, 1):
        raise HTTPException(422, "Invalid layer_num")
    return await dispatch(file, layer_num)


app.include_router(router)
app.include_router(router, prefix="/api/v1")
