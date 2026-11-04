from pathlib import Path
import httpx
import asyncio
from partitioning.server.api import app

images = Path(__file__).resolve().parents[2] / "resource/images"


def request(path, image, data=None):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post(path, files={"file": ("sample.jpg", image, "image/jpeg")}, data=data)
    return asyncio.run(run())


def test_infer_full():
    response = request("/api/v1/infer/full", (images / "test_img.jpg").read_bytes())
    assert response.status_code == 200, response.text
    assert response.json()["plates"][0]["plate_code"]
    assert response.json()["stage_ms"]["detect"] > 0


def test_infer_rec_cls():
    path = next((images / "rec_crop").glob("*.jpg"))
    response = request("/infer/rec_cls", path.read_bytes(), {"layer_num": "0"})
    assert response.status_code == 200, response.text
    assert response.json()["plates"][0]["plate_code"]
    assert response.json()["stage_ms"]["detect"] == 0


def test_invalid_image():
    assert request("/infer/full", b"broken").status_code == 400
    assert request("/api/v1/infer/rec_cls", b"", {"layer_num": "3"}).status_code == 422
