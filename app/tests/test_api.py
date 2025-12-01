import pytest
from fastapi.testclient import TestClient
from main import app
from pathlib import Path
import tempfile
import cv2
import numpy as np

client = TestClient(app)

@pytest.fixture
def sample_video():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        height, width = 240, 320
        fps = 10
        out = cv2.VideoWriter(
            tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        for _ in range(2):
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        yield tmp.name
        Path(tmp.name).unlink()


def test_detect_valid_request(sample_video):
    """Тест успешной обработки видео с корректной моделью."""
    with open(sample_video, "rb") as f:
        response = client.post(
            "/detect",
            data={"model_name": "yolov8x"},
            files={"video": ("test.mp4", f, "video/mp4")},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) > 0


def test_detect_invalid_model():
    """Тест с недопустимым названием модели."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(b"fake video content")
        tmp.seek(0)
        response = client.post(
            "/detect",
            data={"model_name": "invalid_model"},
            files={"video": ("test.mp4", tmp, "video/mp4")},
        )
    assert response.status_code == 422 


def test_detect_empty_video():
    """Тест отправки пустого файла."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        response = client.post(
            "/detect",
            data={"model_name": "yolov8x"},
            files={"video": ("empty.mp4", tmp, "video/mp4")},
        )
    assert response.status_code == 500  


def test_detect_non_video_file():
    """Тест отправки не-видео (например, текстового файла)."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        tmp.write(b"This is not a video")
        tmp.seek(0)
        response = client.post(
            "/detect",
            data={"model_name": "rtdetr-x"},
            files={"video": ("not_video.txt", tmp, "text/plain")},
        )
    assert response.status_code == 500  


def test_detect_missing_video():
    """Тест без передачи видео."""
    response = client.post("/detect", data={"model_name": "yolov8x"})
    assert response.status_code == 422  


def test_detect_unsupported_extension():
    """Тест с расширением, не входящим в список разрешённых."""
    with tempfile.NamedTemporaryFile(suffix=".exe") as tmp:
        tmp.write(b"fake exe")
        tmp.seek(0)
        response = client.post(
            "/detect",
            data={"model_name": "yolov8x"},
            files={"video": ("malicious.exe", tmp, "application/octet-stream")},
        )
    assert response.status_code == 500  