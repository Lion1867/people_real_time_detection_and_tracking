from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import shutil
import uuid
from inference import run_inference

app = FastAPI(title="Person Detection API", version="1.0")

UPLOAD_DIR = Path("/tmp/uploads")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

@app.get("/", summary="Корневой эндпоинт")
async def root():
    return JSONResponse(
        content={
            "service": "Person Detection API",
            "status": "running",
            "docs": "/docs"
        }
    )


@app.get("/health", summary="Health check")
async def health():
    return JSONResponse(content={"status": "healthy"})

@app.post("/detect")
async def detect_people(
    model_name: str = Form(..., regex="^(rtdetr-x|yolov8x)$"),
    video: UploadFile = File(...)
):
    video_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{video_id}.mp4"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    output_path = run_inference(
        model_name=model_name,
        input_video_path=input_path,
        output_dir=OUTPUT_DIR
    )

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"{model_name}_output.mp4"
    )