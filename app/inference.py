from pathlib import Path
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
import torch
import cv2
from tqdm import tqdm
from project_utils import validate_video_file, open_video_capture, draw_detections

def run_inference(model_name: str, input_video_path: Path, output_dir: Path) -> Path:
    validate_video_file(input_video_path)
    cap, meta = open_video_capture(input_video_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = f"/app/models/{model_name}.pt"

    detection_model = UltralyticsDetectionModel(
        model_path=model_path,
        confidence_threshold=0.3,
        device=device
    )

    output_path = output_dir / f"{model_name}_sahi_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, meta["fps"], (meta["width"], meta["height"]))

    for _ in range(meta["total_frames"]):
        ret, frame = cap.read()
        if not ret:
            break

        result = get_sliced_prediction(
            image=frame,
            detection_model=detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0
        )

        boxes, confidences, class_ids = [], [], []
        for obj in result.object_prediction_list:
            if obj.category.id != 0:
                continue
            bbox = obj.bbox.to_xyxy()
            boxes.append(bbox)
            confidences.append(obj.score.value)
            class_ids.append(obj.category.id)

        frame = draw_detections(frame, boxes, confidences, class_ids)
        out.write(frame)

    cap.release()
    out.release()
    return output_path