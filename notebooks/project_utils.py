from pathlib import Path
import mimetypes
import cv2

def validate_video_file(video_path: Path) -> None:
    """Проверяет существование, непустоту и поддерживаемость видеофайла."""
    if not video_path.exists():
        raise FileNotFoundError(f"Файл не найден: {video_path.resolve()}")
    if video_path.stat().st_size == 0:
        raise ValueError(f"Файл пустой: {video_path.resolve()}")

    VIDEO_EXTENSIONS = {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm',
        '.m4v', '.mpeg', '.mpg', '.3gp', '.mxf', '.mts', '.vob'
    }
    ext = video_path.suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise ValueError(f"Неподдерживаемое расширение: '{ext}'")

    mime_type, _ = mimetypes.guess_type(str(video_path))
    if mime_type is None or not mime_type.startswith('video'):
        raise ValueError(f"Файл не определён как видео по MIME-типу: {mime_type}")


def open_video_capture(video_path: Path):
    """Открывает видео и возвращает объект захвата + метаданные."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"OpenCV не может открыть видео: {video_path.resolve()}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("Некорректные метаданные видео")

    metadata = {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames
    }
    return cap, metadata

def draw_detections(frame, boxes, confidences, class_ids):
    """Отрисовывает bounding boxes и confidence только для класса 'person' (id=0)."""
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        if cls_id != 0:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person: {conf:.2f}"
        cv2.putText(
            frame, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )
    return frame