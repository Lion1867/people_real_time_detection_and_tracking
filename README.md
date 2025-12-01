# Person Detection Service  

Данный проект реализует микросервис для детекции людей на видео с возможностью выбора между двумя предобученными моделями: **YOLOv8x** и **RT-DETRx** с использованием техники **SAHI (Slicing Aided Hyper Inference)** для повышения качества обнаружения мелких объектов.

---

## Описание

Проект обрабатывает видеофайл `crowd.mp4`, детектирует людей с помощью двух разных архитектур и сохраняет видео с bounding boxes и confidence scores.  
Поддерживается:
- выбор модели через API или UI,
- кросс-платформенный запуск (Linux/macOS/Windows),
- контейнеризация через Docker и Docker Compose.

---

## Архитектура

- **FastAPI** — REST API для обработки видео (`/detect`).
- **Gradio** — веб-интерфейс для загрузки видео и выбора модели.
- **SAHI + Ultralytics** — единый pipeline для обеих моделей.
- **OpenCV** — работа с видео, отрисовка боксов.
- **Docker** — изоляция зависимостей и кроссплатформенность.

---

## Требования

- Docker и Docker Compose
- Git

---

## Установка и запуск

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/Lion1867/people_real_time_detection_and_tracking.git
cd people_real_time_detection_and_tracking
```

## 2. Поместите предобученные веса

Скачайте модели и положите их в папку:
app/models/
├── rtdetr-x.pt
└── yolov8x.pt

Веса можно получить через ultralytics:

```python
from ultralytics import YOLO
YOLO('yolov8x.pt')
YOLO('rtdetr-x.pt')
```

## 3. Соберите и запустите контейнеры

```bash
docker-compose up --build
Сервисы будут доступны:

FastAPI: http://localhost:8000/docs

Gradio UI: http://localhost:7860

Использование
Через Gradio (рекомендуется)
Откройте http://localhost:7860

Загрузите видео (поддерживается .mp4, .avi, и др.)

Выберите модель: yolov8x или rtdetr-x

Нажмите «Submit» — через несколько секунд получите обработанное видео

Через FastAPI (программно)
```

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "model_name=yolov8x" \
  -F "video=@crowd.mp4" \
  --output result.mp4
Поддерживаемые значения model_name: yolov8x, rtdetr-x
```

Структура проекта

```text
people_real_time_detection_and_tracking/
├── docker-compose.yml
├── app/                # Backend (FastAPI + inference)
│   ├── models/                 # Предобученные веса (.pt)
│   ├── main.py                 # API эндпоинт
│   ├── inference.py            # Единый pipeline SAHI
│   ├── project_utils.py        # Валидация, работа с видео
│   ├── requirements.txt
│   └── tests/                  # Тесты FastAPI
└── gradio_ui/                 # Frontend (Gradio UI)
```

Тестирование

```bash
cd app
pip install pytest
pytest tests/ -v
Тесты не требуют загрузки весов — они проверяют только слой API и валидацию.
```