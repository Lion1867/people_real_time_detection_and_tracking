import gradio as gr
import requests
import tempfile
import os

API_URL = "http://app:8000/detect"

def process_video(video_file, model_name):
    with open(video_file, "rb") as f:
        files = {"video": f}
        data = {"model_name": model_name}
        resp = requests.post(API_URL, files=files, data=data)
    if resp.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(resp.content)
            return tmp.name
    else:
        raise gr.Error("Ошибка обработки видео")

demo = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Загрузите видео"),
        gr.Radio(choices=["rtdetr-x", "yolov8x"], label="Выберите модель", value="yolov8x")
    ],
    outputs=gr.Video(label="Результат"),
    title="Обнаружение людей на видео",
    description="Загрузите видео и выберите модель для детекции людей."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)