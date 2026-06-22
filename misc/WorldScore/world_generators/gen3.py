import base64
import tempfile
import time
from typing import Literal

import cv2
import requests
from PIL import Image
from runwayml import RunwayML


def extract_frames_from_url(video_url):
    frames = []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp_file:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp_file.write(chunk)
        tmp_file.flush()

        cap = cv2.VideoCapture(tmp_file.name)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)

        finally:
            cap.release()

    return frames


class Gen3:
    def __init__(
        self,
        model_name: str,
        generation_type: Literal["t2v", "i2v", "v2v"],
        model_id: str,
    ):
        self.model_name = model_name
        self.generation_type = generation_type
        self.model_id = model_id
        self.client = RunwayML()

    def generate_video(
        self,
        prompt: str,
        image_path: str | None,
    ):
        # encode image to base64
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # Create a new image-to-video task using the "gen3a_turbo" model
        task = self.client.image_to_video.create(
            model=self.model_id,
            # Point this at your own image file
            prompt_image=f"data:image/png;base64,{base64_image}",
            prompt_text=prompt,
        )
        task_id = task.id

        # Poll the task until it's complete
        time.sleep(10)  # Wait for a second before polling
        task = self.client.tasks.retrieve(task_id)
        while task.status not in ["SUCCEEDED", "FAILED"]:
            time.sleep(10)  # Wait for ten seconds before polling
            task = self.client.tasks.retrieve(task_id)

        print("Task complete:", task)
        video_url = task.output[0]
        frames = extract_frames_from_url(video_url)

        return frames
