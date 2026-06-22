import base64
import json
import os
import tempfile
import time
from typing import Literal

import cv2
import requests
from PIL import Image


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


class Minimax:
    def __init__(
        self,
        model_name: str,
        generation_type: Literal["t2v", "i2v", "v2v"],
        url: str,
        model: str,
    ):
        self.api_key = os.environ["MINIMAX_API_KEY"]
        self.model_name = model_name
        self.generation_type = generation_type
        self.url = url
        self.model = model

    def invoke_video_generation(self, prompt: str, image_path: str) -> str:
        print("-----------------Submit video generation task-----------------")

        with open(image_path, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode("utf-8")

        payload = json.dumps(
            {
                "prompt": prompt,
                "model": self.model,
                "first_frame_image": f"data:image/jpeg;base64,{data}",
            }
        )
        headers = {
            "authorization": "Bearer " + self.api_key,
            "content-type": "application/json",
        }

        response = requests.request("POST", self.url, headers=headers, data=payload)
        print(response.text)
        task_id = response.json()["task_id"]
        print("Video generation task submitted successfully, task ID:" + task_id)
        return task_id

    def query_video_generation(self, task_id: str):
        url = "https://api.minimaxi.chat/v1/query/video_generation?task_id=" + task_id
        headers = {"authorization": "Bearer " + self.api_key}
        response = requests.request("GET", url, headers=headers)
        status = response.json()["status"]
        if status == "Preparing":
            print("...Preparing...")
            return "", "Preparing"
        elif status == "Queueing":
            print("...In the queue...")
            return "", "Queueing"
        elif status == "Processing":
            print("...Generating...")
            return "", "Processing"
        elif status == "Success":
            return response.json()["file_id"], "Finished"
        elif status == "Fail":
            return "", "Fail"
        else:
            return "", "Unknown"

    def fetch_video_result(self, file_id: str):
        print(
            "---------------Video generated successfully, downloading now---------------"
        )
        url = "https://api.minimaxi.chat/v1/files/retrieve?file_id=" + file_id
        headers = {
            "authorization": "Bearer " + self.api_key,
        }

        response = requests.request("GET", url, headers=headers)
        print(response.text)

        download_url = response.json()["file"]["download_url"]
        print("Video download link:" + download_url)
        frames = extract_frames_from_url(download_url)
        return frames

    def generate_video(self, prompt: str, image_path: str):
        task_id = self.invoke_video_generation(prompt, image_path)
        print("-----------------Video generation task submitted -----------------")

        frames = []
        while True:
            time.sleep(10)

            file_id, status = self.query_video_generation(task_id)
            if file_id != "":
                frames = self.fetch_video_result(file_id)
                print("---------------Successful---------------")
                break
            elif status == "Fail" or status == "Unknown":
                print("---------------Failed---------------")
                break

        return frames
