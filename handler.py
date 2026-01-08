import os
import json
from pathlib import Path
import runpod
from inference import get_inference_service
import cv2
import base64
import numpy as np

service = get_inference_service()


def generator_handler(job):
    job_input = job["input"]
    audio_data = job_input.get("audio_data")
    model_name = job_input.get("model_name", "Iris")

    audio_bytes = base64.b64decode(audio_data)
    aud = np.frombuffer(audio_bytes, dtype=np.int16)
    aud = aud.astype(np.float32) / 32768.0
    for frame, _ in service[model_name].generate_frames(aud):
        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        yield {"frame": jpg_as_text}


runpod.serverless.start({"handler": generator_handler, "return_aggregate_stream": True})
