import runpod
import base64
import time


def generator_handler(job):
    for i in range(5):
        payload = f"chunk_{i}_data".encode("utf-8")

        yield {
            "status": "processing",
            "i": i,
            "chunk_b64": base64.b64encode(payload).decode("ascii"),
        }

        time.sleep(0.2)

    yield {"status": "completed"}


runpod.serverless.start({"handler": generator_handler, "return_aggregate_stream": True})
