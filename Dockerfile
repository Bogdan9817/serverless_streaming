FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime


WORKDIR /app

RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y ffmpeg
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY inference.py .
COPY utils.py .
COPY unet_328.py .


CMD [ "python", "-u", "handler.py" ]