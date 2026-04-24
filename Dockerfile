ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update -y && apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --break-system-packages -e .

# Map host directory to cache Whisper and Opus-MT models
VOLUME /root/.cache/huggingface/hub

# Map host directories for input and output files
VOLUME /app/input
VOLUME /app/output

ENTRYPOINT ["./entrypoint.sh"]