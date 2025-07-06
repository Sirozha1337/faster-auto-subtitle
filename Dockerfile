FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update -y && apt-get install -y ffmpeg

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN pip install -e .

# Map host directory to cache Whisper and Opus-MT models
VOLUME /root/.cache/huggingface/hub

# Map host directories for input and output files
VOLUME /app/input
VOLUME /app/output

ENTRYPOINT ["./entrypoint.sh"]