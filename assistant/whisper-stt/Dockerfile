FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart faster-whisper

COPY server.py /opt/whisper-stt/

ENV WHISPER_MODEL=base
ENV WHISPER_DEVICE=cpu
ENV WHISPER_COMPUTE_TYPE=int8
ENV STT_PORT=8000
EXPOSE 8000

WORKDIR /opt/whisper-stt
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python3", "server.py"]
