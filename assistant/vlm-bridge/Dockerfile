ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir fastapi uvicorn[standard] pillow httpx

COPY server.py /opt/vlm-bridge/
COPY models/ /opt/vlm-bridge/models/

ARG VLM_MODEL=gemma
ENV VLM_MODEL=${VLM_MODEL}
ENV VLM_PORT=8001
EXPOSE 8001

WORKDIR /opt/vlm-bridge
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -sf http://localhost:8001/health || exit 1

CMD ["python3", "server.py"]
