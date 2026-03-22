ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install/upgrade deps — use PyPI directly to avoid unreachable Jetson index issues
# Upgrade transformers for Gemma-3 vision support (needs >=4.50)
RUN pip install --no-cache-dir --extra-index-url https://pypi.org/simple/ \
    fastapi uvicorn pillow httpx "transformers>=4.50" accelerate

COPY server.py /opt/vlm-bridge/
COPY models/ /opt/vlm-bridge/models/

ARG VLM_MODEL=gemma
ENV VLM_MODEL=${VLM_MODEL}
ENV VLM_PORT=8001
EXPOSE 8001

WORKDIR /opt/vlm-bridge
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s \
    CMD curl -sf http://localhost:8001/health || exit 1

CMD ["python3", "server.py"]
