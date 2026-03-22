"""Minimal OpenAI-compatible Whisper STT server using faster-whisper."""

import io
import logging
import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-stt")

app = FastAPI(title="Whisper STT")
model = None


@app.on_event("startup")
def startup():
    global model
    model_size = os.environ.get("WHISPER_MODEL", "base")
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
    logger.info(f"Loading faster-whisper model={model_size} device={device} compute={compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    logger.info("Whisper model loaded.")


@app.get("/health")
def health():
    return {"status": True}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model_name: str = Form(default="whisper-1", alias="model"),
    language: str = Form(default=None),
):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments)
    finally:
        os.unlink(tmp_path)

    return {"text": text}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("STT_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
