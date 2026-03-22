import base64
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

from models import get_adapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vlm-bridge")

adapter = None


@asynccontextmanager
async def lifespan(app):
    global adapter
    adapter = get_adapter()
    vlm = os.environ.get("VLM_MODEL", "gemma")
    logger.info(f"Loading VLM adapter: {vlm}")
    adapter.load()
    logger.info(f"VLM loaded: {adapter.model_name()}")
    yield


app = FastAPI(title="VLM Bridge", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str | list


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": adapter.model_name(),
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": True}


async def _extract_images(messages: list[ChatMessage]) -> list[Image.Image]:
    """Extract PIL images from OpenAI-format message content."""
    images = []
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for part in msg.content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            url = part.get("image_url", {}).get("url", "")
            try:
                if url.startswith("data:"):
                    _, data = url.split(",", 1)
                    img_bytes = base64.b64decode(data)
                    images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                elif url.startswith("http"):
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(url, timeout=30)
                        resp.raise_for_status()
                        images.append(
                            Image.open(io.BytesIO(resp.content)).convert("RGB")
                        )
            except Exception as e:
                logger.warning(f"Failed to load image from {url[:80]}: {e}")
    return images


def _make_chunk(content: str, model: str, finish_reason=None) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    images = await _extract_images(req.messages)
    messages = [
        m.model_dump() if hasattr(m, "model_dump") else m.dict()
        for m in req.messages
    ]
    model_id = adapter.model_name()

    if req.stream:
        async def event_stream():
            result = await adapter.generate(
                messages=messages,
                images=images,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stream=True,
            )
            async for chunk in result:
                yield _make_chunk(chunk, model_id)
            yield _make_chunk("", model_id, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = await adapter.generate(
        messages=messages,
        images=images,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=False,
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("VLM_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
