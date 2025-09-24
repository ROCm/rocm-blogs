import asyncio
import json
import logging
import uuid
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from redis.asyncio import Redis

app = FastAPI()
redis = Redis(host="redis", port=6379, db=0, decode_responses=False)

IN_Q = "jobs"  # incoming jobs list (JSON bytes). API RPUSHes, worker BLPOPs.
RESP_Q_PREFIX = "job_resp:"  # per-job response list; worker RPUSHes, API BLPOPs.


class Item(BaseModel):
    name: str
    message: str
    prompt: str


async def push_job(payload: Item) -> str:
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "payload": payload.model_dump(),  # no bytes here; we encode below
    }
    await redis.rpush(IN_Q, json.dumps(job).encode("utf-8"))
    return job_id


async def wait_for_video(job_id: str, timeout_s: int = 300) -> Optional[bytes]:
    key = f"{RESP_Q_PREFIX}{job_id}"
    # BLPOP returns (key, value) or None on timeout
    result = await redis.blpop(key, timeout=timeout_s)
    if result is None:
        return None
    _, video_bytes = result
    # optional: delete the (now-empty) per-job key to keep Redis tidy
    await redis.delete(key)
    # sanity checks
    if not isinstance(video_bytes, (bytes, bytearray, memoryview)):
        logging.error(f"Non-bytes payload for {job_id}: type={type(video_bytes)}")
        return None
    if len(video_bytes) < 24:
        logging.error(
            f"Suspiciously small video for {job_id}: {len(video_bytes)} bytes"
        )
    logging.info(f"Received video for {job_id}: {len(video_bytes)} bytes")
    return bytes(video_bytes)


@app.post("/generate/", response_class=StreamingResponse)
async def generate_video(payload: Item):
    job_id = await push_job(payload)
    video = await wait_for_video(job_id, timeout_s=300)
    if video is None:
        raise HTTPException(status_code=504, detail="Video generation timed out")
    return StreamingResponse(
        BytesIO(video),
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{job_id}.mp4"'},
    )
