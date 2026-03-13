"""FastAPI app entrypoint."""

import logging
from typing import Dict

import uvicorn
from fastapi import FastAPI

from src.v1.router import router_v1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mic Speech Recognize API",
    description="Multi-channel mic synchronization and speaker diarization service",
    version="1.0.0",
)

app.include_router(router_v1)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("src.v1.api:app", host="0.0.0.0", port=8017, reload=True)
