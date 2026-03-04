"""FastAPI 앱 엔트리포인트."""

import logging
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from fastapi import FastAPI

from src.v1.main import get_engine
from src.v1.router import router_v1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ERes2Net 모델을 GPU 메모리에 사전 로드합니다...")
    try:
        get_engine()
        logger.info("ERes2Net 모델 로드 완료.")
    except Exception:
        logger.exception("ERes2Net 모델 로드 실패")
    yield


app = FastAPI(
    title="Mic Speech Recognize API",
    description="다채널 마이크 기반 화자 동기화 및 식별 서비스",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router_v1)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("src.v1.api:app", host="0.0.0.0", port=8017, reload=True)
