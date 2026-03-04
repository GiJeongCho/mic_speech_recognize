"""API 라우트 정의.

엔드포인트:
  POST /v1/recognize  — 다채널 마이크 파일 + 등록 화자 디렉토리 → 화자별 발화 구간
  GET  /v1/jobs/{id}   — 작업 진행률 조회
"""

import logging
import os
import shutil
import uuid
from typing import Dict, List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from .main import get_engine
from .utils.job import JobInfo, job_manager

logger = logging.getLogger(__name__)

router_v1 = APIRouter(prefix="/v1", tags=["mic-speaker"])

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DATA_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "resoursces", "test", "input_Data")
)


def _background_process(
    job_id: str,
    audio_paths: List[str],
    speakers_dir: str,
    threshold: float,
) -> None:
    """백그라운드에서 실행되는 화자 식별 파이프라인."""
    try:
        def update_progress(p: float) -> None:
            job_manager.update_progress(job_id, p)

        engine = get_engine()
        result = engine.process(
            audio_paths=audio_paths,
            speakers_dir=speakers_dir,
            threshold=threshold,
            progress_callback=update_progress,
        )
        job_manager.complete_job(job_id, result)
        logger.info("Job %s 완료", job_id)

    except Exception:
        logger.exception("Job %s 실패", job_id)
        job_manager.fail_job(job_id, str(Exception))

    finally:
        for path in audio_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                logger.warning("임시 파일 삭제 실패: %s", path)


@router_v1.post("/recognize", response_model=Dict[str, str])
async def recognize_speakers(
    background_tasks: BackgroundTasks,
    audio_files: List[UploadFile] = File(
        ..., description="다채널 마이크 녹음 WAV 파일들 (2개 이상)"
    ),
    speakers_dir: str = Form(
        default="",
        description="등록 화자 음성이 있는 디렉토리 경로 (비어있으면 기본 input_Data 사용)",
    ),
    threshold: float = Form(
        default=0.2,
        description="ERes2Net 화자 매칭 임계값",
    ),
) -> Dict[str, str]:
    """다채널 마이크 파일을 업로드하면 화자별 발화 구간을 식별합니다.

    등록 화자 디렉토리 구조:
        speakers_dir/
        ├── 화자A/
        │   ├── sample1.wav
        │   └── sample2.wav
        └── 화자B/
            └── sample1.wav
    """
    if len(audio_files) < 2:
        raise HTTPException(
            status_code=400,
            detail="최소 2개 이상의 마이크 녹음 파일이 필요합니다.",
        )

    target_speakers_dir = speakers_dir.strip() or os.getenv(
        "INPUT_DATA_PATH", DEFAULT_INPUT_DATA_DIR
    )
    if not os.path.exists(target_speakers_dir):
        raise HTTPException(
            status_code=400,
            detail=f"등록 화자 디렉토리가 존재하지 않습니다: {target_speakers_dir}",
        )

    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id)

    temp_audio_paths: List[str] = []
    try:
        for upload_file in audio_files:
            temp_path = f"/tmp/{job_id}_{upload_file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
            temp_audio_paths.append(temp_path)
    except Exception as exc:
        for p in temp_audio_paths:
            if os.path.exists(p):
                os.remove(p)
        job_manager.fail_job(job_id, f"파일 업로드 실패: {exc}")
        raise HTTPException(status_code=500, detail="파일 업로드 실패")

    background_tasks.add_task(
        _background_process,
        job_id,
        temp_audio_paths,
        target_speakers_dir,
        threshold,
    )

    return {"job_id": job_id, "status": "pending"}


@router_v1.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str) -> JobInfo:
    """작업 진행 상태 및 결과를 조회합니다."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job을 찾을 수 없습니다.")
    return job
