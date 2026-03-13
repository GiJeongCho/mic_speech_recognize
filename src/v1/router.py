"""API route definitions.

Endpoints:
  POST /v1/recognize  -- Upload multi-channel WAV files -> speaker segments
  GET  /v1/jobs/{id}  -- Poll job progress
  POST /v1/refine     -- STT + mic 화자 결과를 합쳐 Kiwi 문장 분리 + 화자 매핑
"""

import json
import logging
import os
import shutil
import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from .main import MIXED_AUDIO_DIR, get_engine
from .utils.job import JobInfo, job_manager
from .utils.refine import refine_stt_with_mic

logger = logging.getLogger(__name__)

router_v1 = APIRouter(prefix="/v1", tags=["mic-speaker"])


def _build_speaker_names(
    raw_names: Optional[str],
    file_count: int,
) -> List[str]:
    """speaker_names 문자열을 파싱하여 화자 이름 리스트를 만든다.

    - 입력이 없으면: ["1번", "2번", "3번", ...]
    - 입력이 부족하면: 부족한 부분만 번호로 채움
    """
    names: List[str] = []
    if raw_names:
        names = [n.strip() for n in raw_names.split(",") if n.strip()]

    while len(names) < file_count:
        names.append(f"{len(names) + 1}번")

    return names[:file_count]


def _background_process(
    job_id: str,
    audio_paths: List[str],
    speaker_names: List[str],
) -> None:
    """Background speaker diarization pipeline."""
    try:
        def update_progress(p: float) -> None:
            job_manager.update_progress(job_id, p)

        engine = get_engine()
        result = engine.process(
            audio_paths=audio_paths,
            job_id=job_id,
            speaker_names=speaker_names,
            progress_callback=update_progress,
        )
        job_manager.complete_job(job_id, result)
        logger.info("Job %s completed", job_id)

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        job_manager.fail_job(job_id, str(exc))

    finally:
        for path in audio_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                logger.warning("Failed to cleanup temp file: %s", path)


RECOGNIZE_OPENAPI: Dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["audio_files"],
                    "properties": {
                        "audio_files": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                            "description": "Multi-channel mic WAV files (2 or more)",
                        },
                        "speaker_names": {
                            "type": "string",
                            "description": "Comma-separated speaker names (e.g. 'Alice,Bob,Charlie'). If empty, defaults to 1, 2, 3...",
                        },
                    },
                }
            }
        },
    }
}


@router_v1.post(
    "/recognize",
    response_model=Dict[str, str],
    openapi_extra=RECOGNIZE_OPENAPI,
)
async def recognize_speakers(
    background_tasks: BackgroundTasks,
    audio_files: Annotated[List[UploadFile], File()],
    speaker_names: str = Form(default=""),
) -> Dict[str, str]:
    """Upload multi-channel mic recordings to identify speaker segments.

    - audio_files: WAV files, one per mic channel (min 2)
    - speaker_names: comma-separated names matching file order
      (e.g. "Alice,Bob,Charlie"). If empty, uses "1번, 2번, 3번..."
    """
    if len(audio_files) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 mic recording files are required.",
        )

    names = _build_speaker_names(speaker_names, len(audio_files))

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
        job_manager.fail_job(job_id, f"File upload failed: {exc}")
        raise HTTPException(status_code=500, detail="File upload failed")

    background_tasks.add_task(
        _background_process,
        job_id,
        temp_audio_paths,
        names,
    )

    return {"job_id": job_id, "status": "pending"}


@router_v1.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str) -> JobInfo:
    """Poll job progress and results."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router_v1.get("/jobs/{job_id}/mixed-audio")
async def download_mixed_audio(job_id: str) -> FileResponse:
    """Download the mixed (combined) audio file for a completed job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    mixed_path = os.path.join(MIXED_AUDIO_DIR, f"{job_id}_mixed.wav")
    if not os.path.exists(mixed_path):
        raise HTTPException(status_code=404, detail="Mixed audio file not found")

    return FileResponse(
        path=mixed_path,
        media_type="audio/wav",
        filename=f"{job_id}_mixed.wav",
    )


@router_v1.post("/refine")
async def refine_with_mic(
    stt_json: UploadFile = File(
        ...,
        description="WhisperX STT 결과 JSON 파일 (segments + words 포함)",
    ),
    mic_output_json: UploadFile = File(
        ...,
        description="mic_speech_recognize 결과 JSON 파일 (화자 구간 포함)",
    ),
):
    """STT 결과와 mic 화자 구간을 합쳐 Kiwi 문장 분리 + 화자 매핑을 수행합니다.

    - stt_json: WhisperX `/transcribe` 결과 (segments, words 포함)
    - mic_output_json: `/v1/recognize` 결과 (results에 화자 구간 포함)

    반환값은 { start, end, text, speaker } 리스트입니다.
    """
    try:
        stt_raw = await stt_json.read()
        stt_data = json.loads(stt_raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid stt_json: {e}")

    try:
        mic_raw = await mic_output_json.read()
        mic_data = json.loads(mic_raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid mic_output_json: {e}")

    refined = refine_stt_with_mic(stt_data, mic_data)
    return {
        "status": "success",
        "count": len(refined),
        "results": refined,
    }
