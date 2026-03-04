"""비동기 작업 상태를 관리하는 In-Memory Job Store."""

import threading
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobManager:
    """Thread-safe In-Memory Job 저장소.

    Production 환경에서는 Redis 등으로 교체해야 한다.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, JobInfo] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str) -> JobInfo:
        with self._lock:
            job = JobInfo(job_id=job_id)
            self._jobs[job_id] = job
            return job

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(self, job_id: str, progress: float) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.PROCESSING
            job.progress = min(max(progress, 0.0), 100.0)
            if job.started_at is None:
                job.started_at = datetime.now()

    def complete_job(self, job_id: str, result: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = datetime.now()
            job.result = result

    def fail_job(self, job_id: str, error_msg: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error = error_msg


job_manager = JobManager()
