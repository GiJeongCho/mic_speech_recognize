"""다채널 마이크 화자 분리 엔진.

파이프라인:
  1. 다채널 WAV 로딩 → 타임라인 동기화
  2. 동기화된 채널을 mix-down하여 합본 WAV 저장
  3. RMS 에너지 + Majority Voting 기반 화자 구간 분리
  4. 연속 구간 병합 후 최종 발화 구간 반환
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf

from .utils.sync import (
    SAMPLE_RATE,
    detect_speaker_segments,
    synchronize_audios,
)

logger = logging.getLogger(__name__)

MIXED_AUDIO_DIR = "/tmp/mic_speech_mixed"


class MicSpeakerEngine:
    """다채널 마이크 오디오에서 RMS 에너지 기반으로 화자별 발화 구간을 분리하는 엔진."""

    def _merge_consecutive_segments(self, segments: List[dict]) -> List[dict]:
        """동일 화자의 연속 구간을 하나로 병합한다."""
        if not segments:
            return []

        merged: List[dict] = [segments[0].copy()]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg["speaker"] == prev["speaker"]:
                prev["end"] = seg["end"]
            else:
                merged.append(seg.copy())
        return merged

    def _save_mixed_audio(
        self,
        aligned_stack: np.ndarray,
        job_id: str,
    ) -> str:
        """동기화된 전체 채널을 평균으로 합쳐 하나의 WAV로 저장한다."""
        os.makedirs(MIXED_AUDIO_DIR, exist_ok=True)
        mixed = aligned_stack.mean(axis=0)

        peak = np.abs(mixed).max()
        if peak > 0:
            mixed = mixed / peak * 0.95

        output_path = os.path.join(MIXED_AUDIO_DIR, f"{job_id}_mixed.wav")
        sf.write(output_path, mixed, SAMPLE_RATE)
        logger.info("Mixed audio saved: %s", output_path)
        return output_path

    def process(
        self,
        audio_paths: List[str],
        job_id: str,
        speaker_names: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """전체 파이프라인을 실행한다.

        Args:
            audio_paths: 다채널 마이크 WAV 파일 경로 리스트 (2개 이상).
            job_id: 합본 파일 저장에 사용할 Job ID.
            speaker_names: 각 채널에 대응하는 화자 이름.
                None이면 파일명을 화자 이름으로 사용.
            progress_callback: 진행률(0~100) 콜백.

        Returns:
            처리 결과 딕셔너리 (합본 파일 경로, 시간 정보, 화자 구간 포함).
        """
        start_time = time.time()
        if progress_callback:
            progress_callback(1.0)

        # 1) 오디오 로딩
        audios: List[np.ndarray] = []
        channel_names: List[str] = []
        for i, fpath in enumerate(audio_paths):
            y, _ = librosa.load(fpath, sr=SAMPLE_RATE)
            audios.append(y)
            name = speaker_names[i] if speaker_names and i < len(speaker_names) else Path(fpath).stem
            channel_names.append(name)
            logger.info("Loaded: %s -> %s", Path(fpath).name, name)

        if progress_callback:
            progress_callback(10.0)

        # 2) 타임라인 동기화
        logger.info("Synchronizing %d channels...", len(audios))
        aligned_stack, lags = synchronize_audios(audios)
        if progress_callback:
            progress_callback(30.0)

        # 3) 합본 WAV 저장
        logger.info("Saving mixed audio...")
        mixed_audio_path = self._save_mixed_audio(aligned_stack, job_id)
        total_duration = round(aligned_stack.shape[1] / SAMPLE_RATE, 3)
        if progress_callback:
            progress_callback(40.0)

        # 4) RMS 기반 화자 구간 분리
        logger.info("Detecting speaker segments via RMS + Majority Voting...")
        rms_segments = detect_speaker_segments(aligned_stack, channel_names)
        if progress_callback:
            progress_callback(80.0)

        # 5) 연속 구간 병합
        merged = self._merge_consecutive_segments(rms_segments)

        elapsed = round(time.time() - start_time, 2)
        logger.info("Done (%.2fs, %d segments)", elapsed, len(merged))

        return {
            "status": "success",
            "processing_time": f"{elapsed}s",
            "channel_names": channel_names,
            "lags": [round(lag, 3) for lag in lags],
            "mixed_audio_path": mixed_audio_path,
            "audio_start": 0.0,
            "audio_end": total_duration,
            "total_duration": total_duration,
            "total_segments": len(merged),
            "results": merged,
        }


_engine: Optional[MicSpeakerEngine] = None


def get_engine() -> MicSpeakerEngine:
    global _engine
    if _engine is None:
        _engine = MicSpeakerEngine()
    return _engine
