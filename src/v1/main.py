"""다채널 마이크 화자 식별 엔진.

파이프라인:
  1. 다채널 WAV 로딩 → 타임라인 동기화
  2. RMS 에너지 기반 1차 화자 구간 분리
  3. ERes2Net 임베딩으로 등록 화자와 비교하여 이름 매핑
  4. 연속 구간 병합 후 최종 발화 구간 반환
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from modelscope.pipelines import pipeline as ms_pipeline

from .utils.sync import (
    SAMPLE_RATE,
    detect_speaker_segments,
    synchronize_audios,
)

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".m4a", ".mp3"}


class MicSpeakerEngine:
    """다채널 마이크 오디오에서 등록 화자별 발화 구간을 식별하는 엔진."""

    def __init__(self, model_path: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        os.environ["MS_CACHE_HOME"] = os.path.dirname(model_path)

        logger.info("Loading ERes2Net from %s on %s", model_path, self.device)
        self.sv_pipeline = ms_pipeline(
            task="speaker-verification",
            model=model_path,
            device=self.device,
        )
        logger.info("ERes2Net model loaded successfully.")

    def _ensure_mono_16k(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav

    def _extract_score(self, result: Any) -> float:
        if isinstance(result, (float, int)):
            return float(result)
        if isinstance(result, dict):
            for key in ("score", "scores", "similarity", "cosine_score"):
                if key in result:
                    val = result[key]
                    if isinstance(val, (float, int)):
                        return float(val)
                    if isinstance(val, list) and val:
                        return float(val[0])
        if isinstance(result, list) and result:
            return self._extract_score(result[0])
        return 0.0

    def _load_enrollment_speakers(
        self, speakers_dir: str
    ) -> Tuple[Dict[str, List[Path]], Path]:
        """등록 화자 디렉토리를 읽어 {화자명: [16k mono wav 경로]} 맵을 만든다."""
        speakers_path = Path(speakers_dir)
        temp_dir = Path(f"/tmp/enroll_{int(time.time())}_{os.getpid()}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        enroll_data: Dict[str, List[Path]] = {}

        if not speakers_path.exists():
            raise FileNotFoundError(f"등록 화자 디렉토리가 존재하지 않습니다: {speakers_dir}")

        # 하위 디렉토리 = 화자 1명
        for spk_dir in sorted(p for p in speakers_path.iterdir() if p.is_dir()):
            spk_name = spk_dir.name
            refs: List[Path] = []
            for audio_file in spk_dir.iterdir():
                if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                try:
                    wav, sr = torchaudio.load(str(audio_file))
                    wav = self._ensure_mono_16k(wav, sr)
                    tmp_path = temp_dir / f"{spk_name}_{audio_file.stem}.wav"
                    torchaudio.save(str(tmp_path), wav, 16000)
                    refs.append(tmp_path)
                except Exception:
                    logger.exception("등록 파일 처리 실패: %s", audio_file)
            if refs:
                enroll_data[spk_name] = refs

        # 디렉토리 없이 루트에 바로 놓인 파일 → 파일명이 화자명
        for audio_file in speakers_path.iterdir():
            if audio_file.is_dir() or audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            spk_name = audio_file.stem
            if spk_name in enroll_data:
                continue
            try:
                wav, sr = torchaudio.load(str(audio_file))
                wav = self._ensure_mono_16k(wav, sr)
                tmp_path = temp_dir / f"direct_{spk_name}.wav"
                torchaudio.save(str(tmp_path), wav, 16000)
                enroll_data[spk_name] = [tmp_path]
            except Exception:
                logger.exception("등록 파일 처리 실패: %s", audio_file)

        if not enroll_data:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"등록 화자 음성 파일을 찾을 수 없습니다: {speakers_dir}")

        logger.info("등록 화자 %d명 로드 완료: %s", len(enroll_data), list(enroll_data.keys()))
        return enroll_data, temp_dir

    def _match_segments_to_speakers(
        self,
        rms_segments: List[dict],
        aligned_stack: np.ndarray,
        channel_names: List[str],
        enroll_data: Dict[str, List[Path]],
        threshold: float,
        progress_callback: Optional[Callable[[float], None]],
    ) -> List[dict]:
        """RMS 구간마다 ERes2Net으로 등록 화자와 비교하여 이름을 매핑한다."""
        temp_seg_path = f"/tmp/seg_{int(time.time())}_{os.getpid()}.wav"
        total = len(rms_segments)
        results: List[dict] = []

        try:
            for idx, seg in enumerate(rms_segments):
                start_sec = seg["start"]
                end_sec = seg["end"]
                rms_channel = seg["speaker"]

                channel_idx = channel_names.index(rms_channel)
                s_sample = int(start_sec * SAMPLE_RATE)
                e_sample = min(int(end_sec * SAMPLE_RATE), aligned_stack.shape[1])

                if e_sample <= s_sample:
                    continue

                chunk_audio = aligned_stack[channel_idx, s_sample:e_sample]

                duration = len(chunk_audio) / SAMPLE_RATE
                if duration < 0.5:
                    repeat_count = int(0.5 / duration) + 1
                    chunk_audio = np.tile(chunk_audio, repeat_count)

                sf.write(temp_seg_path, chunk_audio, SAMPLE_RATE)

                best_speaker = "unknown"
                best_score = -1.0

                for spk_name, ref_paths in enroll_data.items():
                    spk_best = -1.0
                    for ref_path in ref_paths:
                        try:
                            result = self.sv_pipeline([temp_seg_path, str(ref_path)])
                            score = self._extract_score(result)
                            spk_best = max(spk_best, score)
                        except Exception:
                            logger.exception("비교 실패: %s vs %s", temp_seg_path, ref_path)
                    if spk_best > best_score:
                        best_score = spk_best
                        best_speaker = spk_name

                assigned = best_speaker if best_score >= threshold else rms_channel
                results.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": assigned,
                    "score": round(float(best_score), 4) if best_score > 0 else 0.0,
                    "rms_channel": rms_channel,
                })

                if progress_callback and total > 0:
                    progress_callback(20.0 + (float(idx + 1) / total) * 79.0)
        finally:
            if os.path.exists(temp_seg_path):
                os.remove(temp_seg_path)

        return results

    def _merge_consecutive_segments(self, segments: List[dict]) -> List[dict]:
        """동일 화자의 연속 구간을 하나로 병합한다."""
        if not segments:
            return []

        merged: List[dict] = [segments[0].copy()]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg["speaker"] == prev["speaker"]:
                prev["end"] = seg["end"]
                prev["score"] = max(prev.get("score", 0.0), seg.get("score", 0.0))
            else:
                merged.append(seg.copy())
        return merged

    def process(
        self,
        audio_paths: List[str],
        speakers_dir: str,
        threshold: float = 0.2,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """전체 파이프라인을 실행한다.

        Args:
            audio_paths: 다채널 마이크 WAV 파일 경로 리스트.
            speakers_dir: 등록 화자 음성이 있는 디렉토리 경로.
            threshold: ERes2Net 화자 매칭 임계값.
            progress_callback: 진행률(0~100) 콜백.

        Returns:
            {"status": "success", "processing_time": "...", "results": [...]}
        """
        start_time = time.time()
        if progress_callback:
            progress_callback(1.0)

        # 1) 오디오 로딩
        audios: List[np.ndarray] = []
        channel_names: List[str] = []
        for fpath in audio_paths:
            y, _ = librosa.load(fpath, sr=SAMPLE_RATE)
            audios.append(y)
            channel_names.append(Path(fpath).stem)
            logger.info("로드 완료: %s", Path(fpath).name)

        if progress_callback:
            progress_callback(5.0)

        # 2) 타임라인 동기화
        logger.info("타임라인 동기화 시작 (%d채널)", len(audios))
        aligned_stack, lags = synchronize_audios(audios)
        if progress_callback:
            progress_callback(10.0)

        # 3) RMS 기반 1차 화자 구간 분리
        logger.info("RMS 기반 화자 구간 분리 중...")
        rms_segments = detect_speaker_segments(aligned_stack, channel_names)
        if progress_callback:
            progress_callback(15.0)

        # 4) ERes2Net으로 등록 화자 매칭
        logger.info("ERes2Net 화자 매칭 시작 (등록 화자 디렉토리: %s)", speakers_dir)
        enroll_data, temp_enroll_dir = self._load_enrollment_speakers(speakers_dir)
        if progress_callback:
            progress_callback(20.0)

        try:
            matched = self._match_segments_to_speakers(
                rms_segments=rms_segments,
                aligned_stack=aligned_stack,
                channel_names=channel_names,
                enroll_data=enroll_data,
                threshold=threshold,
                progress_callback=progress_callback,
            )
        finally:
            shutil.rmtree(temp_enroll_dir, ignore_errors=True)

        # 5) 연속 구간 병합
        merged = self._merge_consecutive_segments(matched)

        elapsed = round(time.time() - start_time, 2)
        logger.info("처리 완료 (%.2f초, %d개 구간)", elapsed, len(merged))

        return {
            "status": "success",
            "processing_time": f"{elapsed}s",
            "channel_names": channel_names,
            "lags": [round(l, 3) for l in lags],
            "total_segments": len(merged),
            "results": merged,
        }


# ── 싱글톤 관리 ──

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(
        CURRENT_DIR, "..", "resoursces", "models", "iic",
        "speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
    )
)
MODEL_PATH = os.getenv("SPEAKER_MODEL_PATH", DEFAULT_MODEL_PATH)

_engine: Optional[MicSpeakerEngine] = None


def get_engine() -> MicSpeakerEngine:
    global _engine
    if _engine is None:
        _engine = MicSpeakerEngine(MODEL_PATH)
    return _engine
