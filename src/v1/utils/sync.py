"""다채널 마이크 오디오 동기화 및 RMS 기반 화자 분리 유틸리티.

기존 mic_speech_recognize.py의 핵심 로직을 모듈화한 것으로,
상호 상관 분석을 통한 타임라인 동기화와 Majority Voting 기반 화자 분리를 제공합니다.
"""

import logging
from typing import List, Tuple

import librosa
import numpy as np
from scipy import signal, stats

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SYNC_SAMPLE_RATE = 1000
SYNC_DURATION_SEC = 240
CHUNK_DURATION_SEC = 0.1
VOTING_WINDOW_SIZE = 15


def _majority_vote_filter(data: np.ndarray, window_size: int) -> np.ndarray:
    """슬라이딩 윈도우 최빈값 필터로 튀는 화자 판별을 보정한다."""
    output = np.zeros_like(data)
    half = window_size // 2
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        window = data[start:end]
        mode_val = stats.mode(window, keepdims=True).mode[0]
        output[i] = mode_val
    return output


def _compute_lag(
    ref_signal: np.ndarray,
    target_signal: np.ndarray,
    sample_rate: int,
    limit_sec: float,
) -> float:
    """상호 상관 분석으로 두 신호 간 지연 시간(초)을 계산한다."""
    n_samples = min(len(ref_signal), len(target_signal), int(limit_sec * sample_rate))
    ref_slice = ref_signal[:n_samples]
    target_slice = target_signal[:n_samples]

    correlation = signal.correlate(ref_slice, target_slice, mode="full")
    lags = signal.correlation_lags(len(ref_slice), len(target_slice), mode="full")

    lag_idx = int(np.argmax(correlation))
    return float(lags[lag_idx]) / sample_rate


def synchronize_audios(
    audio_signals: List[np.ndarray],
    sample_rate: int = SAMPLE_RATE,
    sync_sr: int = SYNC_SAMPLE_RATE,
    sync_duration: float = SYNC_DURATION_SEC,
) -> Tuple[np.ndarray, List[float]]:
    """여러 채널의 오디오를 시간 축 기준으로 정렬한다.

    Args:
        audio_signals: 각 채널의 오디오 신호 리스트 (모두 동일 sample_rate).
        sample_rate: 원본 오디오 샘플 레이트.
        sync_sr: 동기화 계산에 사용할 저해상도 샘플 레이트.
        sync_duration: 동기화 스캔 대상 앞부분 시간(초).

    Returns:
        (정렬된 오디오 스택 (N_channels, Samples), 각 채널의 lag 리스트)
    """
    sync_signals = [
        librosa.resample(y, orig_sr=sample_rate, target_sr=sync_sr) for y in audio_signals
    ]

    ref_sync = sync_signals[0]
    relative_lags = [0.0]
    for i in range(1, len(sync_signals)):
        lag = _compute_lag(ref_sync, sync_signals[i], sync_sr, sync_duration)
        relative_lags.append(lag)
        logger.info("채널 %d: 기준 대비 %.2f초 차이", i, lag)

    max_lag = max(relative_lags)
    aligned: List[np.ndarray] = []
    for i, y in enumerate(audio_signals):
        cut_samples = int((max_lag - relative_lags[i]) * sample_rate)
        if 0 < cut_samples < len(y):
            aligned.append(y[cut_samples:])
        else:
            aligned.append(y)

    max_len = max(len(y) for y in aligned)
    stack = np.array([np.pad(y, (0, max_len - len(y))) for y in aligned])
    return stack, relative_lags


def detect_speaker_segments(
    aligned_stack: np.ndarray,
    file_names: List[str],
    sample_rate: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION_SEC,
    voting_size: int = VOTING_WINDOW_SIZE,
) -> List[dict]:
    """RMS 에너지 + Majority Voting으로 각 시점의 주 화자를 판별한다.

    Args:
        aligned_stack: 동기화된 오디오 스택 (N_channels, Samples).
        file_names: 각 채널에 대응되는 파일/화자 이름.
        sample_rate: 샘플 레이트.
        chunk_duration: 화자 식별 단위(초).
        voting_size: 스무딩 윈도우 크기.

    Returns:
        화자별 발화 구간 리스트.
        [{"speaker": "이름", "start": 0.0, "end": 1.5}, ...]
    """
    chunk_samples = int(chunk_duration * sample_rate)
    num_chunks = aligned_stack.shape[1] // chunk_samples

    raw_winners = np.zeros(num_chunks, dtype=int)
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        rms_values = np.sqrt(np.mean(aligned_stack[:, start:end] ** 2, axis=1))
        raw_winners[i] = int(np.argmax(rms_values))

    smoothed = _majority_vote_filter(raw_winners, window_size=voting_size)

    segments: List[dict] = []
    current_speaker_idx = int(smoothed[0])
    segment_start = 0.0

    for i in range(1, len(smoothed)):
        if int(smoothed[i]) != current_speaker_idx:
            segments.append({
                "speaker": file_names[current_speaker_idx],
                "start": round(segment_start, 3),
                "end": round(i * chunk_duration, 3),
            })
            current_speaker_idx = int(smoothed[i])
            segment_start = i * chunk_duration

    segments.append({
        "speaker": file_names[current_speaker_idx],
        "start": round(segment_start, 3),
        "end": round(len(smoothed) * chunk_duration, 3),
    })

    return segments
