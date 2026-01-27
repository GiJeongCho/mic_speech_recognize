import os
import glob
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy import stats  # 최빈값 계산을 위해 추가
import logging

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# [설정] 최적화된 파라미터
# ==========================================
WORK_DIR = "/home/pps-nipa/NIQ/fish/mic_speech_recognize/음성테스트데이터"
CHUNK_DURATION = 0.1   # 화자 식별 단위 (0.1초)
SR = 16000             # 파일 저장 샘플 레이트

# [사용자 피드백 반영] 동기화 스캔 시간
# 전체 파일을 다 보면 느리니까, 앞부분 4분(240초)만 보고 맞춥니다.
SYNC_DURATION = 240    
SYNC_SR = 1000         # 동기화 계산용 저해상도 (속도 빠름)

# [스무딩] 투표 범위 (15 = 앞뒤 1.5초 문맥 확인)
VOTING_SIZE = 15

def custom_mode_filter(data, size):
    """
    슬라이딩 윈도우를 돌며 최빈값(Majority Vote)을 찾습니다.
    """
    output = np.zeros_like(data)
    half = size // 2
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        window = data[start:end]
        # stats.mode는 (최빈값, 빈도수)를 반환하므로 [0][0]으로 값만 가져옵니다.
        mode_val = stats.mode(window, keepdims=True).mode[0]
        output[i] = mode_val
    return output

def find_lag(ref_sig, target_sig, sr, limit_sec):
    """
    앞부분 limit_sec(4분) 까지만 잘라서 지연 시간을 찾습니다.
    """
    # 비교할 샘플 수 제한
    n_samples = int(limit_sec * sr)
    
    # 너무 짧은 파일 대비
    n_samples = min(len(ref_sig), len(target_sig), n_samples)
    
    ref_slice = ref_sig[:n_samples]
    target_slice = target_sig[:n_samples]

    # 상호 상관 분석
    correlation = signal.correlate(ref_slice, target_slice, mode='full')
    lags = signal.correlation_lags(len(ref_slice), len(target_slice), mode='full')
    
    lag_idx = np.argmax(correlation)
    lag_time = lags[lag_idx] / sr
    
    return lag_time

def main():
    # 1. 파일 목록 가져오기
    file_list = sorted(glob.glob(os.path.join(WORK_DIR, "*.wav")))
    file_list = [f for f in file_list if not os.path.basename(f).startswith("processed_") and not os.path.basename(f).startswith("clean_")]

    if len(file_list) < 2:
        logger.error("분석할 wav 파일이 2개 이상 필요합니다.")
        return

    logger.info(f"총 {len(file_list)}개 파일 발견. 처리 시작...")

    # 2. 오디오 로딩 (메모리 효율을 위해 동기화용은 따로 생성)
    full_audios = []   # 16kHz 원본
    sync_audios = []   # 1kHz 동기화용 (앞부분만 쓸 예정이지만 일단 리샘플링)
    file_names = []

    for fpath in file_list:
        fname = os.path.basename(fpath)
        logger.info(f"Loading: {fname} ...")
        
        y, _ = librosa.load(fpath, sr=SR)
        full_audios.append(y)
        file_names.append(fname)
        
        # 동기화용 리샘플링 (속도 핵심)
        # 4분만 쓸 거니까 로딩할 때 잘라도 되지만, 안전하게 리샘플링
        y_sync = librosa.resample(y, orig_sr=SR, target_sr=SYNC_SR)
        sync_audios.append(y_sync)

    # 3. 타임라인 동기화 (4분 스캔)
    logger.info(f"\n--- 타임라인 동기화 (초반 {SYNC_DURATION/60}분 스캔) ---")
    
    ref_sync = sync_audios[0] # 기준 파일
    relative_lags = [0]       # 첫 파일은 0초
    
    for i in range(1, len(sync_audios)):
        # 여기서 SYNC_DURATION(240초) 만큼만 잘라서 비교함
        lag_sec = find_lag(ref_sync, sync_audios[i], SYNC_SR, SYNC_DURATION)
        relative_lags.append(lag_sec)
        logger.info(f"[{file_names[i]}] 기준 대비 {lag_sec:.2f}초 차이")

    # 가장 늦게 시작한 시점(Max Lag) 찾기
    max_lag = max(relative_lags)
    aligned_audios = []
    
    logger.info(f"가장 늦은 시작점({max_lag:.2f}s)에 맞춰 정렬합니다.")

    # 4. 동기화 적용 (잘라내기)
    for i, y in enumerate(full_audios):
        cut_seconds = max_lag - relative_lags[i]
        cut_samples = int(cut_seconds * SR)
        
        if cut_samples > 0 and cut_samples < len(y):
            aligned_audios.append(y[cut_samples:])
        else:
            aligned_audios.append(y)

    # 길이 통일 (최장 길이 기준)
    max_len = max([len(y) for y in aligned_audios])
    # 모든 파일을 최장 길이에 맞춰 뒤에 0(묵음)을 채웁니다.
    stack = np.array([np.pad(y, (0, max_len - len(y))) for y in aligned_audios]) # (N_files, Samples)

    # 5. 화자 분리 (투표 기반 스무딩)
    logger.info("\n--- 화자 분리 (Majority Voting Smoothing) ---")
    
    chunk_samples = int(CHUNK_DURATION * SR)
    num_chunks = max_len // chunk_samples
    
    # (1) 순간 승자 판별
    raw_winners = np.zeros(num_chunks, dtype=int)
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk_block = stack[:, start:end] 
        
        rms_values = np.sqrt(np.mean(chunk_block**2, axis=1))
        raw_winners[i] = np.argmax(rms_values)

    # (2) 투표 필터 (튀는 값 제거)
    logger.info(f"스무딩 적용 중 (Voting Size={VOTING_SIZE})...")
    # 직접 구현한 mode_filter를 사용하여 주변에서 가장 많이 나타나는 화자 ID로 보정합니다.
    final_winners = custom_mode_filter(raw_winners, size=VOTING_SIZE)

    # (3) 결과 생성 (내 목소리 아니면 Mute)
    processed_stack = np.zeros_like(stack)
    
    for i in range(num_chunks):
        winner_idx = final_winners[i]
        start = i * chunk_samples
        end = start + chunk_samples
        
        # 승자만 데이터 복사 (나머지는 0)
        processed_stack[winner_idx, start:end] = stack[winner_idx, start:end]

    # 6. 저장
    save_dir = os.path.join(WORK_DIR, "processed_result_final")
    os.makedirs(save_dir, exist_ok=True)
    
    for i, fname in enumerate(file_names):
        output_path = os.path.join(save_dir, f"clean_{fname}")
        sf.write(output_path, processed_stack[i], SR)
        logger.info(f"Saved: {output_path}")

    logger.info("\n작업 완료! 확인해보세요.")

if __name__ == "__main__":
    main()