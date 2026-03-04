# Mic Speech Recognize API

다채널 마이크 녹음 파일을 동기화하고, ERes2Net 모델을 활용하여 등록된 화자별 발화 구간을 식별하는 FastAPI 서비스입니다.

## 주요 기능

1. **자동 타임라인 동기화** — 상호 상관(Cross-correlation) 분석으로 녹음 시작 시점 차이를 계산하고 정렬
2. **RMS 기반 1차 화자 분리** — 0.1초 단위 에너지 비교 + Majority Voting 스무딩
3. **ERes2Net 화자 식별** — 등록 화자 음성과 임베딩 유사도 비교로 이름 매핑
4. **비동기 Job 관리** — 대용량 오디오 처리를 백그라운드로 실행, 진행률 조회 가능

## 디렉토리 구조

```
mic_speech_recognize/
├── pyproject.toml              # uv 프로젝트 설정
├── Dockerfile
├── README.md
├── mic_speech_recognize.py     # 기존 오프라인 스크립트 (참고용)
└── src/
    ├── __init__.py
    └── v1/
        ├── __init__.py
        ├── api.py              # FastAPI 앱 엔트리포인트
        ├── main.py             # MicSpeakerEngine (핵심 파이프라인)
        ├── router.py           # API 라우트
        └── utils/
            ├── __init__.py
            ├── sync.py         # 다채널 동기화 + RMS 화자 분리
            └── job.py          # 비동기 Job 관리
```

## 등록 화자 데이터 구조

`input_Data` 디렉토리에 화자별 폴더를 만들고 음성 파일을 넣으면 됩니다.

```
src/resoursces/test/input_Data/
├── 홍길동/
│   ├── sample1.wav
│   └── sample2.wav
├── 김철수/
│   └── sample1.wav
└── 이영희/
    └── meeting_voice.wav
```

또는 폴더 없이 파일을 직접 넣으면 파일명이 화자명으로 사용됩니다:

```
input_Data/
├── 홍길동.wav
├── 김철수.wav
└── 이영희.wav
```

## 설치 및 실행

### 로컬 실행

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# 서버 실행
uv run uvicorn src.v1.api:app --host 0.0.0.0 --port 8017 --reload
```

### 컨테이너 실행

```bash
# 빌드
podman build -t pps/mic_speech_recognize:v0.1.0 -f Dockerfile .

# 실행 (GPU 사용)
podman run --rm -d \
  --gpus all \
  -p 8017:8017 \
  -v ./src/resoursces/models:/app/src/resoursces/models:ro \
  -v ./src/resoursces/test/input_Data:/app/src/resoursces/test/input_Data:ro \
  pps/mic_speech_recognize:v0.1.0
```

## API 사용법

### Swagger 문서

서버 기동 후 `http://localhost:8017/docs` 에서 확인할 수 있습니다.

### 화자 식별 요청

```bash
curl -X POST http://localhost:8017/v1/recognize \
  -F "audio_files=@1번자리_수현.wav" \
  -F "audio_files=@2번자리_용범.wav" \
  -F "audio_files=@3번자리_기정.wav" \
  -F "threshold=0.2"
```

응답:
```json
{"job_id": "abc-123-...", "status": "pending"}
```

### 작업 상태 조회

```bash
curl http://localhost:8017/v1/jobs/{job_id}
```

응답 (완료 시):
```json
{
  "job_id": "abc-123-...",
  "status": "completed",
  "progress": 100.0,
  "result": {
    "status": "success",
    "processing_time": "45.2s",
    "results": [
      {"start": 0.0, "end": 3.5, "speaker": "홍길동", "score": 0.85},
      {"start": 3.5, "end": 7.2, "speaker": "김철수", "score": 0.91}
    ]
  }
}
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `SPEAKER_MODEL_PATH` | `src/resoursces/models/iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k` | ERes2Net 모델 경로 |
| `INPUT_DATA_PATH` | `src/resoursces/test/input_Data` | 등록 화자 음성 디렉토리 |

## Health Check

```bash
curl http://localhost:8017/health
# {"status": "healthy"}
```
