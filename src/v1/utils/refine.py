"""STT(WhisperX) 결과를 Kiwi 문장 분리 + mic 화자 매핑으로 정제한다.

입력:
  - stt_json: WhisperX가 반환한 Job 결과 JSON (segments + words)
  - mic_output_json: mic_speech_recognize가 반환한 화자 구간 JSON (results)

파이프라인:
  1. STT segments에서 word 단위 추출
  2. 각 word의 midpoint로 mic 화자 구간 매핑 → speaker 결정
  3. 동일 화자 연속 단어를 그룹핑
  4. 그룹별 Kiwi 문장 분리
  5. 0.2초 이하 짧은 구간 병합
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Optional, Tuple, TypedDict

from .kr_tag import kiwi_tagger

logger = logging.getLogger(__name__)

MIN_SPEAKER_DURATION: Final[float] = 0.5
SPEAKER_VERY_SHORT: Final[str] = "very_short"
SPEAKER_UNKNOWN: Final[str] = "unknown"


class RefinedChunk(TypedDict):
    start: float
    end: float
    text: str
    speaker: str


@dataclass
class WordItem:
    start: float
    end: float
    text: str
    speaker: str


@dataclass
class SpeakerGroup:
    speaker: str
    words: List[WordItem]


@dataclass
class ChunkInternal:
    start: float
    end: float
    text: str
    speaker: str


def extract_stt_segments(stt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """STT JSON에서 segments를 꺼낸다. (최상위 또는 result 하위)"""
    segs = stt_data.get("segments") or stt_data.get("chunks")
    if segs is None and "result" in stt_data:
        result_payload = stt_data["result"]
        if isinstance(result_payload, dict):
            segs = result_payload.get("segments") or result_payload.get("chunks")
    return segs or []


def extract_mic_results(mic_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """mic_output JSON에서 화자 구간 리스트를 꺼낸다."""
    if isinstance(mic_data, list):
        return mic_data
    results = mic_data.get("results")
    if results is None and "result" in mic_data:
        result_payload = mic_data["result"]
        if isinstance(result_payload, dict):
            results = result_payload.get("results")
    return results or []


def _find_mic_speaker(midpoint: float, mic_results: List[Dict[str, Any]]) -> Optional[str]:
    """midpoint가 속하는 mic 화자 구간을 찾아 speaker를 반환한다."""
    for seg in mic_results:
        if seg["start"] <= midpoint < seg["end"]:
            return seg["speaker"]
    return None


def _extract_words(
    raw_segments: List[Dict[str, Any]],
    mic_results: List[Dict[str, Any]],
) -> List[WordItem]:
    """STT segments에서 word 단위를 추출하고, mic 결과로 화자를 결정한다."""
    all_words: List[WordItem] = []

    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue

        words = seg.get("words", [])
        if not words:
            seg_text = str(seg.get("text", "")).strip()
            if not seg_text:
                continue
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            mid = (start + end) / 2.0
            speaker = _find_mic_speaker(mid, mic_results) or SPEAKER_UNKNOWN
            all_words.append(WordItem(start=start, end=end, text=seg_text, speaker=speaker))
            continue

        for w in words:
            if not isinstance(w, dict):
                continue
            w_text = str(w.get("word") or w.get("text", "")).strip()
            if not w_text:
                continue
            start = float(w.get("start", 0))
            end = float(w.get("end", 0))
            mid = (start + end) / 2.0
            speaker = _find_mic_speaker(mid, mic_results) or SPEAKER_UNKNOWN
            all_words.append(WordItem(start=start, end=end, text=w_text, speaker=speaker))

    return all_words


def _group_words_by_speaker(words: List[WordItem]) -> List[SpeakerGroup]:
    groups: List[SpeakerGroup] = []
    for word in words:
        if not groups or groups[-1].speaker != word.speaker:
            groups.append(SpeakerGroup(speaker=word.speaker, words=[word]))
        else:
            groups[-1].words.append(word)
    return groups


def _build_text_and_map(
    words: List[WordItem],
) -> Tuple[str, List[Tuple[int, int, WordItem]]]:
    full_text = ""
    word_map: List[Tuple[int, int, WordItem]] = []
    for w in words:
        if not w.text:
            continue
        start_char = len(full_text)
        if full_text:
            full_text += " "
            start_char += 1
        full_text += w.text
        end_char = len(full_text)
        word_map.append((start_char, end_char, w))
    return full_text, word_map


def _split_group_by_kiwi(group: SpeakerGroup) -> List[ChunkInternal]:
    full_text, word_map = _build_text_and_map(group.words)
    if not full_text:
        return []

    try:
        kiwi_sentences = kiwi_tagger.split_into_sents(full_text)
    except Exception as e:
        logger.error(f"Kiwi split_into_sents failed: {e}")
        kiwi_sentences = []

    chunks: List[ChunkInternal] = []
    if not kiwi_sentences:
        start_t = float(group.words[0].start)
        end_t = float(group.words[-1].end)
        chunks.append(ChunkInternal(start=start_t, end=end_t, text=full_text, speaker=group.speaker))
        return chunks

    for sent in kiwi_sentences:
        sent_words = [w for s, e, w in word_map if not (e <= sent.start or s >= sent.end)]
        if not sent_words:
            continue
        start_t = float(sent_words[0].start)
        end_t = float(sent_words[-1].end)
        chunks.append(ChunkInternal(start=start_t, end=end_t, text=sent.text, speaker=group.speaker))

    return chunks


def _merge_short_segments(chunks: List[ChunkInternal]) -> List[ChunkInternal]:
    if not chunks:
        return []

    final_results: List[ChunkInternal] = []
    processing_chunks = chunks[:]
    i = 0
    n = len(processing_chunks)

    while i < n:
        curr = processing_chunks[i]
        duration = curr.end - curr.start

        if duration > 0.2:
            final_results.append(curr)
            i += 1
            continue

        prev = final_results[-1] if final_results else None
        gap_prev = float('inf')
        if prev:
            gap_prev = curr.start - prev.end

        next_chunk = None
        gap_next = float('inf')
        if i + 1 < n:
            next_chunk = processing_chunks[i + 1]
            gap_next = next_chunk.start - curr.end

        if not prev and not next_chunk:
            final_results.append(curr)
            i += 1
            continue

        if gap_prev <= gap_next:
            prev.end = curr.end
            prev.text = f"{prev.text} {curr.text}".strip()
        else:
            next_chunk.start = curr.start
            next_chunk.text = f"{curr.text} {next_chunk.text}".strip()

        i += 1

    return final_results


def refine_stt_with_mic(
    stt_data: Dict[str, Any],
    mic_data: Dict[str, Any],
) -> List[RefinedChunk]:
    """STT 결과를 mic 화자 구간으로 매핑하고 Kiwi로 문장 단위 정제한다.

    Args:
        stt_data: WhisperX Job 결과 JSON.
        mic_data: mic_speech_recognize Job 결과 JSON.

    Returns:
        정제된 (start, end, text, speaker) 리스트.
    """
    raw_segments = extract_stt_segments(stt_data)
    mic_results = extract_mic_results(mic_data)

    if not raw_segments:
        return []

    all_words = _extract_words(raw_segments, mic_results)
    if not all_words:
        return []

    speaker_groups = _group_words_by_speaker(all_words)
    split_chunks: List[ChunkInternal] = []
    for group in speaker_groups:
        split_chunks.extend(_split_group_by_kiwi(group))

    merged_chunks = _merge_short_segments(split_chunks)
    return [
        {
            "start": chunk.start,
            "end": chunk.end,
            "text": chunk.text,
            "speaker": chunk.speaker,
        }
        for chunk in merged_chunks
    ]
