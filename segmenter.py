"""Transcript segmentation for batch processing."""

from typing import List, Dict, Any
from config import Config


def segment_transcript(
    transcript: List[Dict[str, Any]],
    max_duration: int = None,
    min_words: int = None
) -> List[Dict[str, Any]]:
    """Segment transcript into chunks for LLM processing."""
    if max_duration is None:
        max_duration = Config.DEFAULT_SEGMENT_DURATION
    if min_words is None:
        min_words = Config.MIN_SEGMENT_WORDS

    if not transcript:
        return []

    segments = []
    first = transcript[0]
    current = {
        "start": float(first.get("start", 0)),
        "end": float(first.get("start", 0)) + float(first.get("duration", 0)),
        "text": str(first.get("text", "")).strip()
    }

    for entry in transcript[1:]:
        entry_start = float(entry.get("start", 0))
        entry_duration = float(entry.get("duration", 0))
        entry_text = str(entry.get("text", "")).strip()

        segment_duration = entry_start - current["start"]
        word_count = len(current["text"].split())

        if segment_duration >= max_duration and word_count >= min_words:
            segments.append(current)
            current = {
                "start": entry_start,
                "end": entry_start + entry_duration,
                "text": entry_text
            }
        else:
            if entry_text:
                current["text"] += " " + entry_text
            current["end"] = entry_start + entry_duration

    if current["text"].strip():
        segments.append(current)

    return segments