"""Transcript extraction from YouTube videos."""

import re
from typing import List, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:
    """Extract video ID from a YouTube URL."""
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    url = url.strip()
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL format")


def fetch_transcript(video_id: str) -> List[Dict[str, Any]]:
    """Fetch transcript for a YouTube video."""
    if not video_id or not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        raise ValueError(f"Invalid video ID: {video_id}")

    try:
        raw = YouTubeTranscriptApi().fetch(video_id)
        transcript = [
            {
                "start": float(entry.start),
                "duration": float(entry.duration),
                "text": str(entry.text).strip()
            }
            for entry in raw if entry.text
        ]

        if not transcript:
            raise RuntimeError("Transcript is empty")

        return transcript

    except Exception as e:
        error_msg = str(e).lower()
        if "disabled" in error_msg:
            raise RuntimeError("Subtitles are disabled for this video")
        elif "not found" in error_msg or "unavailable" in error_msg:
            raise RuntimeError("Video not found or unavailable")
        elif "no transcript" in error_msg:
            raise RuntimeError("No transcript available")
        else:
            raise RuntimeError(f"Failed to fetch transcript: {e}")
