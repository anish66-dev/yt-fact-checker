"""LLM-based claim extraction from transcript segments."""

import json
import time
from typing import List, Dict, Any, Optional
from groq import Groq

from config import Config, get_groq_api_key


def _get_client() -> Groq:
    """Get configured Groq client."""
    return Groq(api_key=get_groq_api_key())


def _parse_response(content: str) -> List[Dict[str, Any]]:
    """Parse LLM response, handling markdown code blocks."""
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            if content.startswith("json"):
                content = content[4:]
            elif content.startswith("\n"):
                content = content[1:]
    return json.loads(content.strip())


def extract_claims_batch(
    segments: List[Dict[str, Any]],
    batch_size: int = 10,
    delay: float = 1.0,
    max_llm_calls: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Extract factual claims from transcript segments using batch processing."""
    if max_llm_calls is None:
        max_llm_calls = Config.MAX_LLM_CALLS

    if not segments:
        return []

    claims = []
    total_batches = (len(segments) + batch_size - 1) // batch_size
    batches_to_process = min(total_batches, max_llm_calls)
    llm_calls_made = 0
    client = _get_client()

    for batch_idx in range(0, len(segments), batch_size):
        if llm_calls_made >= max_llm_calls:
            break

        batch = segments[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        print(f"[INFO] Processing batch {batch_num}/{batches_to_process}...")

        # Build prompt
        segments_text = ""
        for i, seg in enumerate(batch):
            start = seg.get("start", 0)
            end = seg.get("end", start)
            text = seg.get("text", "")
            segments_text += f"\n\n--- Segment {i+1} ({start:.0f}s-{end:.0f}s) ---\n{text}"

        prompt = f"""Extract factual claims from this transcript. Ignore opinions, promotions, and questions.

SEGMENTS:{segments_text}

Return JSON array (no markdown):
[{{"segment_num": 1, "claim": "factual claim", "confidence": "high/medium/low", "timestamp_start": 0}}]

Return [] if no claims found."""

        try:
            response = client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=Config.LLM_MAX_TOKENS_EXTRACTION
            )
            llm_calls_made += 1
            result = _parse_response(response.choices[0].message.content)

            if isinstance(result, list):
                for claim_data in result:
                    if not isinstance(claim_data, dict):
                        continue
                    seg_idx = claim_data.get("segment_num", 1) - 1
                    if 0 <= seg_idx < len(batch):
                        claim_text = claim_data.get("claim", "").strip()
                        if claim_text:
                            claims.append({
                                "claim": claim_text,
                                "timestamp": {
                                    "start": batch[seg_idx].get("start", 0),
                                    "end": batch[seg_idx].get("end", 0)
                                },
                                "confidence": claim_data.get("confidence", "medium"),
                                "original_text": batch[seg_idx].get("text", "")
                            })

            if batch_num < batches_to_process:
                time.sleep(delay)

        except json.JSONDecodeError:
            time.sleep(delay)
        except Exception as e:
            print(f"[WARNING] Batch {batch_num} failed: {e}")
            time.sleep(delay * 2)

    print(f"[INFO] Extraction complete. LLM calls: {llm_calls_made}/{max_llm_calls}")
    return claims


def filter_and_deduplicate(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate claims."""
    unique = []
    seen = set()
    for claim in claims:
        text = claim.get("claim", "").lower().strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(claim)
    return unique