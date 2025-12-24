"""LLM-based claim classification against scientific evidence."""

import json
from typing import Dict, Any, List
from groq import Groq

from config import (
    Config,
    get_groq_api_key,
    is_medical_claim,
    get_medical_warning,
    get_preprint_warning,
)


def _get_client() -> Groq:
    """Get configured Groq client."""
    return Groq(api_key=get_groq_api_key())


def _format_evidence(evidence: Dict[str, Any]) -> str:
    """Format evidence sources for the LLM prompt."""
    text = ""
    
    for source in evidence.get("pubmed", []):
        title = source.get("title", "Unknown")
        date = source.get("publication_date", source.get("pubdate", ""))
        url = source.get("url", "")
        text += f"\n- PubMed [Peer-Reviewed]: {title} ({date})\n  {url}"

    for source in evidence.get("semantic_scholar", []):
        title = source.get("title", "Unknown")
        date = source.get("publication_date", "")
        url = source.get("url", "")
        is_pr = source.get("is_peer_reviewed", False)
        status = "[Peer-Reviewed]" if is_pr else "[Preprint]"
        text += f"\n- Academic {status}: {title} ({date})\n  {url}"

    return text.strip()


def _parse_response(content: str) -> Dict[str, Any]:
    """Parse LLM response, handling markdown code blocks."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if len(lines) >= 2:
            content = "\n".join(lines[1:])
            if content.endswith("```"):
                content = content[:-3]
    return json.loads(content.strip())


def _normalize_classification(raw: str) -> str:
    """Normalize classification to canonical labels."""
    lower = raw.lower().strip()
    if "support" in lower:
        return Config.CLASSIFICATION_SUPPORTED
    elif "refut" in lower or "contradict" in lower:
        return Config.CLASSIFICATION_REFUTED
    elif "insufficient" in lower or "inconclusive" in lower:
        return Config.CLASSIFICATION_INCONCLUSIVE
    elif "unverif" in lower or "cannot" in lower:
        return Config.CLASSIFICATION_UNVERIFIABLE
    return Config.CLASSIFICATION_INCONCLUSIVE


def _has_peer_reviewed(evidence: Dict[str, Any]) -> bool:
    """Check if any evidence is peer-reviewed."""
    if evidence.get("pubmed"):
        return True
    for s in evidence.get("semantic_scholar", []):
        if s.get("is_peer_reviewed"):
            return True
    return evidence.get("has_peer_reviewed", False)


def _has_preprints(evidence: Dict[str, Any]) -> bool:
    """Check if evidence includes preprints."""
    for s in evidence.get("semantic_scholar", []):
        if not s.get("is_peer_reviewed", False):
            return True
    return False


def classify_claim(
    claim_text: str,
    evidence: Dict[str, Any],
    confidence: str
) -> Dict[str, Any]:
    """Classify a claim based on available evidence."""
    evidence_text = _format_evidence(evidence)
    has_pr = _has_peer_reviewed(evidence)
    has_preprint = _has_preprints(evidence)

    if not evidence_text:
        result = {
            "classification": Config.CLASSIFICATION_UNVERIFIABLE,
            "reasoning": "No scientific sources found.",
            "confidence": "low",
            "evidence_quality": "none"
        }
        if is_medical_claim(claim_text):
            result["medical_warning"] = get_medical_warning()
        return result

    prompt = f"""Classify this claim against the evidence.

CLAIM: "{claim_text}"

EVIDENCE:{evidence_text}

Weight peer-reviewed sources higher than preprints.

Return JSON only:
{{"classification": "SUPPORTED/REFUTED/INCONCLUSIVE/UNVERIFIABLE", "reasoning": "brief explanation", "confidence": "high/medium/low"}}"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS_CLASSIFICATION
        )

        result = _parse_response(response.choices[0].message.content)
        result["classification"] = _normalize_classification(result.get("classification", ""))
        
        if has_pr:
            result["evidence_quality"] = "peer_reviewed"
        elif has_preprint:
            result["evidence_quality"] = "preprint_only"
        else:
            result["evidence_quality"] = "limited"

        if is_medical_claim(claim_text):
            result["medical_warning"] = get_medical_warning()
        
        if has_preprint and not has_pr:
            result["preprint_warning"] = get_preprint_warning()

        return result

    except Exception as e:
        return {
            "classification": Config.CLASSIFICATION_ERROR,
            "reasoning": f"Classification error: {e}",
            "confidence": "low",
            "evidence_quality": "error"
        }