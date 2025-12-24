"""YouTube Fact-Checker CLI."""

import json
import sys
from typing import List, Dict, Any

from config import Config, print_disclaimer
from transcript import extract_video_id, fetch_transcript
from segmenter import segment_transcript
from claim_extractor import extract_claims_batch, filter_and_deduplicate
from evidence_retriever import EvidenceRetriever
from claim_classifier import classify_claim


def display_results(claims: List[Dict[str, Any]]) -> None:
    """Display fact-checking results."""
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    supported = sum(1 for c in claims if c["classification"] == "SUPPORTED")
    refuted = sum(1 for c in claims if c["classification"] == "REFUTED")
    inconclusive = sum(1 for c in claims if c["classification"] == "INCONCLUSIVE")
    unverifiable = sum(1 for c in claims if c["classification"] == "UNVERIFIABLE")

    print(f"\nSummary: {supported} supported, {refuted} refuted, {inconclusive} inconclusive, {unverifiable} unverifiable")
    print("-" * 50)

    for i, claim in enumerate(claims, 1):
        ts = claim.get("timestamp", {}).get("start", 0)
        print(f"\n{i}. [{ts:.0f}s] {claim['claim']}")
        print(f"   {claim['classification']}: {claim.get('reasoning', '')}")

    print_disclaimer()


def main() -> int:
    """Main entry point."""
    print("\nYouTube Fact-Checker")
    print_disclaimer()

    url = input("\nEnter YouTube URL: ").strip()
    if not url:
        print("Error: No URL provided")
        return 1

    try:
        # Extract transcript
        print("\n[1/4] Extracting transcript...")
        video_id = extract_video_id(url)
        transcript = fetch_transcript(video_id)
        segments = segment_transcript(transcript)
        print(f"  {len(segments)} segments")

        # Extract claims
        print("\n[2/4] Extracting claims...")
        max_segments = Config.MAX_LLM_CALLS * 10
        claims = extract_claims_batch(segments[:max_segments], batch_size=10)
        claims = filter_and_deduplicate(claims)[:Config.MAX_CLAIMS]
        print(f"  {len(claims)} claims")

        if not claims:
            print("\nNo verifiable claims found.")
            return 0

        # Retrieve evidence
        print("\n[3/4] Retrieving evidence...")
        retriever = EvidenceRetriever()
        for claim in claims:
            evidence = retriever.retrieve_evidence(claim["claim"])
            claim["evidence"] = evidence
            claim["evidence_count"] = evidence["total_results"]

        # Classify claims
        print("\n[4/4] Classifying claims...")
        for i, claim in enumerate(claims, 1):
            print(f"  [{i}/{len(claims)}] {claim['claim'][:40]}...")
            result = classify_claim(claim["claim"], claim["evidence"], claim["confidence"])
            claim["classification"] = result["classification"]
            claim["reasoning"] = result.get("reasoning", "")

        # Save and display
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(claims, f, indent=2)
        print("\nResults saved to results.json")

        display_results(claims)
        return 0

    except ValueError as e:
        print(f"\nError: {e}")
        return 1
    except RuntimeError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())