"""YouTube Fact-Checker Web Application."""

from flask import Flask, render_template, request, jsonify
from typing import Dict, Any, List
import time

from config import Config, DISCLAIMER_TEXT, is_medical_claim, get_medical_warning
from transcript import extract_video_id, fetch_transcript
from segmenter import segment_transcript
from claim_extractor import extract_claims_batch, filter_and_deduplicate
from evidence_retriever import EvidenceRetriever
from claim_classifier import classify_claim

app = Flask(__name__)


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Process a YouTube video and return fact-checking results."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        url = data.get('url', '').strip()
        if not url:
            return jsonify({"error": "Please provide a YouTube URL"}), 400

        # Extract transcript
        print(f"[INFO] Processing: {url}")
        try:
            video_id = extract_video_id(url)
            transcript = fetch_transcript(video_id)
        except (ValueError, RuntimeError) as e:
            return jsonify({"error": str(e)}), 400

        # Segment and extract claims
        segments = segment_transcript(transcript)
        print(f"[INFO] {len(segments)} segments")

        max_segments = Config.MAX_LLM_CALLS * 10
        claims = extract_claims_batch(segments[:max_segments], batch_size=10)
        claims = filter_and_deduplicate(claims)[:Config.MAX_CLAIMS]
        print(f"[INFO] {len(claims)} claims")

        if not claims:
            return jsonify({
                "success": True,
                "video_id": video_id,
                "total_segments": len(segments),
                "total_claims": 0,
                "results": [],
                "disclaimer": DISCLAIMER_TEXT,
                "message": "No verifiable claims found."
            })

        # Retrieve evidence and classify
        retriever = EvidenceRetriever()
        results: List[Dict[str, Any]] = []

        for i, claim in enumerate(claims, 1):
            print(f"[{i}/{len(claims)}] {claim['claim'][:40]}...")

            evidence = retriever.retrieve_evidence(claim['claim'])
            classification = classify_claim(claim['claim'], evidence, claim.get('confidence', 'medium'))

            result = {
                "claim": claim['claim'],
                "timestamp": claim.get('timestamp', {'start': 0, 'end': 0}),
                "classification": classification['classification'],
                "reasoning": classification.get('reasoning', ''),
                "evidence": evidence,
                "evidence_count": evidence["total_results"]
            }

            if is_medical_claim(claim['claim']):
                result["medical_warning"] = get_medical_warning()

            results.append(result)

            if i < len(claims):
                time.sleep(Config.API_DELAY_SECONDS)

        return jsonify({
            "success": True,
            "video_id": video_id,
            "total_segments": len(segments),
            "total_claims": len(claims),
            "results": results,
            "disclaimer": DISCLAIMER_TEXT
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    print("\nYouTube Fact-Checker - http://localhost:5000")
    print("For educational purposes only.\n")
    app.run(debug=True, port=5000)
