# YouTube Fact-Checker

An educational tool that extracts factual claims from YouTube video transcripts and checks them against peer-reviewed scientific sources.

## Disclaimer

This tool is for educational purposes only. Results should not be used for medical decisions. Always consult qualified healthcare professionals.

## How It Works

1. Extracts transcript from YouTube video
2. Uses LLM (Groq) to identify factual claims
3. Searches PubMed and Semantic Scholar for relevant evidence
4. Classifies claims as SUPPORTED, REFUTED, INCONCLUSIVE, or UNVERIFIABLE

## Evidence Sources

- **PubMed**: Peer-reviewed biomedical literature (primary for medical claims)
- **Semantic Scholar**: Academic papers across all disciplines

Wikipedia and paid sources are excluded to ensure verifiable, authoritative evidence.

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Create `.env` file with your Groq API key:
```
GROQ_API_KEY=your_key_here
```

Get a free API key at https://console.groq.com

## Usage

### Web Interface
```
python app.py
```
Open http://localhost:5000 in your browser.

### Command Line
```
python main.py
```

## Project Structure

```
app.py              - Web application
main.py             - CLI entry point
config.py           - Configuration
transcript.py       - YouTube transcript extraction
segmenter.py        - Transcript segmentation
claim_extractor.py  - LLM claim extraction
evidence_retriever.py - PubMed/Semantic Scholar search
claim_classifier.py - LLM classification
```

## Limitations

- Classification based on article metadata only, not full papers
- English transcripts only
- Subject to API rate limits
- Results require human verification
- Not suitable for production use
