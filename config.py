"""Configuration and constants for the YouTube Fact-Checker."""

import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


# --- Enums ---

class SourceAuthority(Enum):
    """Authority level of evidence sources."""
    PEER_REVIEWED = "peer_reviewed"
    ACADEMIC = "academic"
    GENERAL = "general"


class EvidenceType(Enum):
    """Type of evidence source."""
    PEER_REVIEWED_RESEARCH = "peer_reviewed_research"
    ACADEMIC_PREPRINT = "academic_preprint"
    HEALTH_GUIDANCE = "health_guidance"


class ClaimType(Enum):
    """Type of claim for routing to appropriate sources."""
    MEDICAL = "medical"
    HEALTH_GUIDANCE = "health_guidance"
    ACADEMIC = "academic"
    GENERAL = "general"


# --- Configuration ---

class Config:
    """Application configuration."""
    
    # LLM settings
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS_EXTRACTION: int = 2000
    LLM_MAX_TOKENS_CLASSIFICATION: int = 500
    
    # Free-tier limits
    MAX_LLM_CALLS: int = 2
    MAX_CLAIMS: int = 5
    API_DELAY_SECONDS: float = 1.0
    
    # Segmentation
    DEFAULT_SEGMENT_DURATION: int = 30
    MIN_SEGMENT_WORDS: int = 20
    
    # PubMed API
    PUBMED_MAX_RESULTS: int = 3
    PUBMED_TIMEOUT: int = 10
    PUBMED_RATE_LIMIT_DELAY: float = 0.4
    
    # Semantic Scholar API
    SEMANTIC_SCHOLAR_MAX_RESULTS: int = 3
    SEMANTIC_SCHOLAR_TIMEOUT: int = 15
    SEMANTIC_SCHOLAR_RATE_LIMIT_DELAY: float = 0.5
    
    # Classification labels
    CLASSIFICATION_SUPPORTED: str = "SUPPORTED"
    CLASSIFICATION_REFUTED: str = "REFUTED"
    CLASSIFICATION_INCONCLUSIVE: str = "INCONCLUSIVE"
    CLASSIFICATION_UNVERIFIABLE: str = "UNVERIFIABLE"
    CLASSIFICATION_ERROR: str = "ERROR"


# --- Helper Functions ---

def get_groq_api_key() -> str:
    """Get Groq API key from environment."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file."
        )
    return key


def get_user_agent() -> str:
    """Get User-Agent string for API requests."""
    email = os.getenv("CONTACT_EMAIL", "user@example.com")
    return f"YouTubeFactChecker/1.0 ({email})"


# --- Claim Classification ---

MEDICAL_KEYWORDS = [
    "health", "medical", "disease", "treatment", "symptom", "diagnosis",
    "medicine", "drug", "therapy", "clinical", "patient", "doctor",
    "hospital", "cancer", "diabetes", "heart", "blood", "brain",
    "vitamin", "supplement", "diet", "nutrition", "exercise", "fitness",
    "skin", "body", "muscle", "bone", "organ", "cell", "immune",
    "infection", "virus", "bacteria", "antibiotic", "vaccine", "cure"
]


def classify_claim_type(claim_text: str) -> ClaimType:
    """Classify a claim to route to appropriate sources."""
    claim_lower = claim_text.lower()
    
    # Check for medical/health keywords
    if any(kw in claim_lower for kw in MEDICAL_KEYWORDS):
        return ClaimType.MEDICAL
    
    return ClaimType.GENERAL


def is_medical_claim(claim_text: str) -> bool:
    """Check if a claim is medical/health-related."""
    return classify_claim_type(claim_text) == ClaimType.MEDICAL


# --- Warnings and Disclaimers ---

DISCLAIMER_TEXT = """
DISCLAIMER: This tool is for educational purposes only.
Results should NOT be used for medical decisions.
Always consult qualified healthcare professionals.
"""


def get_medical_warning() -> str:
    """Get warning for medical claims."""
    return "This is a health-related claim. Consult a healthcare professional."


def get_preprint_warning() -> str:
    """Get warning for preprint evidence."""
    return "Evidence includes non-peer-reviewed sources. Interpret with caution."


def print_disclaimer() -> None:
    """Print the disclaimer to console."""
    print(DISCLAIMER_TEXT)
