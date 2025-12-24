"""Evidence retrieval from PubMed and Semantic Scholar."""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests

from config import (
    Config,
    get_user_agent,
    SourceAuthority,
    EvidenceType,
    ClaimType,
    classify_claim_type,
)


class EvidenceSource(ABC):
    """Abstract base class for evidence sources."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def authority(self) -> SourceAuthority:
        pass
    
    @property
    @abstractmethod
    def evidence_type(self) -> EvidenceType:
        pass
    
    @abstractmethod
    def search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class PubMedSource(EvidenceSource):
    """PubMed evidence source for peer-reviewed medical literature."""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.headers = {"User-Agent": get_user_agent()}
    
    @property
    def name(self) -> str:
        return "PubMed"
    
    @property
    def authority(self) -> SourceAuthority:
        return SourceAuthority.PEER_REVIEWED
    
    @property
    def evidence_type(self) -> EvidenceType:
        return EvidenceType.PEER_REVIEWED_RESEARCH
    
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}einfo.fcgi", params={"db": "pubmed", "retmode": "json"}, timeout=5)
            return r.status_code == 200
        except Exception:
            return False
    
    def search(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        if max_results is None:
            max_results = Config.PUBMED_MAX_RESULTS
        
        try:
            # Search for IDs
            search_r = requests.get(
                f"{self.base_url}esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
                headers=self.headers,
                timeout=Config.PUBMED_TIMEOUT
            )
            search_r.raise_for_status()
            ids = search_r.json().get("esearchresult", {}).get("idlist", [])
            
            if not ids:
                return []
            
            # Fetch summaries
            sum_r = requests.get(
                f"{self.base_url}esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
                headers=self.headers,
                timeout=Config.PUBMED_TIMEOUT
            )
            sum_r.raise_for_status()
            summaries = sum_r.json()
            
            results = []
            for pmid in ids:
                article = summaries.get("result", {}).get(pmid, {})
                if article and isinstance(article, dict):
                    authors = article.get("authors", [])
                    if isinstance(authors, list):
                        names = [a.get("name", "") for a in authors[:3] if isinstance(a, dict)]
                        authors = ", ".join(filter(None, names))
                    else:
                        authors = ""
                    
                    results.append({
                        "title": article.get("title", "Unknown"),
                        "source": self.name,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "authority": self.authority,
                        "evidence_type": self.evidence_type,
                        "is_peer_reviewed": True,
                        "publication_date": article.get("pubdate", ""),
                        "authors": authors,
                        "pmid": pmid
                    })
            
            time.sleep(Config.PUBMED_RATE_LIMIT_DELAY)
            return results
            
        except Exception as e:
            print(f"[WARNING] PubMed error: {e}")
            return []


class SemanticScholarSource(EvidenceSource):
    """Semantic Scholar source for academic papers."""
    
    _rate_limited_until: float = 0
    
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"User-Agent": get_user_agent()}
    
    @property
    def name(self) -> str:
        return "Semantic Scholar"
    
    @property
    def authority(self) -> SourceAuthority:
        return SourceAuthority.ACADEMIC
    
    @property
    def evidence_type(self) -> EvidenceType:
        return EvidenceType.ACADEMIC_PREPRINT
    
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/paper/search", params={"query": "test", "limit": 1}, timeout=5)
            return r.status_code == 200
        except Exception:
            return False
    
    def search(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        if time.time() < SemanticScholarSource._rate_limited_until:
            return []
        
        if max_results is None:
            max_results = Config.SEMANTIC_SCHOLAR_MAX_RESULTS
        
        try:
            r = requests.get(
                f"{self.base_url}/paper/search",
                params={"query": query, "limit": max_results, "fields": "title,authors,year,venue,url,publicationTypes"},
                headers=self.headers,
                timeout=Config.SEMANTIC_SCHOLAR_TIMEOUT
            )
            
            if r.status_code == 429:
                SemanticScholarSource._rate_limited_until = time.time() + 300
                print("[WARNING] Semantic Scholar rate limited")
                return []
            
            r.raise_for_status()
            papers = r.json().get("data", [])
            
            results = []
            for paper in papers:
                if not paper:
                    continue
                
                authors = paper.get("authors", [])
                if isinstance(authors, list):
                    names = [a.get("name", "") for a in authors[:3] if isinstance(a, dict)]
                    authors = ", ".join(filter(None, names))
                else:
                    authors = ""
                
                pub_types = paper.get("publicationTypes") or []
                is_pr = bool(paper.get("venue") and "Preprint" not in pub_types)
                
                results.append({
                    "title": paper.get("title", "Unknown"),
                    "source": self.name,
                    "url": paper.get("url") or f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}",
                    "authority": SourceAuthority.PEER_REVIEWED if is_pr else SourceAuthority.ACADEMIC,
                    "evidence_type": EvidenceType.PEER_REVIEWED_RESEARCH if is_pr else EvidenceType.ACADEMIC_PREPRINT,
                    "is_peer_reviewed": is_pr,
                    "publication_date": str(paper.get("year", "")),
                    "authors": authors,
                    "venue": paper.get("venue", "")
                })
            
            time.sleep(Config.SEMANTIC_SCHOLAR_RATE_LIMIT_DELAY)
            return results
            
        except Exception as e:
            print(f"[WARNING] Semantic Scholar error: {e}")
            return []


class EvidenceRetriever:
    """Routes claims to appropriate evidence sources."""
    
    def __init__(self):
        self.pubmed = PubMedSource()
        self.semantic_scholar = SemanticScholarSource()
        self._routing = {
            ClaimType.MEDICAL: [self.pubmed, self.semantic_scholar],
            ClaimType.HEALTH_GUIDANCE: [self.pubmed],
            ClaimType.ACADEMIC: [self.semantic_scholar, self.pubmed],
            ClaimType.GENERAL: [self.semantic_scholar, self.pubmed],
        }
    
    def retrieve_evidence(self, claim_text: str, claim_type: ClaimType = None) -> Dict[str, Any]:
        """Retrieve evidence from appropriate sources for a claim."""
        if claim_type is None:
            claim_type = classify_claim_type(claim_text)
        
        preview = claim_text[:60] + "..." if len(claim_text) > 60 else claim_text
        print(f"  Searching: {preview}")
        print(f"    Type: {claim_type.value}")
        
        sources = self._routing.get(claim_type, [self.pubmed])
        
        evidence = {
            "claim_type": claim_type.value,
            "sources_queried": [],
            "pubmed": [],
            "semantic_scholar": [],
            "total_results": 0,
            "has_peer_reviewed": False,
            "primary_source": sources[0].name if sources else None
        }
        
        for source in sources:
            results = source.search(claim_text)
            
            # Convert enums to strings for JSON
            serializable = []
            for r in results:
                sr = r.copy()
                if hasattr(sr.get("authority"), "value"):
                    sr["authority"] = sr["authority"].value
                if hasattr(sr.get("evidence_type"), "value"):
                    sr["evidence_type"] = sr["evidence_type"].value
                serializable.append(sr)
            
            evidence["sources_queried"].append(source.name)
            
            if isinstance(source, PubMedSource):
                evidence["pubmed"] = serializable
            elif isinstance(source, SemanticScholarSource):
                evidence["semantic_scholar"] = serializable
            
            if any(r.get("is_peer_reviewed") for r in results):
                evidence["has_peer_reviewed"] = True
        
        evidence["total_results"] = len(evidence["pubmed"]) + len(evidence["semantic_scholar"])
        print(f"    Found {evidence['total_results']} sources ({len(evidence['pubmed'])} PubMed, {len(evidence['semantic_scholar'])} Semantic Scholar)")
        
        return evidence