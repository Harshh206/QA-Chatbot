from langchain_core.documents import Document
from typing import List, Optional
from collections import Counter
import hashlib
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts and enriches document metadata"""

    def __init__(self):
        self.default_metadata = {
            "ingestion_timestamp": datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
        }

    def extract_metadata(self, document: Document) -> Document:
        """Extract metadata from document content and existing metadata"""

        # Start with existing metadata
        metadata = document.metadata.copy()

        # Add document hash for deduplication
        content_hash = hashlib.md5(document.page_content.encode()).hexdigest()
        metadata["content_hash"] = content_hash

        # Extract title if missing
        if "title" not in metadata or not metadata["title"]:
            title = self._extract_title(document.page_content)
            if title:
                metadata["title"] = title

        # Extract language (simple heuristic)
        metadata["language"] = self._detect_language(document.page_content)

        # Count stats
        metadata["word_count"] = len(document.page_content.split())
        metadata["char_count"] = len(document.page_content)

        # Extract keywords
        keywords = self._extract_keywords(document.page_content, top_n=5)
        if not keywords:
            fallback = metadata.get("title") or "text"
            keywords = [fallback.split()[0] if isinstance(fallback, str) and fallback.strip() else "text"]
        metadata["keywords"] = keywords

        # Add default metadata
        metadata.update(self.default_metadata)

        document.metadata = metadata
        return document

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from text using heuristics"""
        lines = text.split("\n")

        # Check for markdown headings
        for line in lines[:20]:  # First 20 lines
            if line.startswith("#"):
                return line.lstrip("#").strip()

        # Use first non-empty line
        for line in lines[:10]:
            if line.strip():
                return line.strip()[:100]  # Truncate

        return None

    def _detect_language(self, text: str) -> str:
        """Simple language detection (placeholder)"""
        # You can integrate with langdetect or lingua-py
        return "en"

    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract keywords using simple TF-IDF or frequency"""
        # Simplified: return most common words
        words = re.findall(r"\b\w{4,}\b", text.lower())
        from collections import Counter

        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "were",
            "which",
            "they",
            "would",
        }
        word_freq = Counter([w for w in words if w not in stopwords])
        return [word for word, _ in word_freq.most_common(top_n)]

    def batch_extract(self, documents: List[Document]) -> List[Document]:
        """Extract metadata from multiple documents"""
        return [self.extract_metadata(doc) for doc in documents]
