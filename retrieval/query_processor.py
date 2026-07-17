from typing import List, Optional, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Process and enhance queries before retrieval"""

    def __init__(self, embedding_manager=None):
        self.embedding_manager = embedding_manager

    def preprocess_query(self, query: str) -> str:
        """Clean and normalize query"""
        # Lowercase
        query = query.lower()

        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query).strip()

        # Remove special characters (keep alphanumeric and basic punctuation)
        query = re.sub(r"[^a-zA-Z0-9\s\?\!\.]", "", query)

        return query

    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        # Simple expansion - in production, use synonym dictionaries or LLM
        expansions = []

        # Add original query
        expansions.append(query)

        # Common expansions (example)
        expansion_map = {
            "how": ["what", "why", "when"],
            "help": ["assistance", "support", "guide"],
            "problem": ["issue", "error", "bug"],
            "solution": ["fix", "workaround", "resolution"],
        }

        words = query.split()
        for word in words:
            if word in expansion_map:
                for replacement in expansion_map[word]:
                    expanded = " ".join(replacement if w == word else w for w in words)
                    expansions.append(expanded)

        return list(set(expansions))  # Remove duplicates

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query (simplified)"""
        entities = {
            "dates": [],
            "numbers": [],
            "names": [],
            "topics": [],
        }

        # Extract numbers
        numbers = re.findall(r"\b\d+\b", query)
        entities["numbers"] = numbers

        # Extract potential dates (simple pattern)
        dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", query)
        entities["dates"] = dates

        # Extract capitalized words (potential names)
        names = re.findall(r"\b[A-Z][a-z]+\b", query)
        entities["names"] = names

        return entities

    def enhance_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Enhance query with additional information"""
        enhanced = {
            "original_query": query,
            "processed_query": self.preprocess_query(query),
            "expanded_queries": self.expand_query(query),
            "entities": self.extract_entities(query),
        }

        # Add embedding if available
        if self.embedding_manager:
            try:
                embedding = self.embedding_manager.embed_query(query)
                enhanced["embedding"] = embedding
            except Exception as e:
                logger.warning(f"Could not generate embedding: {e}")

        return enhanced
