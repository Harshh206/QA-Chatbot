from langchain_core.documents import Document
from typing import List, Optional, Dict, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
    """Base class for document reranking"""

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on query"""
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Reranker using cross-encoder model"""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs
    ):
        """
        Initialize cross-encoder reranker

        Args:
            model_name: Name of the cross-encoder model
            **kwargs: Additional arguments (ignored)
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Using fallback reranking."
            )
            self.model = None
        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {e}")
            self.model = None

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using cross-encoder"""
        if not documents:
            return []

        if self.model is None:
            logger.warning(
                "Cross-encoder model not available, returning original order"
            )
            return documents

        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, doc.page_content] for doc in documents]

            # Get scores
            scores = self.model.predict(pairs)

            # Sort documents by score
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return [doc for doc, _ in scored_docs]
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            return documents


class LLMReranker(Reranker):
    """Reranker using LLM"""

    def __init__(self, llm=None, prompt_template: Optional[str] = None, **kwargs):
        """
        Initialize LLM reranker

        Args:
            llm: LLM instance (required)
            prompt_template: Custom prompt template
            **kwargs: Additional arguments (ignored)
        """
        self.llm = llm
        self.prompt_template = prompt_template or self._default_prompt()

        if self.llm is None:
            logger.warning(
                "No LLM provided for LLMReranker. Will return original order."
            )

    def _default_prompt(self) -> str:
        return """Rate the relevance of the following document to the query on a scale of 1-10.
Only respond with the number.

Query: {query}
Document: {document}

Rating: """

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank using LLM"""
        if not documents:
            return []

        if self.llm is None:
            logger.warning("LLM not available for reranking, returning original order")
            return documents

        try:
            scored_docs = []

            # Limit documents to avoid excessive API calls
            max_docs = min(len(documents), 10)

            for i, doc in enumerate(documents[:max_docs]):
                prompt = self.prompt_template.format(
                    query=query, document=doc.page_content[:500]  # Truncate for LLM
                )

                try:
                    # Get rating from LLM
                    response = self.llm.invoke(prompt)
                    rating_text = (
                        response.content.strip()
                        if hasattr(response, "content")
                        else str(response).strip()
                    )

                    # Extract number from response
                    import re

                    numbers = re.findall(r"\d+", rating_text)
                    rating = float(numbers[0]) if numbers else 5.0
                    rating = min(max(rating, 1), 10)  # Clamp between 1-10

                    scored_docs.append((doc, rating))
                except Exception as e:
                    logger.warning(f"Error scoring document {i}: {e}")
                    scored_docs.append((doc, 5.0))  # Default middle score

            # Sort by rating
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Add remaining documents (not scored) at the end
            if len(documents) > max_docs:
                unscored = documents[max_docs:]
                scored_docs.extend([(doc, 0) for doc in unscored])

            return [doc for doc, _ in scored_docs]

        except Exception as e:
            logger.error(f"Error in LLM reranking: {e}")
            return documents


class RRFReReranker(Reranker):
    """Reciprocal Rank Fusion reranker for combining multiple result sets"""

    def __init__(self, **kwargs):
        """
        Initialize RRF reranker

        Args:
            **kwargs: Additional arguments (ignored)
        """
        pass

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        RRF reranking (assumes documents already have scores in metadata)

        Args:
            query: Search query (unused for RRF)
            documents: List of documents with scores in metadata
        """
        if not documents:
            return []

        # Collect scores from metadata
        scored_docs = []
        for i, doc in enumerate(documents):
            # Check for various score field names
            score = (
                doc.metadata.get("rrf_score")
                or doc.metadata.get("score")
                or doc.metadata.get("relevance_score")
                or doc.metadata.get("similarity_score")
            )

            if score is None:
                # If no score, use reciprocal rank based on position
                score = 1 / (i + 1)

            scored_docs.append((doc, float(score)))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs]


class EnsembleReranker(Reranker):
    """Ensemble of multiple rerankers"""

    def __init__(
        self, rerankers: List[Reranker], weights: Optional[List[float]] = None, **kwargs
    ):
        """
        Initialize ensemble reranker

        Args:
            rerankers: List of reranker instances
            weights: Optional weights for each reranker
            **kwargs: Additional arguments (ignored)
        """
        self.rerankers = rerankers
        self.weights = weights or [1.0] * len(rerankers)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank using ensemble of rerankers"""
        if not documents:
            return []

        if not self.rerankers:
            return documents

        # Score each document with each reranker
        scores = {doc: 0.0 for doc in documents}

        for reranker, weight in zip(self.rerankers, self.weights):
            try:
                reranked = reranker.rerank(query, documents)
                # Assign scores based on position
                for position, doc in enumerate(reranked):
                    # Position-based score: higher position = higher score
                    position_score = 1.0 - (position / len(reranked))
                    scores[doc] += weight * position_score
            except Exception as e:
                logger.error(f"Error in reranker {reranker.__class__.__name__}: {e}")

        # Sort by combined score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]


def get_reranker(reranker_type: str = "cross_encoder", **kwargs) -> Reranker:
    """
    Factory function for rerankers

    Args:
        reranker_type: Type of reranker ('cross_encoder', 'llm', 'rrf', 'ensemble')
        **kwargs: Additional arguments for the reranker

    Returns:
        Reranker instance
    """
    # Filter out kwargs that shouldn't be passed to reranker
    # Common pipeline kwargs that rerankers don't need
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in ["k", "retrieve_k", "strategy", "retriever_strategy", "reranker_type"]
    }

    rerankers = {
        "cross_encoder": CrossEncoderReranker,
        "llm": LLMReranker,
        "rrf": RRFReReranker,
        "ensemble": EnsembleReranker,
    }

    reranker_class = rerankers.get(reranker_type.lower(), CrossEncoderReranker)

    # Special handling for ensemble
    if reranker_type.lower() == "ensemble":
        # Expect rerankers list in kwargs
        rerankers_list = kwargs.get("rerankers", [])
        return EnsembleReranker(rerankers_list, **filtered_kwargs)

    return reranker_class(**filtered_kwargs)
