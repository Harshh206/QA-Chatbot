import logging
from typing import Any, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank retrieved documents with a cross-encoder relevance model."""

    def __init__(
        self,
        model_name: str = "qllama/bge-reranker-v2-m3",
        batch_size: int = 16,
        max_length: Optional[int] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None

        try:
            from sentence_transformers import CrossEncoder

            model_kwargs = dict(kwargs)
            if max_length is not None:
                model_kwargs["max_length"] = max_length
            self.model = CrossEncoder(model_name, **model_kwargs)
            logger.info(f"Initialized cross-encoder reranker: {model_name}")
        except Exception as exc:
            logger.error(
                f"Could not initialize cross-encoder reranker '{model_name}': {exc}"
            )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        if not documents:
            return []

        if self.model is None:
            logger.warning("Cross-encoder unavailable; returning retrieval order")
            return documents[:k] if k else documents

        pairs = [(query, document.page_content) for document in documents]

        try:
            scores = self.model.predict(
                pairs,
                batch_size=kwargs.get("batch_size", self.batch_size),
                show_progress_bar=kwargs.get("show_progress_bar", False),
            )
        except Exception as exc:
            logger.error(f"Cross-encoder reranking failed: {exc}")
            return documents[:k] if k else documents

        scored_documents = []
        for document, score in zip(documents, scores):
            metadata = dict(document.metadata or {})
            metadata["cross_encoder_score"] = float(score)
            metadata["reranker_model"] = self.model_name
            metadata["retrieval_stage"] = "cross_encoder_reranked"
            scored_documents.append(
                Document(page_content=document.page_content, metadata=metadata)
            )

        scored_documents.sort(
            key=lambda document: document.metadata.get("cross_encoder_score", 0.0),
            reverse=True,
        )
        return scored_documents[:k] if k else scored_documents


def create_reranker(reranker_type: Optional[str], **kwargs: Any):
    if reranker_type is None:
        return None

    reranker_type = reranker_type.lower()
    if reranker_type in {"cross_encoder", "cross-encoder", "bge", "bge_reranker"}:
        return CrossEncoderReranker(**kwargs)

    raise ValueError(f"Unsupported reranker type: {reranker_type}")
