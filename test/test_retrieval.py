import logging
from config import config
from ingestion.embeddings import EmbeddingManager
from ingestion.vector_store import ChromaVectorStore
from retrieval.retrieval_pipeline import RetrievalPipeline, create_retrieval_pipeline
from retrieval.retriever import HybridRetriever, BM25SparseRetriever, DenseRetriever
from retrieval.reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_bm25_retriever():
    """Test BM25 sparse retriever standalone"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: BM25 Sparse Retriever")
    logger.info("=" * 60)

    embedding_manager = EmbeddingManager(
        model_name=config.embedding_model, base_url=config.base_url
    )
    vector_store = ChromaVectorStore(
        embedding_manager=embedding_manager,
        persist_directory=config.chroma_persist_dir,
        collection_name=config.collection_name,
    )

    bm25 = BM25SparseRetriever(vector_store)
    results = bm25.retrieve("What is Machine Learning?", k=5)

    for i, (doc, score) in enumerate(results, 1):
        logger.info(f"\nResult {i} | BM25 Score: {score:.4f}")
        logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"Content: {doc.page_content[:200]}...")

    return results


def test_dense_retriever():
    """Test dense vector retriever standalone"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Dense Vector Retriever (qwen3-embedding)")
    logger.info("=" * 60)

    embedding_manager = EmbeddingManager(
        model_name=config.embedding_model, base_url=config.base_url
    )
    vector_store = ChromaVectorStore(
        embedding_manager=embedding_manager,
        persist_directory=config.chroma_persist_dir,
        collection_name=config.collection_name,
    )

    dense = DenseRetriever(vector_store, embedding_manager)
    results = dense.retrieve("What is Machine Learning?", k=5)

    for i, (doc, score) in enumerate(results, 1):
        logger.info(f"\nResult {i} | Cosine Score: {score:.4f}")
        logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"Content: {doc.page_content[:200]}...")

    return results


def test_hybrid_retriever():
    """Test hybrid retriever with RRF fusion"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Hybrid Retriever (BM25 + Dense + RRF)")
    logger.info("=" * 60)

    embedding_manager = EmbeddingManager(
        model_name=config.embedding_model, base_url=config.base_url
    )
    vector_store = ChromaVectorStore(
        embedding_manager=embedding_manager,
        persist_directory=config.chroma_persist_dir,
        collection_name=config.collection_name,
    )

    hybrid = HybridRetriever(vector_store, embedding_manager)
    results = hybrid.retrieve("What is Machine Learning?", k=5)

    for i, doc in enumerate(results, 1):
        logger.info(f"\nResult {i}")
        logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"Content: {doc.page_content[:200]}...")

    return results


def test_reranker():
    """Test Ollama cross-encoder reranker"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Cross-Encoder Reranker (bge-reranker-v2-m3)")
    logger.info("=" * 60)

    from langchain_core.documents import Document

    reranker = CrossEncoderReranker(
        model_name= config.reranker_model,
        score_threshold=config.threshold,
    )

    dummy_docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "doc1"}),
        Document(page_content="The weather today is sunny and warm.", metadata={"source": "doc2"}),
        Document(page_content="Deep learning uses neural networks with many layers.", metadata={"source": "doc3"}),
    ]

    results = reranker.rerank("What is Machine Learning?", dummy_docs)

    for i, doc in enumerate(results, 1):
        logger.info(f"\nReranked {i}")
        logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"Content: {doc.page_content[:200]}...")

    return results


def test_full_pipeline():
    """Test complete retrieval pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Full Retrieval Pipeline")
    logger.info("  Query → BM25 + Dense → RRF → Cross-Encoder → Score Filter")
    logger.info("=" * 60)

    pipeline = create_retrieval_pipeline(
        config=config,
        top_k=3,
        score_threshold=0.70,
    )

    queries = [
        "What is Machine Learning?",
    ]

    for query in queries:
        logger.info(f"\nQuery: {query}")
        logger.info("-" * 40)

        results = pipeline.retrieve(query, k=3)

        if not results:
            logger.info("No results above threshold.")
            continue

        for i, doc in enumerate(results, 1):
            logger.info(f"\nFinal Result {i}")
            logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
            logger.info(f"Content: {doc.page_content[:300]}...")

    return results


if __name__ == "__main__":
    # test_bm25_retriever()
    # test_dense_retriever()
    # test_hybrid_retriever()
    test_reranker()
    # # test_full_pipeline()

    logger.info("\n" + "=" * 60)
    logger.info("All retrieval tests completed.")
    logger.info("=" * 60)
