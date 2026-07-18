#!/usr/bin/env python
from config import config
from .retrieval_pipeline import create_retrieval_pipeline
from ingestion.embedding import EmbeddingManager
from ingestion.vectorstore import ChromaVectorStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_retrieval():
    """Test different retrieval strategies"""

    # Initialize components
    embedding_manager = EmbeddingManager(
        model_name=config.embedding_model, base_url=config.base_url
    )

    vector_store = ChromaVectorStore(
        embedding_manager=embedding_manager,
        persist_directory=config.chroma_persist_dir,
        collection_name=config.collection_name,
    )

    # Test queries
    test_queries = [
        "What is  Machine Learning?",
        # "How does the system work?",
        # "What are the key features?",
    ]

    strategies = ["simple", "mmr", "hybrid", "multi_query"]

    for strategy in strategies:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing strategy: {strategy}")
        logger.info("=" * 50)

        # Create pipeline with this strategy
        pipeline = create_retrieval_pipeline(config, retriever_strategy=strategy, k=3)

        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            results = pipeline.retrieve(query, k=3)

            for i, doc in enumerate(results, 1):
                logger.info(f"Result {i}:")
                logger.info(f"Content: {doc.page_content[:150]}...")
                logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                logger.info(f"Metadata: {doc.metadata}")
                logger.info("-" * 30)


def test_with_reranking():
    """Test retrieval with reranking"""

    logger.info("\n" + "=" * 50)
    logger.info("Testing retrieval with reranking")
    logger.info("=" * 50)

    # Create pipeline with reranker
    pipeline = create_retrieval_pipeline(
        config, retriever_strategy="simple", reranker_type="cross_encoder", k=3
    )

    query = "What is the purpose of Machine learning?"

    logger.info(f"Query: {query}")

    # Without reranking
    logger.info("\nWithout reranking:")
    results = pipeline.retrieve(query, k=5, use_reranker=False)
    for i, doc in enumerate(results[:3], 1):
        logger.info(f"{i}. {doc.page_content[:100]}...")

    # With reranking
    logger.info("\nWith reranking:")
    results = pipeline.retrieve(query, k=3, use_reranker=True)
    for i, doc in enumerate(results, 1):
        logger.info(f"{i}. {doc.page_content[:100]}...")


if __name__ == "__main__":
    # Uncomment to run tests
    test_retrieval()
    test_with_reranking()

    logger.info(
        "Retrieval system ready. Use create_retrieval_pipeline() to get started."
    )
