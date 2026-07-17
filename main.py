from .ingestion.pipeline import IngestionPipeline
from .retrieval.retrieval_pipeline import create_retrieval_pipeline
from .config import config
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Document Ingestion and Retrieval Pipeline"
    )
    parser.add_argument("--file", type=str, help="Path to single file to process")
    parser.add_argument("--dir", type=str, help="Directory to process")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--query", type=str, help="Test query after ingestion")
    parser.add_argument(
        "--strategy",
        type=str,
        default="simple",
        choices=["simple", "mmr", "hybrid", "multi_query", "parent"],
        help="Retrieval strategy",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default=None,
        choices=["cross_encoder", "llm", "rrf"],
        help="Reranker type",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Skip ingestion and only run retrieval",
    )

    args = parser.parse_args()

    # Update config if directory provided
    if args.dir:
        config.input_dir = args.dir

    # Initialize pipeline
    pipeline = IngestionPipeline(config)

    # Process documents if not query-only
    if not args.query_only:
        if args.file:
            result = pipeline.process_single_file(args.file)
        else:
            result = pipeline.process()

        # Print results
        print(json.dumps(result, indent=2))

        if result["status"] != "success":
            logger.error("Ingestion failed, cannot perform retrieval")
            return

    # Test query if requested
    if args.query:
        logger.info(f"\nQuery: {args.query}")
        logger.info(f"Retrieval strategy: {args.strategy}")
        logger.info(f"Reranker: {args.reranker if args.reranker else 'None'}")
        logger.info(f"Number of results: {args.k}")

        try:
            # Create retrieval pipeline
            retrieval_pipeline = create_retrieval_pipeline(
                config,
                retriever_strategy=args.strategy,
                reranker_type=args.reranker,
                k=args.k,
            )

            # Get results
            results = retrieval_pipeline.retrieve(args.query, k=args.k)

            print("\n" + "=" * 60)
            print(f"Retrieved {len(results)} documents:")
            print("=" * 60)

            for i, doc in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Score: {doc.metadata.get('score', 'N/A')}")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Content: {doc.page_content[:300]}...")
                if len(doc.page_content) > 300:
                    print("...")
                print("-" * 40)

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
