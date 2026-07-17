from .pipeline import IngestionPipeline
from ..config import config
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")
    parser.add_argument("--file", type=str, help="Path to single file to process")
    parser.add_argument("--dir", type=str, help="Directory to process")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--query", type=str, help="Test query after ingestion")

    args = parser.parse_args()

    # Update config if directory provided
    if args.dir:
        config.input_dir = args.dir

    # Initialize pipeline
    pipeline = IngestionPipeline(config)

    # Process documents
    if args.file:
        result = pipeline.process_single_file(args.file)
    else:
        result = pipeline.process()

    # Print results
    print(json.dumps(result, indent=2))

    # Test query if requested
    if args.query and result["status"] == "success":
        print(f"\nTesting query: {args.query}")
        results = pipeline.vector_store.search(args.query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    main()
