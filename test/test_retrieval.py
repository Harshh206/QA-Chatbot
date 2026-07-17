from ingestion.pipeline import IngestionPipeline
from config import config


def test_retrieval():
    pipeline = IngestionPipeline(config)

    queries = [
        "Explain the difference between supervised and unsupervised machine learning?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        results = pipeline.vector_store.search_with_score(
            query=query, k=5
        )

        if not results:
            print("No results found.")
            continue

        for i, (doc, score) in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Score : {score:.4f}")
            print(f"Source: {doc.metadata.get('source')}")
            print(f"Section: {doc.metadata.get('section_title')}")
            print(doc.page_content[:300])


if __name__ == "__main__":
    test_retrieval()
