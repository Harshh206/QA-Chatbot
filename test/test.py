from ingestion.pipeline import IngestionPipeline
from ingestion.config import config
pipeline = IngestionPipeline(config)


stats = pipeline.vector_store.get_collection_stats()
print(stats)

query = "What is overfitting?"
results = pipeline.vector_store.search(query, k=3)
print(results)