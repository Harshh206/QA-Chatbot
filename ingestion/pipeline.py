from .loader import DocumentLoader

from .processors.cleaner import TextCleaner
from .processors.metadata_extract import MetadataExtractor
from .embeddings import EmbeddingManager
from .vector_store import ChromaVectorStore
from config import config
from . import splitter
import logging
from typing import Optional
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Complete ingestion pipeline with all stages"""

    def __init__(self, config):
        self.config = config

        # Initialize components
        self.loader = DocumentLoader(config.input_dir)

        self.text_cleaner = TextCleaner(preserve_markdown=True)
        self.metadata_extractor = MetadataExtractor()
        self.chunker = splitter.AutoChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        
        self.embedding_manager = EmbeddingManager(model_name=config.embedding_model)
        self.vector_store = ChromaVectorStore(
            embedding_manager=self.embedding_manager,
            persist_directory=config.chroma_persist_dir,
            collection_name=config.collection_name,
        )

    def process(self, input_path: Optional[str] = None) -> dict:
        """Run the complete ingestion pipeline"""
        logger.info("Starting ingestion pipeline...")
        print("=" * 60)

        try:
            # 1. Load documents
            logger.info("Step 1: Loading documents")

            if input_path:
                documents = self.loader.load_single_file(Path(input_path))
            else:
                documents = self.loader.load_directory()

            if not documents:
                logger.warning("No documents loaded")
                return {"status": "failed", "reason": "No documents found"}

            logger.info(f"Loaded {len(documents)} documents")

            # 2. Clean text
            logger.info("Step 2: Cleaning text")
            documents = self.text_cleaner.batch_clean(documents)

            # 3 Extract metadata
            logger.info("Step 3: Extracting metadata")
            documents = self.metadata_extractor.batch_extract(documents)

            # 4. Chunk documents
            logger.info("Step 4: Chunking documents")
            chunks = self.chunker.split(documents)
            logger.info(f"Created {len(chunks)} chunks")

            # 5. Add embeddings
            logger.info("Step 5: Generating embeddings")
            ids = self.vector_store.add_documents(chunks)
            logger.info(f"Stored {len(ids)} chunks in vector DB")

            # 6. Get statistics
            stats = self.vector_store.get_collection_stats()

            return {
                "status": "success",
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "stored_ids": ids[:5],  # Sample IDs
                "collection_stats": stats,
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}

    def process_single_file(self, file_path: str) -> dict:
        """
        Process a single file
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary with processing results
        """
        return self.process(input_path=file_path)

    def process_directory(self, dir_path: Optional[str]) -> dict:
        """
        Process all files in a directory
        
        Args:
            dir_path: Optional directory path (uses config.input_dir if not provided)
            
        Returns:
            Dictionary with processing results
        """
        if dir_path:
            self.loader.input_dir = Path(dir_path)
        return self.process()


# Example usage
if __name__ == "__main__":
    pipeline = IngestionPipeline(config)
    result = pipeline.process()
    print(f"Pipeline completed with status: {result['status']}")
