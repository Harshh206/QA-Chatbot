from .loader import DocumentLoader
from .processors.markdown import MarkdownConverter
from .processors.cleaner import TextCleaner
from .processors.metadata_extract import MetadataExtractor
from .embedding import EmbeddingManager
from .vectorstore import ChromaVectorStore
from config import config
from . import chunker
import logging
from typing import List, Optional
from pathlib import Path
import uuid


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Complete ingestion pipeline with all stages"""

    def __init__(self, config):
        self.config = config

        # Initialize components
        self.loader = DocumentLoader(config.input_dir)
        self.markdown_converter = MarkdownConverter()
        self.text_cleaner = TextCleaner(preserve_markdown=True)
        self.metadata_extractor = MetadataExtractor()
        self.chunker = chunker.get_chunker(
            strategy="hybrid",
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

            # 2. Convert to Markdown
            logger.info("Step 2: Converting to Markdown")
            documents = self.markdown_converter.process_documents(documents)

            # 3. Clean text
            logger.info("Step 3: Cleaning text")
            documents = self.text_cleaner.batch_clean(documents)

            # 4. Extract metadata
            logger.info("Step 4: Extracting metadata")
            documents = self.metadata_extractor.batch_extract(documents)
            documents = self._add_parent_metadata(documents)

            # 5. Chunk documents
            logger.info("Step 5: Chunking documents")
            chunks = self.chunker.split(documents)
            chunks = self._add_child_metadata(chunks)
            logger.info(f"Created {len(chunks)} chunks")

            # 6. Add embeddings
            logger.info("Step 6: Generating embeddings")
            ids = self.vector_store.add_documents(chunks)
            logger.info(f"Stored {len(ids)} chunks in vector DB")

            # 8. Get statistics
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

    def _add_parent_metadata(self, documents: List) -> List:
        """Tag source documents so child chunks can be expanded later."""
        for index, document in enumerate(documents):
            parent_id = (
                document.metadata.get("parent_id")
                or document.metadata.get("content_hash")
                or document.metadata.get("source")
                or str(uuid.uuid4())
            )
            document.metadata["parent_id"] = str(parent_id)
            document.metadata["parent_rank"] = index
            document.metadata["parent_source"] = document.metadata.get(
                "source", f"document_{index}"
            )
        return documents

    def _add_child_metadata(self, chunks: List) -> List:
        """Tag chunks with stable child metadata for parent-child retrieval."""
        parent_counts = {}
        for index, chunk in enumerate(chunks):
            parent_id = str(chunk.metadata.get("parent_id") or "unknown_parent")
            child_index = parent_counts.get(parent_id, 0)
            parent_counts[parent_id] = child_index + 1

            chunk.metadata["child_id"] = chunk.metadata.get(
                "child_id", f"{parent_id}:{child_index}"
            )
            chunk.metadata["chunk_index"] = child_index
            chunk.metadata["global_chunk_index"] = index
            chunk.metadata["is_child_chunk"] = True
        return chunks

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
