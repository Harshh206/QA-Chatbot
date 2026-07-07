from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
)
from langchain_core.documents import Document
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles loading documents from various file formats"""

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".md": UnstructuredMarkdownLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".json": JSONLoader,
    }

    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.supported_extensions = set(self.LOADER_MAP.keys())

    def load_single_file(self, file_path: Path) -> List[Document]:
        """Load a single file based on its extension"""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {ext} for {file_path}")
            return []

        loader_class = self.LOADER_MAP[ext]

        # Special handling for different loaders
        if ext == ".json":
            loader = loader_class(file_path, jq_schema=".", text_content=False)
        else:
            loader = loader_class(str(file_path))

        documents = loader.load()

        # Add file metadata
        for doc in documents:
            doc.metadata.update(
                {
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": ext[1:],
                    "file_size": file_path.stat().st_size,
                }
            )

        return documents

    def load_directory(self, recursive: bool = True) -> List[Document]:
        """Load all supported documents from a directory"""
        all_docs = []

        pattern = "**/*" if recursive else "*"
        for file_path in self.input_dir.glob(pattern):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_extensions
            ):
                try:
                    docs = self.load_single_file(file_path)
                    all_docs.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        return all_docs
