from pathlib import Path
from typing import List
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
)

from markitdown import MarkItDown

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from different file formats."""

    MARKITDOWN_FORMATS = {
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xlsx",
        ".xls",
        ".html",
    }

    LOADER_MAP = {
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
        ".json": JSONLoader,
    }

    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.markitdown = MarkItDown()

        self.supported_extensions = (
            set(self.LOADER_MAP.keys()) | self.MARKITDOWN_FORMATS
        )

    def load_single_file(self, file_path: Path) -> List[Document]:
        """Load a single document."""

        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in self.supported_extensions:
            logger.warning(f"Unsupported file: {file_path}")
            return []

        try:
            if ext in self.MARKITDOWN_FORMATS:
                result = self.markitdown.convert(str(file_path))

                documents = [
                    Document(
                        page_content=result.text_content,
                        metadata={},
                    )
                ]

            else:
                loader_class = self.LOADER_MAP[ext]

                if ext == ".json":
                    loader = loader_class(
                        str(file_path),
                        jq_schema=".",
                        text_content=False,
                    )
                else:
                    loader = loader_class(str(file_path))

                documents = loader.load()

            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": ext.lstrip("."),
                "file_size": file_path.stat().st_size,
                "is_markdown": ext in self.MARKITDOWN_FORMATS or ext == ".md",
            }

            for doc in documents:
                doc.metadata.update(metadata)

            logger.info(f"Loaded {file_path.name}")

            return documents

        except Exception as e:
            logger.exception(f"Failed to load {file_path}: {e}")
            return []

    def load_directory(self, recursive: bool = True) -> List[Document]:
        """Load all supported documents from the input directory."""

        pattern = "**/*" if recursive else "*"

        documents: List[Document] = []

        for file_path in self.input_dir.glob(pattern):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_extensions
            ):
                documents.extend(self.load_single_file(file_path))

        logger.info(f"Loaded {len(documents)} document(s)")

        return documents
