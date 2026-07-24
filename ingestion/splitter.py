import re
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)

logger = logging.getLogger(__name__)


class ChunkStrategy:
    """Base class for all chunking strategies."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError


class RecursiveChunker(ChunkStrategy):
    """Character-based chunking for plain text."""

    def split(self, documents: List[Document]) -> List[Document]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = splitter.split_documents(documents)

        logger.info("RecursiveChunker created %d chunks", len(chunks))

        return chunks


class MarkdownChunker(ChunkStrategy):
    """Markdown-aware chunking."""

    def split(self, documents: List[Document]) -> List[Document]:

        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            self._add_metadata(chunk)

        logger.info("MarkdownChunker created %d chunks", len(chunks))

        return chunks

    def _add_metadata(self, chunk: Document) -> None:
        """Extract heading information."""

        lines = chunk.page_content.splitlines()

        for line in lines:
            line = line.strip()

            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))

                chunk.metadata["heading_level"] = level
                chunk.metadata["section_title"] = line.lstrip("#").strip()
                break

class AutoChunker(ChunkStrategy):
    """Automatically select the best chunking strategy."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__(chunk_size, chunk_overlap)

        self.md_chunker = MarkdownChunker(
            chunk_size,
            chunk_overlap,
        )

        self.recursive_chunker = RecursiveChunker(
            chunk_size,
            chunk_overlap,
        )

    def split(self, documents):

        chunks = []

        for document in documents:

            loader = document.metadata.get("loader")
            file_type = document.metadata.get("file_type")

            if loader == "markitdown":
                chunks.extend(
                    self.md_chunker.split([document])
                )

            elif file_type == "md":
                chunks.extend(
                    self.md_chunker.split([document])
                )

            else:
                chunks.extend(
                    self.recursive_chunker.split([document])
                )

        return chunks

def get_chunker(
    strategy: str = "auto",
    **kwargs,
) -> ChunkStrategy:
    """Factory function."""

    strategies = {
        "recursive": RecursiveChunker,
        "markdown": MarkdownChunker,
        "auto": AutoChunker,
    }

    chunker_class = strategies.get(
        strategy.lower(),
        AutoChunker,
    )

    return chunker_class(**kwargs)
