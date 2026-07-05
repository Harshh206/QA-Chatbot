from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    SentenceTransformersTokenTextSplitter,
    CharacterTextSplitter,
)
from langchain_core.documents import Document
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class ChunkStrategy:
    """Base chunking strategy"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError


class RecursiveChunker(ChunkStrategy):
    """Recursive character splitter with fallbacks"""

    def split(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        return splitter.split_documents(documents)


class MarkdownChunker(ChunkStrategy):
    """Markdown-aware chunker that preserves structure"""

    def split(self, documents: List[Document]) -> List[Document]:
        # First, use Markdown text splitter
        md_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = md_splitter.split_documents(documents)

        # Then, refine chunks to preserve markdown hierarchy
        refined_chunks = []
        for chunk in chunks:
            # Preserve heading context
            content = chunk.page_content
            lines = content.split("\n")

            # Find first heading level
            heading_level = None
            for line in lines:
                if line.startswith("#"):
                    heading_level = len(line) - len(line.lstrip("#"))
                    break

            if heading_level is not None:
                # Add heading context to metadata
                chunk.metadata["heading_level"] = heading_level

                # Extract section title
                for line in lines:
                    if line.startswith("#" * heading_level + " "):
                        chunk.metadata["section_title"] = line.strip("#").strip()
                        break

            refined_chunks.append(chunk)

        return refined_chunks


class HybridChunker(ChunkStrategy):
    """Hybrid approach: use markdown if available, else recursive"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
        self.md_chunker = MarkdownChunker(chunk_size, chunk_overlap)
        self.recursive_chunker = RecursiveChunker(chunk_size, chunk_overlap)

    def split(self, documents: List[Document]) -> List[Document]:
        # Check if content has markdown structure
        has_markdown = any(
            doc.page_content.startswith("#") or "```" in doc.page_content
            for doc in documents
        )

        if has_markdown:
            return self.md_chunker.split(documents)
        else:
            return self.recursive_chunker.split(documents)


class TokenChunker(ChunkStrategy):
    """Token-based chunker using sentence transformers"""

    def split(self, documents: List[Document]) -> List[Document]:
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(documents)


def get_chunker(strategy: str = "hybrid", **kwargs) -> ChunkStrategy:
    """Factory function to get appropriate chunker"""
    strategies = {
        "recursive": RecursiveChunker,
        "markdown": MarkdownChunker,
        "hybrid": HybridChunker,
        "token": TokenChunker,
    }

    chunker_class = strategies.get(strategy.lower(), HybridChunker)
    return chunker_class(**kwargs)
