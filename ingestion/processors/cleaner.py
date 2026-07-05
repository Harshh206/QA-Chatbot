from langchain_core.documents import Document
from typing import List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """Cleans and normalizes text content"""

    def __init__(self, preserve_markdown: bool = True):
        self.preserve_markdown = preserve_markdown

    def clean_document(self, document: Document) -> Document:
        """Clean a single document"""
        text = document.page_content

        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Multiple newlines
        text = re.sub(r" +", " ", text)  # Multiple spaces

        # Remove weird Unicode artifacts
        text = text.replace("\ufeff", "")  # BOM
        text = text.replace("\xa0", " ")  # Non-breaking space

        # If preserving markdown, keep structural markers
        if not self.preserve_markdown:
            # Remove markdown syntax if needed
            text = re.sub(r"#+ ", "", text)  # Headers
            text = re.sub(r"[*_~`]", "", text)  # Formatting
            text = re.sub(r"!?\[.*?\]\(.*?\)", "", text)  # Links/images

        document.page_content = text.strip()
        document.metadata["cleaned"] = True

        return document

    def batch_clean(self, documents: List[Document]) -> List[Document]:
        """Clean multiple documents"""
        return [self.clean_document(doc) for doc in documents]
