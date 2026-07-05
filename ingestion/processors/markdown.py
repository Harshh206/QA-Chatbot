from langchain_core.documents import Document
from typing import List
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class MarkdownConverter:
    """Converts various document formats to Markdown for better structure preservation"""

    def __init__(self):
        self.supported_formats = [".pdf", ".docx", ".doc", ".html", ".pptx"]
        self._conversion_cache = {}

    def convert_document(self, document: Document) -> Document:
        """Convert a document to Markdown if needed"""
        file_path = document.metadata.get("source")
        if not file_path:
            return document

        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        # If already markdown, return as-is
        if ext in [".md", ".markdown"]:
            return document

        # Convert if supported
        if ext in self.supported_formats:
            key = str(file_path.resolve())
            if key in self._conversion_cache:
                markdown_content = self._conversion_cache[key]
            else:
                try:
                    markdown_content = self._convert_to_markdown(file_path)
                    self._conversion_cache[key] = markdown_content
                except Exception as e:
                    logger.error(f"Failed to convert {file_path} to markdown: {e}")
                    return document

            if markdown_content:
                document.page_content = markdown_content
                document.metadata["converted_to_markdown"] = True
                document.metadata["original_format"] = ext

        return document

    def _convert_to_markdown(self, file_path: Path) -> str:
        """Use external tools to convert to markdown"""
        # Try using markitdown (newer tool)
        markitdown = None
        try:
            from markitdown import MarkItDown
            markitdown = MarkItDown
        except ImportError:
            logger.warning("markitdown not installed, falling back to pandoc")
        
        if markitdown:
            coverter = markitdown()
            result = coverter.convert(str(file_path))
            return result.text_content

        # Fallback: Use pandoc if available
        try:
            with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
                cmd = ["pandoc", str(file_path), "-o", tmp.name, "--to", "markdown"]
                subprocess.run(cmd, capture_output=True, check=True)
                with open(tmp.name, "r") as f:
                    return f.read()
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Pandoc not found or conversion failed")
            raise

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process multiple documents"""
        return [self.convert_document(doc) for doc in documents]
