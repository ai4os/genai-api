
import anydoc

def local_document_extractor(content: bytes, filename: str) -> bytes:
    """Extracts text from a document using local extraction."""
    doc_format = anydoc.format_from_path(filename)
    return anydoc.to_markdown_bytes(content, format=doc_format)