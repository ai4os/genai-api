"""
Turns an uploaded file into vectors: file -> markdown -> chunks -> embeddings.

Synchronous (see services/milvus.py for why): call it from `def` routes.
"""

import re

import anydoc
from fastapi import HTTPException, status
from openai import OpenAI
from genai_api.extractors import oscar, local

from genai_api.core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDINGS_MODEL,
    settings,
)


def get_openai_client(api_key: str | None = None) -> OpenAI:
    """OpenAI-compatible client pointing to LiteLLM (used only for embeddings)."""
    base_url = settings.litellm_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return OpenAI(base_url=base_url, api_key=api_key or settings.litellm_api_key)


def convert_to_markdown(content: bytes, filename: str) -> str:
    """Converts a document to markdown with anydoc (fallback: plain UTF-8)."""
    try:
        if settings.extractor == "oscar":
            return oscar.oscar_document_extractor(content, filename)
        else:
            return local.local_document_extractor(content, filename)
    except Exception as exc:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to convert document '{filename}' to markdown: {exc}",
            ) from exc


def semantic_chunking(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Splits text into chunks respecting semantic boundaries, with overlap."""
    separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def recursive_split(sub_text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [sub_text]
        sep = seps[0]
        if sep == "":
            return list(sub_text)

        splits = sub_text.split(sep)
        splits = [s + sep for s in splits[:-1]] + [splits[-1]]

        new_chunks: list[str] = []
        current = ""
        for split in splits:
            if len(split) > chunk_size:
                if current:
                    new_chunks.append(current)
                    current = ""
                new_chunks.extend(recursive_split(split, seps[1:]))
            elif len(current) + len(split) > chunk_size:
                if current:
                    new_chunks.append(current)
                current = split
            else:
                current += split
        if current:
            new_chunks.append(current)
        return new_chunks

    # 1) Pack the pieces into chunks of at most `chunk_size` characters.
    packed: list[str] = []
    current = ""
    for block in recursive_split(text, separators):
        if len(current) + len(block) <= chunk_size:
            current += block
        else:
            if current.strip():
                packed.append(current)
            current = block
    if current.strip():
        packed.append(current)

    # 2) Each chunk (except the first) starts with the last ~`overlap` characters
    #    of the previous one, so a sentence cut at the boundary appears whole in one
    #    of them. Result: every chunk has at most `chunk_size + overlap` characters.
    return [
        (_overlap_tail(packed[i - 1], overlap) + chunk).strip() if i else chunk.strip()
        for i, chunk in enumerate(packed)
    ]


def _overlap_tail(previous: str, overlap: int) -> str:
    """Last `overlap` characters of `previous`, starting at a word boundary."""
    if overlap <= 0 or len(previous) <= overlap:
        return previous if overlap > 0 else ""
    tail = previous[-overlap:]
    match = re.search(r"\s", tail)
    return tail[match.end() :] if match else tail


def build_chunks(content: bytes, filename: str) -> list[str]:
    """File bytes -> text chunks. Raises 400 if there is no text to index."""
    chunks = semantic_chunking(convert_to_markdown(content, filename))
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document resulted in empty text content after processing",
        )
    return chunks


def generate_embeddings(
    chunks: list[str], openai_client: OpenAI, batch_size: int = 10
) -> list[list[float]]:
    """One embedding vector per chunk, requested in batches."""
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        response = openai_client.embeddings.create(
            model=EMBEDDINGS_MODEL, input=chunks[i : i + batch_size]
        )
        embeddings.extend(item.embedding for item in response.data)
    return embeddings
