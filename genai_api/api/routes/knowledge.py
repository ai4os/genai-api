from typing import Any
import uuid

import anydoc
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from openai import OpenAI
from pymilvus import DataType, MilvusClient

from genai_api.api.deps import CurrentUser, get_current_user
from genai_api.core.config import settings
from genai_api.schemas.knowledge import (
    CollectionCreate,
    CollectionDeleteResponse,
    CollectionInfo,
    CollectionScope,
    CollectionType,
    DocumentDeleteResponse,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

EMBEDDINGS_MODEL = "AI4EOSC/Qwen/Qwen3-Embedding-4B"
EMBEDDINGS_SIZE = 2560  # specific to Qwen3-Embedding-4B
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 256


# ==========================================
# Helpers & Milvus / Embeddings Initializers
# ==========================================


def get_milvus_client() -> MilvusClient:
    """Initializes and returns a MilvusClient instance."""
    if not settings.milvus_pwd:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Milvus password is not configured",
        )
    token = f"{settings.milvus_user}:{settings.milvus_pwd}"
    return MilvusClient(uri=settings.milvus_uri, token=token)


def get_openai_client(api_key: str | None = None) -> OpenAI:
    """Initializes OpenAI client pointing to the LiteLLM base URL."""
    base_url = settings.litellm_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return OpenAI(
        base_url=base_url,
        api_key=api_key or settings.litellm_api_key,
    )


def parse_collection_info(
    collection_name: str,
    desc: dict[str, Any] | None = None,
) -> CollectionInfo:
    """
    Parses Milvus collection metadata from describe_collection properties.
    """
    properties = desc.get("properties", {}) if desc else {}
    owner = properties.get("owner")
    name = properties.get("name") or collection_name
    description = properties.get("description")
    conversation_id = properties.get("conversation_id")
    type_str = properties.get("type")

    scope = "public" if owner == "public" else "private"
    col_type = None
    if type_str in [t.value for t in CollectionType]:
        col_type = CollectionType(type_str)
    elif scope == "private":
        col_type = CollectionType.NONE

    return CollectionInfo(
        id=collection_name,
        name=name,
        description=description,
        scope=scope,
        type=col_type,
        conversation_id=conversation_id,
        owner=owner,
    )


def is_user_collection_owner(
    milvus_client: MilvusClient,
    collection_name: str,
    user_id: uuid.UUID,
) -> bool:
    """Checks whether the collection exists and belongs to the given user."""
    if not milvus_client.has_collection(collection_name=collection_name):
        return False
    desc = milvus_client.describe_collection(collection_name=collection_name)
    owner = desc.get("properties", {}).get("owner")
    return str(owner) == str(user_id)


def is_user_collection_accessible(
    milvus_client: MilvusClient,
    collection_name: str,
    user_id: uuid.UUID,
) -> bool:
    """Checks whether the collection exists and is accessible by the user (public or owned)."""
    if not milvus_client.has_collection(collection_name=collection_name):
        return False
    desc = milvus_client.describe_collection(collection_name=collection_name)
    owner = desc.get("properties", {}).get("owner")
    return owner == "public" or str(owner) == str(user_id)


def find_conversation_collection_id(
    milvus_client: MilvusClient,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID | str,
) -> str | None:
    """
    Returns the id of the user's collection of type `conversation` linked to the
    given chat, or None.
    """
    for name in milvus_client.list_collections():
        props = milvus_client.describe_collection(collection_name=name).get(
            "properties", {}
        )
        if (
            str(props.get("owner")) == str(user_id)
            and props.get("type") == CollectionType.CONVERSATION.value
            and props.get("conversation_id") == str(conversation_id)
        ):
            return name
    return None


def ensure_collection_schema(
    milvus_client: MilvusClient,
    collection_name: str,
    owner: str,
    name: str | None = None,
    description: str | None = None,
    collection_type: CollectionType | None = None,
    conversation_id: str | None = None,
    embeddings_size: int = EMBEDDINGS_SIZE,
    chunk_size: int = CHUNK_SIZE,
):
    """Creates the standard Milvus collection schema, index, and properties if not present."""
    if not milvus_client.has_collection(collection_name=collection_name):
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=embeddings_size,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=chunk_size * 4,
        )
        schema.add_field(
            field_name="filename",
            datatype=DataType.VARCHAR,
            max_length=512,
        )
        schema.add_field(
            field_name="url",
            datatype=DataType.VARCHAR,
            max_length=2048,
        )

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="HNSW",
        )

        properties: dict[str, Any] = {"owner": owner}
        if name:
            properties["name"] = name
        if description:
            properties["description"] = description
        if conversation_id:
            properties["conversation_id"] = conversation_id
        if collection_type is not None:
            type_val = (
                collection_type.value
                if isinstance(collection_type, CollectionType)
                else str(collection_type)
            )
            properties["type"] = type_val

        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            properties=properties,
        )


def semantic_chunking(
    text: str, chunk_size: int = 2048, overlap: int = 256
) -> list[str]:
    """Splits text into chunks respecting semantic boundaries with overlap."""
    separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def recursive_split(sub_text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [sub_text]
        sep = seps[0]
        if sep == "":
            return list(sub_text)

        try:
            splits = sub_text.split(sep)
            splits = [s + sep for s in splits[:-1]] + [splits[-1]]
        except Exception:
            return [sub_text]

        new_chunks = []
        current_chunk = ""

        for split in splits:
            if len(split) > chunk_size:
                if current_chunk:
                    new_chunks.append(current_chunk)
                    current_chunk = ""
                sub_chunks = recursive_split(split, seps[1:])
                new_chunks.extend(sub_chunks)
            elif len(current_chunk) + len(split) > chunk_size:
                if current_chunk:
                    new_chunks.append(current_chunk)
                current_chunk = split
            else:
                current_chunk += split

        if current_chunk:
            new_chunks.append(current_chunk)

        return new_chunks

    raw_chunks = recursive_split(text, separators)
    final_chunks = []
    current_chunk = ""

    for i, block in enumerate(raw_chunks):
        if len(current_chunk) + len(block) <= chunk_size:
            current_chunk += block
        else:
            if current_chunk.strip():
                final_chunks.append(current_chunk.strip())

            overlap_buffer = ""
            backtrack_idx = i - 1
            while backtrack_idx >= 0 and len(overlap_buffer) < overlap:
                overlap_buffer = raw_chunks[backtrack_idx] + overlap_buffer
                backtrack_idx -= 1

            current_chunk = overlap_buffer + block

    if current_chunk and current_chunk.strip():
        final_chunks.append(current_chunk.strip())

    return final_chunks


def convert_document_to_markdown(file_content: bytes, filename: str) -> str:
    """Converts a document to markdown using anydoc."""
    try:
        doc_format = anydoc.format_from_path(filename)
        return anydoc.to_markdown_bytes(file_content, format=doc_format)
    except Exception as e:
        # Fallback to UTF-8 decoding if anydoc throws or for plain text/markdown
        try:
            return file_content.decode("utf-8")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to convert document '{filename}' to markdown: {str(e)}",
            ) from e


def generate_embeddings(
    chunks: list[str],
    openai_client: OpenAI,
    batch_size: int = 10,
) -> list[list[float]]:
    """Generates vector embeddings for a list of text chunks in batches."""
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        response = openai_client.embeddings.create(
            model=EMBEDDINGS_MODEL,
            input=batch,
        )
        for item in response.data:
            embeddings.append(item.embedding)
    return embeddings


def delete_conversation_collections(
    milvus_client: MilvusClient,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID | str,
) -> list[str]:
    """
    Drops every `conversation` collection owned by the user and linked to the
    given chat. Returns the names of the dropped collections (empty if none).
    """
    deleted: list[str] = []
    for name in milvus_client.list_collections():
        props = milvus_client.describe_collection(collection_name=name).get(
            "properties", {}
        )
        if (
            str(props.get("owner")) == str(user_id)
            and props.get("type") == CollectionType.CONVERSATION.value
            and props.get("conversation_id") == str(conversation_id)
        ):
            milvus_client.drop_collection(collection_name=name)
            deleted.append(name)
    return deleted


# ==========================================
# Routes
# ==========================================


@router.get("/collections", response_model=list[CollectionInfo])
async def list_collections(
    scope: CollectionScope = Query(
        CollectionScope.ALL,
        description="Filter collections by scope: 'public', 'private', or 'all'",
    ),
    current_user: CurrentUser = Depends(get_current_user),
) -> list[CollectionInfo]:
    """
    List public, private, or all collections.
    For private collections, returns the collection type (none, conversation, knowledge_base).
    """
    client = get_milvus_client()
    try:
        all_collections = client.list_collections()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing Milvus collections: {str(e)}",
        ) from e

    user_id_str = str(current_user.id)
    results: list[CollectionInfo] = []

    for col in all_collections:
        desc = client.describe_collection(collection_name=col)
        owner = desc.get("properties", {}).get("owner")
        info = parse_collection_info(col, desc)

        if owner == "public":
            if scope in [CollectionScope.ALL, CollectionScope.PUBLIC]:
                results.append(info)
        elif owner == user_id_str:
            if scope in [CollectionScope.ALL, CollectionScope.PRIVATE]:
                results.append(info)

    return results


@router.post(
    "/collections", response_model=CollectionInfo, status_code=status.HTTP_201_CREATED
)
async def create_collection(
    payload: CollectionCreate,
    current_user: CurrentUser = Depends(get_current_user),
) -> CollectionInfo:
    """
    Create a new private collection for the current user.
    Generates a UUID collection ID in Milvus and stores name, description, type, and optional conversation_id in properties.
    """
    collection_id = f"col_{uuid.uuid4().hex}"
    user_id_str = str(current_user.id)

    client = get_milvus_client()
    try:
        ensure_collection_schema(
            milvus_client=client,
            collection_name=collection_id,
            owner=user_id_str,
            name=payload.name,
            description=payload.description,
            collection_type=payload.type,
            conversation_id=payload.conversation_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating collection in Milvus: {str(e)}",
        ) from e

    return CollectionInfo(
        id=collection_id,
        name=payload.name,
        description=payload.description,
        scope="private",
        type=payload.type,
        conversation_id=payload.conversation_id,
        owner=user_id_str,
    )


@router.get("/collections/{collection_id}/documents", response_model=list[str])
async def list_collection_documents(
    collection_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> list[str]:
    """
    List all document filenames within a particular collection.
    """
    client = get_milvus_client()

    if not client.has_collection(collection_name=collection_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_id}' not found",
        )

    if not is_user_collection_accessible(client, collection_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to view documents in this collection",
        )

    try:
        client.load_collection(collection_name=collection_id)
        iterator = client.query_iterator(
            collection_name=collection_id,
            filter="",
            output_fields=["filename"],
            batch_size=1000,
        )
        filenames: set[str] = set()
        while True:
            result_batch = iterator.next()
            if not result_batch:
                break
            for entry in result_batch:
                fname = entry.get("filename")
                if fname:
                    filenames.add(fname)
        iterator.close()
        return sorted(list(filenames))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents in collection '{collection_id}': {str(e)}",
        ) from e


@router.post(
    "/collections/{collection_id}/documents", response_model=DocumentUploadResponse
)
async def upload_document_to_collection(
    collection_id: str,
    file: UploadFile = File(..., description="Document file to convert and index"),
    url: str | None = Form(None, description="Optional source URL of the document"),
    current_user: CurrentUser = Depends(get_current_user),
) -> DocumentUploadResponse:
    """
    Upload a document, convert to markdown using anydoc, generate embeddings,
    and save chunks into the given collection.
    Validates that the user is the owner of the collection.
    """
    client = get_milvus_client()

    if not client.has_collection(collection_name=collection_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_id}' not found",
        )

    if not is_user_collection_owner(client, collection_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to upload documents to this collection",
        )

    # Read and convert document
    try:
        content = await file.read()
        markdown_text = convert_document_to_markdown(
            content, file.filename or "doc.txt"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing file '{file.filename}': {str(e)}",
        ) from e

    # Chunk text
    chunks = semantic_chunking(
        markdown_text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document resulted in empty text content after processing",
        )

    # Generate embeddings
    openai_client = get_openai_client()
    try:
        embeddings = generate_embeddings(chunks, openai_client)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error generating embeddings: {str(e)}",
        ) from e

    # Prepare Milvus data records
    doc_url = url or file.filename or ""
    data_to_insert = [
        {
            "vector": embeddings[i],
            "text": chunks[i],
            "filename": file.filename or "uploaded_doc",
            "url": doc_url,
        }
        for i in range(len(chunks))
    ]

    try:
        # We do not delete prior chunks for this file if re-uploading
        # Leave commented as this may be a feature we want for the future
        # client.delete(
        #     collection_name=collection_id,
        #     filter=f'filename == "{file.filename}"',
        # )
        client.insert(collection_name=collection_id, data=data_to_insert)
        client.load_collection(collection_name=collection_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inserting document vectors into Milvus: {str(e)}",
        ) from e

    return DocumentUploadResponse(
        filename=file.filename or "unknown",
        collection_id=collection_id,
        chunks_count=len(chunks),
        status="indexed",
    )


@router.delete(
    "/collections/{collection_id}/documents/{filename}",
    response_model=DocumentDeleteResponse,
)
async def delete_collection_document(
    collection_id: str,
    filename: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> DocumentDeleteResponse:
    """
    Delete a particular document (all chunks with matching filename) within a collection.
    Validates that the user is the owner of the collection.
    """
    client = get_milvus_client()

    if not client.has_collection(collection_name=collection_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_id}' not found",
        )

    if not is_user_collection_owner(client, collection_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to delete documents from this collection",
        )

    try:
        client.delete(
            collection_name=collection_id,
            filter=f'filename == "{filename}"',
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document '{filename}' from collection '{collection_id}': {str(e)}",
        ) from e

    return DocumentDeleteResponse(
        filename=filename,
        collection_id=collection_id,
        message=f"Document '{filename}' deleted successfully from collection '{collection_id}'",
    )


@router.delete("/collections/{collection_id}", response_model=CollectionDeleteResponse)
async def delete_collection(
    collection_id: str,
    current_user: CurrentUser = Depends(get_current_user),
) -> CollectionDeleteResponse:
    """
    Delete a given collection.
    Validates that the user is the owner of the collection.
    """
    client = get_milvus_client()

    if not client.has_collection(collection_name=collection_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_id}' not found",
        )

    if not is_user_collection_owner(client, collection_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to delete this collection",
        )

    try:
        client.drop_collection(collection_name=collection_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting collection '{collection_id}': {str(e)}",
        ) from e

    return CollectionDeleteResponse(
        collection_id=collection_id,
        message=f"Collection '{collection_id}' deleted successfully",
    )
