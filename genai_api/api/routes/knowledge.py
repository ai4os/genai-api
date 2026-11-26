"""
Knowledge collections (Milvus): HTTP layer only.

Routes are plain `def` (not `async def`) on purpose: Milvus and the embeddings
client are synchronous, and FastAPI runs `def` routes in a threadpool, so they
do not block the event loop. The real work lives in services/milvus.py and
services/documents.py.
"""

import uuid

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
from pymilvus import MilvusClient

from genai_api.api.deps import CurrentUser, get_current_user
from genai_api.schemas.knowledge import (
    CollectionCreate,
    CollectionDeleteResponse,
    CollectionInfo,
    CollectionScope,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from genai_api.services import documents, milvus

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _require_collection(
    client: MilvusClient, collection_id: str, user_id: uuid.UUID, *, write: bool
) -> None:
    """404 if the collection does not exist, 403 if the user may not use it.
    `write=True` requires ownership; otherwise public collections are readable too."""
    props = milvus.get_collection_properties(client, collection_id)
    if props is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Collection '{collection_id}' not found"
        )
    allowed = milvus.owns(props, user_id) if write else milvus.can_read(props, user_id)
    if not allowed:
        action = "modify" if write else "view"
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"You are not authorized to {action} this collection",
        )


def _upstream_error(message: str, exc: Exception) -> HTTPException:
    return HTTPException(status.HTTP_502_BAD_GATEWAY, f"{message}: {exc}")


@router.get("/collections", response_model=list[CollectionInfo])
def list_collections(
    scope: CollectionScope = Query(CollectionScope.ALL),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Public, private or all collections visible to the user."""
    try:
        collections = milvus.list_user_collections(
            milvus.get_milvus_client(), current_user.id, scope
        )
        return sorted(
            collections,
            key=lambda collection: (collection.name.casefold(), collection.id),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _upstream_error("Error listing Milvus collections", exc) from exc


@router.post(
    "/collections", response_model=CollectionInfo, status_code=status.HTTP_201_CREATED
)
def create_collection(
    payload: CollectionCreate, current_user: CurrentUser = Depends(get_current_user)
):
    """Creates a private collection owned by the current user."""
    collection_id = f"col_{uuid.uuid4().hex}"
    try:
        milvus.create_collection(
            milvus.get_milvus_client(),
            collection_name=collection_id,
            owner=str(current_user.id),
            name=payload.name,
            description=payload.description,
            collection_type=payload.type,
            conversation_id=payload.conversation_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _upstream_error("Error creating collection in Milvus", exc) from exc

    return CollectionInfo(
        id=collection_id,
        name=payload.name,
        description=payload.description,
        scope="private",
        type=payload.type,
        conversation_id=payload.conversation_id,
        owner=str(current_user.id),
    )


@router.delete("/collections/{collection_id}", response_model=CollectionDeleteResponse)
def delete_collection(
    collection_id: str, current_user: CurrentUser = Depends(get_current_user)
):
    client = milvus.get_milvus_client()
    _require_collection(client, collection_id, current_user.id, write=True)
    try:
        milvus.drop_collection(client, collection_id)
    except Exception as exc:
        raise _upstream_error(
            f"Error deleting collection '{collection_id}'", exc
        ) from exc
    return CollectionDeleteResponse(
        collection_id=collection_id,
        message=f"Collection '{collection_id}' deleted successfully",
    )


@router.get(
    "/collections/{collection_id}/documents", response_model=DocumentListResponse
)
def list_documents(
    collection_id: str,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(50, ge=1, le=200),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Document filenames of a collection, sorted and paginated."""
    client = milvus.get_milvus_client()
    _require_collection(client, collection_id, current_user.id, write=False)
    try:
        names = milvus.list_filenames(client, collection_id)
    except Exception as exc:
        raise _upstream_error(
            f"Error listing documents in collection '{collection_id}'", exc
        ) from exc

    start = (page - 1) * page_size
    return DocumentListResponse(
        items=names[start : start + page_size],
        total=len(names),
        page=page,
        page_size=page_size,
    )


@router.post(
    "/collections/{collection_id}/documents", response_model=DocumentUploadResponse
)
def upload_document(
    collection_id: str,
    file: UploadFile = File(..., description="Document to convert and index"),
    url: str | None = Form(None, description="Optional source URL"),
    current_user: CurrentUser = Depends(get_current_user),
):
    """file -> markdown -> chunks -> embeddings (LiteLLM) -> Milvus."""
    client = milvus.get_milvus_client()
    _require_collection(client, collection_id, current_user.id, write=True)

    filename = file.filename or "uploaded_doc"
    content = file.file.read()
    chunks = documents.build_chunks(content, filename)  # 400 on bad/empty file

    try:
        embeddings = documents.generate_embeddings(
            chunks, documents.get_openai_client()
        )
    except Exception as exc:
        raise _upstream_error("Error generating embeddings", exc) from exc

    try:
        milvus.insert_chunks(
            client, collection_id, filename, url or filename, chunks, embeddings
        )
    except Exception as exc:
        raise _upstream_error(
            "Error inserting document vectors into Milvus", exc
        ) from exc

    return DocumentUploadResponse(
        filename=filename,
        collection_id=collection_id,
        chunks_count=len(chunks),
        status="indexed",
    )


@router.delete(
    "/collections/{collection_id}/documents/{filename}",
    response_model=DocumentDeleteResponse,
)
def delete_document(
    collection_id: str,
    filename: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Deletes every chunk of the document."""
    client = milvus.get_milvus_client()
    _require_collection(client, collection_id, current_user.id, write=True)
    try:
        milvus.delete_document(client, collection_id, filename)
    except Exception as exc:
        raise _upstream_error(f"Error deleting document '{filename}'", exc) from exc
    return DocumentDeleteResponse(
        filename=filename,
        collection_id=collection_id,
        message=f"Document '{filename}' deleted successfully from collection '{collection_id}'",
    )
