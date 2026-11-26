"""
Everything that talks to Milvus.

All functions here are SYNCHRONOUS (pymilvus is sync). Call them from routes
declared with plain `def` (FastAPI runs those in a threadpool), or from async
code through `asyncio.to_thread(...)`. Never call them directly from an
`async def`, or they block the event loop (and every other user's streaming).

Ownership model: each collection stores its owner in its Milvus *properties*:
    owner            user id (str) or "public"
    type             "none" | "conversation" | "knowledge_base"
    conversation_id  chat id (only for type "conversation")
    name, description
"""

import threading
import uuid
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from cachetools import TTLCache
from fastapi import HTTPException, status
from pymilvus import DataType, MilvusClient

from genai_api.core.config import CHUNK_SIZE, EMBEDDINGS_SIZE, settings
from genai_api.schemas.knowledge import CollectionInfo, CollectionScope, CollectionType


@lru_cache
def get_milvus_client() -> MilvusClient:
    """One shared client for the whole process (not one connection per call)."""
    if not settings.milvus_pwd:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Milvus password is not configured",
        )
    return MilvusClient(
        uri=settings.milvus_uri, token=f"{settings.milvus_user}:{settings.milvus_pwd}"
    )


# ==========================================
# Properties and permissions
# ==========================================


# Properties of a collection (owner, type, ...) never change after creation, and
# scanning every collection of every user was one network call per collection on
# each request. Cached for a few minutes (short, in case a public collection is
# dropped and re-created under the same name by an admin). Dropping a collection
# through this module invalidates its entry. TTLCache is not thread-safe and our
# routes run in a threadpool, hence the lock.
_props_cache: TTLCache = TTLCache(maxsize=10_000, ttl=300)
_props_lock = threading.Lock()


def _describe(client: MilvusClient, name: str) -> dict[str, Any]:
    with _props_lock:
        cached = _props_cache.get(name)
    if cached is not None:
        return cached
    props = client.describe_collection(collection_name=name).get("properties", {})
    with _props_lock:
        _props_cache[name] = props
    return props


def _forget(name: str) -> None:
    with _props_lock:
        _props_cache.pop(name, None)


def get_collection_properties(client: MilvusClient, name: str) -> dict[str, Any] | None:
    """Properties of the collection, or None if it does not exist."""
    if not client.has_collection(collection_name=name):
        return None
    return _describe(client, name)


def owns(props: dict[str, Any], user_id: uuid.UUID) -> bool:
    return str(props.get("owner")) == str(user_id)


def can_read(props: dict[str, Any], user_id: uuid.UUID) -> bool:
    return props.get("owner") == "public" or owns(props, user_id)


def user_can_read(client: MilvusClient, name: str, user_id: uuid.UUID) -> bool:
    """True if the collection exists and is public or owned by the user."""
    props = get_collection_properties(client, name)
    return props is not None and can_read(props, user_id)


def collection_info(name: str, props: dict[str, Any]) -> CollectionInfo:
    owner = props.get("owner")
    scope = "public" if owner == "public" else "private"

    raw_type = props.get("type")
    if raw_type in {t.value for t in CollectionType}:
        col_type = CollectionType(raw_type)
    else:
        col_type = CollectionType.NONE if scope == "private" else None

    return CollectionInfo(
        id=name,
        name=props.get("display_name") or props.get("name") or name,
        description=props.get("description"),
        scope=scope,
        type=col_type,
        conversation_id=props.get("conversation_id"),
        owner=owner,
    )


# ==========================================
# Collections
# ==========================================


def list_user_collections(
    client: MilvusClient, user_id: uuid.UUID, scope: CollectionScope
) -> list[CollectionInfo]:
    """Public and/or private collections visible to the user."""
    results: list[CollectionInfo] = []
    for name in client.list_collections():
        props = _describe(client, name)
        if props.get("owner") == "public":
            wanted = scope in (CollectionScope.ALL, CollectionScope.PUBLIC)
        elif owns(props, user_id):
            wanted = scope in (CollectionScope.ALL, CollectionScope.PRIVATE)
        else:
            wanted = False
        if wanted:
            results.append(collection_info(name, props))
    return results


def create_collection(
    client: MilvusClient,
    collection_name: str,
    owner: str,
    name: str,
    description: str | None,
    collection_type: CollectionType,
    conversation_id: str | None,
) -> None:
    """Creates a collection with the standard schema, index and properties."""
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDINGS_SIZE
    )
    schema.add_field(
        field_name="text", datatype=DataType.VARCHAR, max_length=CHUNK_SIZE * 4
    )
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=2048)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="COSINE", index_type="HNSW")

    properties: dict[str, Any] = {
        "owner": owner,
        "name": name,
        "type": collection_type.value,
    }
    if description:
        properties["description"] = description
    if conversation_id:
        properties["conversation_id"] = conversation_id

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        properties=properties,
    )


def drop_collection(client: MilvusClient, name: str) -> None:
    client.drop_collection(collection_name=name)
    _forget(name)


# ==========================================
# Conversation collections (one per chat with attached files)
# ==========================================


def _conversation_collections(
    client: MilvusClient, user_id: uuid.UUID, chat_id: uuid.UUID | str
) -> Iterator[str]:
    """Names of the user's collections of type `conversation` linked to the chat.

    Scans every collection in Milvus (properties are cached, see `_describe`).
    Storing chat_id -> collection_id in Postgres would make this one indexed query.
    """
    for name in client.list_collections():
        props = _describe(client, name)
        if (
            owns(props, user_id)
            and props.get("type") == CollectionType.CONVERSATION.value
            and props.get("conversation_id") == str(chat_id)
        ):
            yield name


def find_conversation_collection_id(
    client: MilvusClient, user_id: uuid.UUID, chat_id: uuid.UUID | str
) -> str | None:
    return next(_conversation_collections(client, user_id, chat_id), None)


def delete_conversation_collections(
    client: MilvusClient, user_id: uuid.UUID, chat_id: uuid.UUID | str
) -> list[str]:
    """Drops the chat's conversation collections. Returns the dropped names."""
    names = list(_conversation_collections(client, user_id, chat_id))
    for name in names:
        drop_collection(client, name)
    return names


# ==========================================
# Documents (chunks) inside a collection
# ==========================================


def list_filenames(client: MilvusClient, collection_name: str) -> list[str]:
    """Distinct document filenames of a collection, sorted (so pages are stable)."""
    client.load_collection(collection_name=collection_name)
    iterator = client.query_iterator(
        collection_name=collection_name,
        filter="",
        output_fields=["filename"],
        batch_size=1000,
    )
    filenames: set[str] = set()
    try:
        while batch := iterator.next():
            filenames.update(e["filename"] for e in batch if e.get("filename"))
    finally:
        iterator.close()
    return sorted(filenames)


def insert_chunks(
    client: MilvusClient,
    collection_name: str,
    filename: str,
    url: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    # Re-uploading the same filename adds new chunks without removing the old ones.
    data = [
        {"vector": emb, "text": chunk, "filename": filename, "url": url}
        for chunk, emb in zip(chunks, embeddings, strict=True)
    ]
    client.insert(collection_name=collection_name, data=data)
    client.load_collection(collection_name=collection_name)


def _quote(value: str) -> str:
    """Escapes a string for use inside a Milvus filter expression."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def delete_document(client: MilvusClient, collection_name: str, filename: str) -> None:
    """Deletes every chunk of a document."""
    client.delete(
        collection_name=collection_name, filter=f"filename == {_quote(filename)}"
    )
