"""
Business logic of chats: validation, history, and streaming a reply.

routes/chats.py only does HTTP; everything that needs more than a few lines
lives here. Order of `POST /chats/{id}/messages`:

    check_single_target ─► get_user_chat ─► validate_and_resolve_collections
      ─► save_user_message ─► load_llm_messages ─► open_stream ─► stream_reply
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator

import anyio
from fastapi import HTTPException
from sqlalchemy import delete, exists, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from genai_api.core.config import settings
from genai_api.db.session import async_session
from genai_api.models.chat import Chat
from genai_api.models.message import Message, MessageRole
from genai_api.schemas.message import MessageCreate
from genai_api.services import milvus
from genai_api.services.litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)


# ==========================================
# Chats
# ==========================================


async def get_user_chat(
    db: AsyncSession, chat_id: uuid.UUID, user_id: uuid.UUID, *, with_messages=False
) -> Chat:
    """The chat if it belongs to the user, else 404 (same answer for 'does not
    exist' and 'is someone else's', so ids cannot be probed)."""
    stmt = select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
    if with_messages:
        stmt = stmt.options(selectinload(Chat.messages))
    chat = (await db.execute(stmt)).scalar_one_or_none()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


async def delete_chat(db: AsyncSession, chat_id: uuid.UUID, user_id: uuid.UUID) -> None:
    await get_user_chat(db, chat_id, user_id)  # 404 if not the user's

    # Milvus first: if it fails the chat is kept and the client can retry,
    # so no orphan collections are left behind.
    try:
        await asyncio.to_thread(
            lambda: milvus.delete_conversation_collections(
                milvus.get_milvus_client(), user_id, chat_id
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not delete the chat's documents. The chat was not deleted",
        ) from exc

    # Messages go away through the DB's ON DELETE CASCADE.
    await db.execute(delete(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
    await db.commit()


async def touch_chat(db: AsyncSession, chat_id: uuid.UUID) -> None:
    """Bumps chats.updated_at. Saving a message only INSERTs into `messages`, so
    without this the sidebar (ordered by updated_at) would never reorder."""
    await db.execute(
        update(Chat).where(Chat.id == chat_id).values(updated_at=func.now())
    )


# ==========================================
# Validation of an outgoing message
# ==========================================


def check_single_target(payload: MessageCreate) -> None:
    if bool(payload.model_id) == bool(payload.agent_id):
        raise HTTPException(
            status_code=400, detail="Exactly one of model_id or agent_id is required"
        )


async def _chat_has_files(db: AsyncSession, chat_id: uuid.UUID) -> bool:
    result = await db.execute(
        select(
            exists().where(Message.chat_id == chat_id, Message.extra.has_key("files"))
        )
    )
    return bool(result.scalar())


def _milvus_lookup(
    user_id: uuid.UUID,
    chat_id: uuid.UUID,
    collection_ids: list[str],
    look_for_conversation: bool,
) -> tuple[list[str], str | None]:
    """(collections the user cannot read, id of the chat's own collection).
    Synchronous (pymilvus): run it with asyncio.to_thread."""
    client = milvus.get_milvus_client()
    forbidden = [
        c for c in collection_ids if not milvus.user_can_read(client, c, user_id)
    ]
    conversation = (
        milvus.find_conversation_collection_id(client, user_id, chat_id)
        if look_for_conversation
        else None
    )
    return forbidden, conversation


async def validate_and_resolve_collections(
    db: AsyncSession,
    chat: Chat,
    payload: MessageCreate,
    user_id: uuid.UUID,
    litellm: LiteLLMClient,
) -> list[str]:
    """Checks the message against what the target (model or agent) supports and
    returns the Milvus collection ids to attach: the ones the user ticked, plus the
    chat's own collection if it has uploaded files. Raises HTTPException if invalid."""
    collection_ids = list(dict.fromkeys(payload.collection_ids or []))

    if payload.agent_id:
        if payload.images or payload.files or collection_ids:
            raise HTTPException(
                status_code=400,
                detail="Images, files and collections are not supported when sending to an agent",
            )
        return []

    model_id = payload.model_id
    capabilities = await litellm.get_model_capabilities(model_id)

    if payload.images:
        if not capabilities.get("supports_vision"):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' does not support image input",
            )
        total_bytes = sum(len(img.data) for img in payload.images) * 3 // 4
        if total_bytes > settings.max_images_total_bytes:
            raise HTTPException(status_code=400, detail="Images payload too large")

    supports_tools = bool(capabilities.get("supports_function_calling"))
    if collection_ids and not supports_tools:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' does not support tool calling, "
            "which is required to use collections",
        )

    chat_has_files = bool(payload.files) or await _chat_has_files(db, chat.id)
    if not (collection_ids or chat_has_files):
        return []

    forbidden, conversation = await asyncio.to_thread(
        _milvus_lookup, user_id, chat.id, collection_ids, chat_has_files
    )
    if forbidden:
        raise HTTPException(
            status_code=404,
            detail=f"Collections not found or not accessible: {forbidden}",
        )

    if conversation:
        if not supports_tools:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' does not support tool calling, "
                "which is required to query the files of this chat",
            )
        collection_ids = list(dict.fromkeys([*collection_ids, conversation]))
    elif payload.files:
        raise HTTPException(
            status_code=400,
            detail="This chat has no document collection to attach files to",
        )
    return collection_ids


# ==========================================
# Saving and loading messages
# ==========================================


async def save_user_message(
    db: AsyncSession, chat_id: uuid.UUID, payload: MessageCreate
) -> Message:
    extra: dict = {}
    if payload.images:
        extra["images"] = [img.model_dump() for img in payload.images]
    if payload.files:
        extra["files"] = [f.model_dump() for f in payload.files]

    message = Message(
        chat_id=chat_id, role=MessageRole.user, content=payload.content, extra=extra
    )
    db.add(message)
    await touch_chat(db, chat_id)
    await db.commit()
    await db.refresh(message)
    return message


def _to_llm_content(message: Message, include_images: bool) -> str | list[dict]:
    """`content` field for the LLM payload. Attached files are only mentioned by
    name (their content is in Milvus). Images are sent only for the message being
    answered now; older ones are replaced by a marker to keep the payload small."""
    text = message.content
    if message.files:
        names = ", ".join(f["filename"] for f in message.files)
        marker = f"[Attached files: {names}]"
        text = f"{text}\n{marker}" if text else marker

    images = message.images
    if not images:
        return text

    if not include_images:
        marker = f"[{len(images)} image(s) attached, not resent]"
        return f"{text}\n{marker}" if text else marker

    blocks: list[dict] = [{"type": "text", "text": text}] if text else []
    blocks += [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"},
        }
        for img in images
    ]
    return blocks


async def load_llm_messages(
    db: AsyncSession, chat_id: uuid.UUID, current_message_id: uuid.UUID
) -> list[dict]:
    """Full history of the chat in LLM format."""
    result = await db.execute(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at)
    )
    return [
        {
            "role": m.role.value,
            "content": _to_llm_content(m, include_images=(m.id == current_message_id)),
        }
        for m in result.scalars()
    ]


async def save_assistant_message(
    chat_id: uuid.UUID,
    model_id: str | None,
    agent_id: str | None,
    content: str,
    thinking: str,
) -> Message:
    """Saves the reply using its OWN session: it runs while (or after) the response
    is streaming, when the request's session must not be relied on."""
    async with async_session() as db:
        message = Message(
            chat_id=chat_id,
            role=MessageRole.assistant,
            content=content,
            model_id=model_id,
            agent_id=agent_id,
            extra={"thinking": thinking} if thinking else {},
        )
        db.add(message)
        await touch_chat(db, chat_id)
        await db.commit()
        await db.refresh(message)
        return message


# ==========================================
# Streaming
# ==========================================


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def open_stream(
    litellm: LiteLLMClient,
    chat_id: uuid.UUID,
    payload: MessageCreate,
    llm_messages: list[dict],
    collection_ids: list[str],
):
    """Async generator of (channel, text) from a model or from an agent."""
    if payload.model_id:
        return litellm.stream_chat_completion(
            model_id=payload.model_id,
            messages=llm_messages,
            collection_ids=collection_ids or None,
        )
    return litellm.stream_agent_run(
        chat_id=str(chat_id), agent_id=payload.agent_id, messages=llm_messages
    )


async def stream_reply(
    *,
    chat_id: uuid.UUID,
    model_id: str | None,
    agent_id: str | None,
    chunks,
) -> AsyncIterator[str]:
    """Turns the upstream chunks into SSE events (status / delta / done / error)
    and saves the assistant message.

    The message is saved in `finally`, so it is also saved when the user presses
    Stop (the request is cancelled) or the model fails halfway: what was already
    generated is kept. The save is shielded from cancellation, otherwise the
    cancelled request would cancel the DB write too.
    """
    content = thinking = ""
    completed = False
    error: str | None = None
    saved: Message | None = None

    try:
        async for channel, delta in chunks:
            if channel == "heartbeat":
                yield ": ping\n\n"  # keeps the connection alive (avoids timeouts)
            elif channel == "status":
                yield sse_event("status", {"detail": delta})
            elif channel == "thinking":
                thinking += delta
                yield sse_event("delta", {"thinking": delta})
            else:
                content += delta
                yield sse_event("delta", {"content": delta})
        completed = True
    except HTTPException as exc:
        error = str(exc.detail)
    except Exception:
        logger.exception("Unexpected error while streaming the reply")
        error = "The model request failed unexpectedly"
    finally:
        if completed or content or thinking:  # nothing to keep if it died empty
            with anyio.CancelScope(shield=True):
                saved = await save_assistant_message(
                    chat_id, model_id, agent_id, content, thinking
                )

    if error is not None:
        yield sse_event("error", {"detail": error})
        return

    done = {"message_id": str(saved.id), "created_at": saved.created_at.isoformat()}
    done["model_id" if model_id else "agent_id"] = model_id or agent_id
    yield sse_event("done", done)
