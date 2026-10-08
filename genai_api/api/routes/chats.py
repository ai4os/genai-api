import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, exists, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from genai_api.api.deps import get_current_user_id, get_user_litellm_client
from genai_api.api.routes.knowledge import (
    delete_conversation_collections,
    find_conversation_collection_id,
    get_milvus_client,
    is_user_collection_accessible,
)
from genai_api.core.config import settings
from genai_api.db.session import get_db
from genai_api.models.chat import Chat
from genai_api.models.message import Message, MessageRole
from genai_api.schemas.chat import ChatCreate, ChatDetail, ChatSummary
from genai_api.schemas.message import MessageCreate
from genai_api.services.litellm_client import LiteLLMClient

router = APIRouter(prefix="/chats", tags=["chats"])


# ==========================================
# Helpers
# =========================================


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _to_llm_content(message: Message, include_images: bool) -> str | list[dict]:
    """
    Builds the `content` field for the LiteLLM payload from a saved Message.
    If the message contains images and `include_images` is True, it returns
    a list of OpenAI-style blocks (text + image_url). Otherwise, plain text.
    Attached files are only mentioned by name; their content is in Milvus.
    """
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

    blocks = []
    if text:
        blocks.append({"type": "text", "text": text})
    for img in images:
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"},
            }
        )
    return blocks


async def _find_inaccessible_collections(
    collection_ids: list[str], user_id: uuid.UUID
) -> list[str]:
    """
    Returns the collections that do not exist or that the user can neither own
    nor read (public). pymilvus is synchronous, so it runs in a thread to avoid
    blocking the event loop.
    """

    def _check() -> list[str]:
        client = get_milvus_client()
        return [
            c
            for c in collection_ids
            if not is_user_collection_accessible(client, c, user_id)
        ]

    return await asyncio.to_thread(_check)


async def _chat_has_files(db: AsyncSession, chat_id: uuid.UUID) -> bool:
    """True if any message of the chat references uploaded files (extra.files)."""
    result = await db.execute(
        select(
            exists().where(Message.chat_id == chat_id, Message.extra.has_key("files"))
        )
    )
    return bool(result.scalar())


async def _find_conversation_collection(
    user_id: uuid.UUID, chat_id: uuid.UUID
) -> str | None:
    """Collection of type `conversation` linked to the chat."""
    return await asyncio.to_thread(
        lambda: find_conversation_collection_id(get_milvus_client(), user_id, chat_id)
    )


# ==========================================
# Routes
# ==========================================


@router.post("", response_model=ChatSummary, status_code=201)
async def create_chat(
    payload: ChatCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    chat = Chat(user_id=user_id, model_id=payload.model_id, title=payload.title)
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat


@router.delete("/{chat_id}", status_code=204)
async def delete_chat(
    chat_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    # The chat must exist and belong to the user
    result = await db.execute(
        select(Chat.id).where(Chat.id == chat_id, Chat.user_id == user_id)
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Drop the Milvus collections first. If this fails, the chat is kept
    # and the client can retry (to avoid zombie collections)
    try:
        await asyncio.to_thread(
            lambda: delete_conversation_collections(
                get_milvus_client(), user_id, chat_id
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not delete the chat's documents. The chat was not deleted",
        ) from exc

    # Delete the chat
    result = await db.execute(
        delete(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Chat not found")

    await db.commit()


@router.get("", response_model=list[ChatSummary])
async def list_chats(
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Chat).where(Chat.user_id == user_id).order_by(Chat.updated_at.desc())
    )
    return result.scalars().all()


@router.get("/{chat_id}", response_model=ChatDetail)
async def get_chat(
    chat_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Chat)
        .options(selectinload(Chat.messages))
        .where(Chat.id == chat_id, Chat.user_id == user_id)
    )
    chat = result.scalar_one_or_none()

    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.post("/{chat_id}/messages")
async def send_message(
    chat_id: uuid.UUID,
    payload: MessageCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    litellm_client: LiteLLMClient = Depends(get_user_litellm_client),
):
    """
    Sends a message and streams the response (SSE), either from a model
    or from an agent depending on the payload.

    Events emitted:
    - `status` {"detail": "searching"}  (the model is querying the collections)
    - `delta` {"content"?: str, "thinking"?: str}
    - `done`  {"message_id", "created_at", "model_id"?, "agent_id"?}
    - `error` {"detail"}
    """
    if bool(payload.model_id) == bool(payload.agent_id):
        raise HTTPException(
            status_code=400,
            detail="Exactly one of model_id or agent_id is required",
        )

    # Milvus knowledge collections ids
    collection_ids = list(dict.fromkeys(payload.collection_ids or []))

    if payload.agent_id and (payload.images or payload.files or collection_ids):
        raise HTTPException(
            status_code=400,
            detail="Images, files and collections are not supported when sending to an agent",
        )

    if payload.images or collection_ids:
        capabilities = await litellm_client.get_model_capabilities(payload.model_id)

        # Image management
        if payload.images:
            if not capabilities.get("supports_vision"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{payload.model_id}' does not support image input",
                )

            total_bytes = sum(len(img.data) for img in payload.images) * 3 // 4
            if total_bytes > settings.max_images_total_bytes:
                raise HTTPException(status_code=400, detail="Images payload too large")

        # Milvus MCP tool
        if collection_ids:
            if not capabilities.get("supports_function_calling"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{payload.model_id}' does not support tool calling, "
                    "which is required to use collections",
                )
            forbidden = await _find_inaccessible_collections(collection_ids, user_id)
            if forbidden:
                raise HTTPException(
                    status_code=404,
                    detail=f"Collections not found or not accessible: {forbidden}",
                )

    result = await db.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
    )
    chat = result.scalar_one_or_none()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    if payload.model_id and (payload.files or await _chat_has_files(db, chat.id)):
        conversation_collection = await _find_conversation_collection(user_id, chat.id)
        if conversation_collection:
            capabilities = await litellm_client.get_model_capabilities(payload.model_id)
            if not capabilities.get("supports_function_calling"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{payload.model_id}' does not support tool calling, "
                    "which is required to query the files of this chat",
                )
            collection_ids = list(
                dict.fromkeys([*collection_ids, conversation_collection])
            )
        elif payload.files:
            raise HTTPException(
                status_code=400,
                detail="This chat has no document collection to attach files to",
            )

    # Save user message
    user_message = Message(
        chat_id=chat.id,
        role=MessageRole.user,
        content=payload.content,
        extra={
            **(
                {"images": [img.model_dump() for img in payload.images]}
                if payload.images
                else {}
            ),
            **(
                {"files": [f.model_dump() for f in payload.files]}
                if payload.files
                else {}
            ),
        },
    )
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)

    # Add message history for context
    history_result = await db.execute(
        select(Message).where(Message.chat_id == chat.id).order_by(Message.created_at)
    )
    history = history_result.scalars().all()
    llm_messages = [
        {
            "role": m.role.value,
            "content": _to_llm_content(m, include_images=(m.id == user_message.id)),
        }
        for m in history
    ]

    if payload.model_id:
        response_generator = litellm_client.stream_chat_completion(
            model_id=payload.model_id,
            messages=llm_messages,
            collection_ids=collection_ids or None,
        )
    else:
        response_generator = litellm_client.stream_agent_run(
            chat_id=str(chat_id),
            agent_id=payload.agent_id,
            messages=llm_messages,
        )

    # 4. Flujo de eventos SSE
    async def event_stream():
        full_content = ""
        full_thinking = ""
        try:
            async for channel, delta in response_generator:
                if channel == "heartbeat":
                    yield ": ping\n\n"  # to keep the conection alive (avoid timeouts)
                    continue
                if channel == "status":
                    yield _sse_event("status", {"detail": delta})
                elif channel == "thinking":
                    full_thinking += delta
                    yield _sse_event("delta", {"thinking": delta})
                else:
                    full_content += delta
                    yield _sse_event("delta", {"content": delta})
        except HTTPException as exc:
            yield _sse_event("error", {"detail": exc.detail})
            return

        assistant_message = Message(
            chat_id=chat.id,
            role=MessageRole.assistant,
            content=full_content,
            model_id=payload.model_id,
            agent_id=payload.agent_id,
            extra=({"thinking": full_thinking} if full_thinking else {}),
        )
        db.add(assistant_message)
        await db.commit()
        await db.refresh(assistant_message)

        done_data = {
            "message_id": str(assistant_message.id),
            "created_at": assistant_message.created_at.isoformat(),
        }
        if payload.model_id:
            done_data["model_id"] = payload.model_id
        else:
            done_data["agent_id"] = payload.agent_id

        yield _sse_event("done", done_data)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
