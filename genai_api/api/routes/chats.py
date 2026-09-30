import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from genai_api.api.deps import get_current_user_id, get_user_litellm_client
from genai_api.api.routes.knowledge import (
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


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _to_llm_content(message: Message, include_images: bool) -> str | list[dict]:
    """
    Builds the `content` field for the LiteLLM payload from a saved Message.
    If the message contains images and `include_images` is True, it returns
    a list of OpenAI-style blocks (text + image_url). Otherwise, plain text .
    """
    images = message.images
    if not images:
        return message.content

    if not include_images:
        marker = f"[{len(images)} image(s) attached, not resent]"
        return f"{message.content}\n{marker}" if message.content else marker

    blocks = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
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

    if payload.agent_id and (payload.images or collection_ids):
        raise HTTPException(
            status_code=400,
            detail="Images and collections are not supported when sending to an agent",
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

    # Save user message
    user_message = Message(
        chat_id=chat.id,
        role=MessageRole.user,
        content=payload.content,
        extra=(
            {"images": [img.model_dump() for img in payload.images]}
            if payload.images
            else {}
        ),
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
