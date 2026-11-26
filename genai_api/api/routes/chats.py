"""Chats: HTTP layer only. The logic lives in services/chat_service.py."""

import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from genai_api.api.deps import get_current_user_id, get_user_litellm_client
from genai_api.db.session import get_db
from genai_api.models.chat import Chat
from genai_api.schemas.chat import ChatCreate, ChatDetail, ChatSummary
from genai_api.schemas.message import MessageCreate
from genai_api.services import chat_service
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
    return await chat_service.get_user_chat(db, chat_id, user_id, with_messages=True)


@router.delete("/{chat_id}", status_code=204)
async def delete_chat(
    chat_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    await chat_service.delete_chat(db, chat_id, user_id)


@router.delete("", status_code=204)
async def delete_chats(
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Chat).where(Chat.user_id == user_id))
    for chat in result.scalars().all():
        await chat_service.delete_chat(db, chat.id, user_id)


@router.post("/{chat_id}/messages")
async def send_message(
    chat_id: uuid.UUID,
    payload: MessageCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    litellm: LiteLLMClient = Depends(get_user_litellm_client),
):
    """
    Sends a message to a model or an agent and streams the reply (SSE).

    Events:
    - `status` {"detail": "searching"}   the model is querying the collections
    - `delta`  {"content"?: str, "thinking"?: str}
    - `done`   {"message_id", "created_at", "model_id"?, "agent_id"?}
    - `error`  {"detail"}
    """
    chat_service.check_single_target(payload)
    chat = await chat_service.get_user_chat(db, chat_id, user_id)
    collection_ids = await chat_service.validate_and_resolve_collections(
        db, chat, payload, user_id, litellm
    )

    user_message = await chat_service.save_user_message(db, chat.id, payload)
    llm_messages = await chat_service.load_llm_messages(db, chat.id, user_message.id)
    # End the read transaction now so the DB connection goes back to the pool
    # instead of staying checked out for the whole (possibly long) streaming.
    await db.commit()

    chunks = chat_service.open_stream(
        litellm, chat.id, payload, llm_messages, collection_ids
    )
    return StreamingResponse(
        chat_service.stream_reply(
            chat_id=chat.id,
            model_id=payload.model_id,
            agent_id=payload.agent_id,
            chunks=chunks,
        ),
        media_type="text/event-stream",
    )
