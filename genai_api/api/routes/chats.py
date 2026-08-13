import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from genai_api.api.deps import get_current_user_id, get_user_litellm_client
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
    - `delta` {"content": str}
    - `done`  {"message_id", "created_at", "model_id"?, "agent_id"?}
    - `error` {"detail"}
    """
    # Exclusividad: exactamente uno de los dos
    if bool(payload.model_id) == bool(payload.agent_id):
        raise HTTPException(
            status_code=400,
            detail="Exactly one of model_id or agent_id is required",
        )

    result = await db.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
    )
    chat = result.scalar_one_or_none()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Save user message
    user_message = Message(
        chat_id=chat.id, role=MessageRole.user, content=payload.content
    )
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)

    # Add message history for context
    history_result = await db.execute(
        select(Message).where(Message.chat_id == chat.id).order_by(Message.created_at)
    )
    history = history_result.scalars().all()
    llm_messages = [{"role": m.role.value, "content": m.content} for m in history]

    if payload.model_id:
        response_generator = litellm_client.stream_chat_completion(
            model_id=payload.model_id, messages=llm_messages
        )
    else:
        response_generator = litellm_client.stream_agent_run(
            agent_id=payload.agent_id, messages=llm_messages
        )

    # 4. Flujo de eventos SSE
    async def event_stream():
        full_content = ""
        try:
            async for delta in response_generator:
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
