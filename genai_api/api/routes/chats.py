import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from genai_api.api.deps import get_current_user_id
from genai_api.db.session import get_db
from genai_api.models.chat import Chat
from genai_api.models.message import Message, MessageRole
from genai_api.schemas.chat import ChatCreate, ChatDetail, ChatSummary
from genai_api.schemas.message import MessageCreate, MessageOut

router = APIRouter(prefix="/chats", tags=["chats"])


@router.post("", response_model=ChatSummary, status_code=201)
async def create_chat(
    payload: ChatCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    chat = Chat(user_id=user_id, model=payload.model, title=payload.title)
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


@router.post("/{chat_id}/messages", response_model=MessageOut, status_code=201)
async def send_message(
    chat_id: uuid.UUID,
    payload: MessageCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
    chat = result.scalar_one_or_none()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    user_message = Message(chat_id=chat.id, role=MessageRole.user, content=payload.content)
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)

    # TODO: connect with LiteLLM and store assistant's reponse.

    return user_message
