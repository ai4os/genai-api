import uuid
from datetime import datetime

from pydantic import BaseModel

from genai_api.schemas.message import MessageOut


class ChatCreate(BaseModel):
    model: str
    title: str | None = None


class ChatSummary(BaseModel):
    id: uuid.UUID
    title: str | None
    model: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatDetail(ChatSummary):
    config: dict
    messages: list[MessageOut]
