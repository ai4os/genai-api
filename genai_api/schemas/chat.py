import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from genai_api.schemas.message import MessageOut


class ChatCreate(BaseModel):
    model_id: str | None = None
    title: str | None = None


class ChatSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    model_id: str | None
    created_at: datetime
    updated_at: datetime


class ChatDetail(ChatSummary):
    config: dict
    messages: list[MessageOut]
