import uuid
from datetime import datetime
from pydantic import BaseModel
from genai_api.models.message import MessageRole


class MessageCreate(BaseModel):
    content: str
    model_id: str | None = None
    agent_id: str | None = None


class MessageOut(BaseModel):
    id: uuid.UUID
    role: MessageRole
    content: str
    model_id: str | None
    agent_id: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True
