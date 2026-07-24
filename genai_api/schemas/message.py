import uuid
from datetime import datetime

from pydantic import BaseModel

from genai_api.models.message import MessageRole


class MessageCreate(BaseModel):
    content: str
    model: str

class MessageOut(BaseModel):
    id: uuid.UUID
    role: MessageRole
    content: str
    model: str | None
    created_at: datetime

    class Config:
        from_attributes = True
