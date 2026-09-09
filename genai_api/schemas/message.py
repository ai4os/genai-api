import uuid
from datetime import datetime
from pydantic import BaseModel
from genai_api.models.message import MessageRole


class ImageInput(BaseModel):
    mime_type: str
    data: str  # base64, sin el prefijo "data:...;base64,"


class ImageOut(BaseModel):
    mime_type: str
    data: str


class MessageCreate(BaseModel):
    content: str
    model_id: str | None = None
    agent_id: str | None = None
    images: list[ImageInput] | None = None


class MessageOut(BaseModel):
    id: uuid.UUID
    role: MessageRole
    content: str
    thinking: str | None = None  # Message.extra['thinking']
    model_id: str | None
    agent_id: str | None = None
    images: list[ImageOut] | None = None
    created_at: datetime

    class Config:
        from_attributes = True
