import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from genai_api.models.message import MessageRole


class ImageInput(BaseModel):
    mime_type: str
    data: str  # base64, without the "data:...;base64," prefix


class ImageOut(BaseModel):
    mime_type: str
    data: str


class FileRef(BaseModel):
    """Reference to a document already uploaded to the chat's Milvus collection.
    Only metadata travels here, never the content."""

    filename: str
    mimetype: str | None = None
    size: int | None = None


class MessageCreate(BaseModel):
    content: str
    # Exactly one of model_id / agent_id (validated in the endpoint)
    model_id: str | None = None
    agent_id: str | None = None
    images: list[ImageInput] | None = None
    files: list[FileRef] | None = None
    collection_ids: list[str] | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    thinking: str | None = None  # Message.extra["thinking"]
    model_id: str | None
    agent_id: str | None = None
    images: list[ImageOut] | None = None  # Message.extra["images"]
    files: list[FileRef] | None = None  # Message.extra["files"]
    created_at: datetime
