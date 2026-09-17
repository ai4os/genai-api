from enum import Enum
from pydantic import BaseModel, Field


class CollectionType(str, Enum):
    NONE = "none"
    CONVERSATION = "conversation"
    KNOWLEDGE_BASE = "knowledge_base"


class CollectionScope(str, Enum):
    ALL = "all"
    PUBLIC = "public"
    PRIVATE = "private"


class CollectionCreate(BaseModel):
    name: str = Field(..., description="Name of the collection to create", min_length=1)
    description: str | None = Field(
        default=None,
        description="Optional description of the collection",
    )
    type: CollectionType = Field(
        default=CollectionType.KNOWLEDGE_BASE,
        description="Type of collection: conversation or knowledge_base",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation/chat ID if collection is of type conversation",
    )


class CollectionInfo(BaseModel):
    id: str = Field(..., description="Unique Milvus collection ID (UUID)")
    name: str | None = Field(None, description="User-assigned name of the collection")
    description: str | None = Field(None, description="Description of the collection")
    scope: str
    type: CollectionType | None = None
    conversation_id: str | None = Field(
        None, description="Conversation/chat ID if associated with a conversation"
    )
    owner: str | None = None


class DocumentUploadResponse(BaseModel):
    filename: str
    collection_id: str
    chunks_count: int
    status: str


class CollectionDeleteResponse(BaseModel):
    collection_id: str
    message: str
