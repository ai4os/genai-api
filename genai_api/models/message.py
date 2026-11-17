import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, Enum, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from genai_api.db.base import Base

if TYPE_CHECKING:
    from .chat import Chat


class MessageRole(str, enum.Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        # An assistant message comes from exactly one source: a model OR an agent.
        # (Alembic does not detect CHECK constraints: changing this needs a manual migration.)
        CheckConstraint(
            "role <> 'assistant' OR num_nonnulls(model_id, agent_id) = 1",
            name="ck_messages_assistant_model_xor_agent",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chats.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole, name="message_role"))
    content: Mapped[str] = mapped_column(Text)

    model_id: Mapped[str | None] = mapped_column(nullable=True)
    agent_id: Mapped[str | None] = mapped_column(nullable=True)

    # Free-form payload: {"images": [...], "files": [...], "thinking": "..."}.
    # Future: tool_calls, citations.
    extra: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chat: Mapped["Chat"] = relationship(back_populates="messages")

    # The three properties below expose keys of `extra` as flat attributes, so
    # `MessageOut` (from_attributes=True) can serialise them directly.

    @property
    def images(self) -> list[dict] | None:
        return (self.extra or {}).get("images")

    @property
    def files(self) -> list[dict] | None:
        """References (filename, mimetype, size) to documents stored in the chat's
        Milvus collection. Only metadata is stored here, never the content."""
        return (self.extra or {}).get("files")

    @property
    def thinking(self) -> str | None:
        """Reasoning text, only on assistant messages where the model exposed it."""
        return (self.extra or {}).get("thinking")
