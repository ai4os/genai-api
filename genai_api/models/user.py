import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from genai_api.db.base import Base

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chat import Chat


class User(Base):
    """User authenticated in Keycloak. id == 'sub' in the token."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    email: Mapped[str | None] = mapped_column(default=None)
    username: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    litellm_api_key: Mapped[str | None] = mapped_column(default=None)

    chats: Mapped[list["Chat"]] = relationship(back_populates="user")
