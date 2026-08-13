"""rename chats.model to model_id, make nullable

Revision ID: fe049d72baa1
Revises: c5eda0b369d7
Create Date: 2026-07-31 10:36:20.765276

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "fe049d72baa1"
down_revision: Union[str, Sequence[str], None] = "c5eda0b369d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column("chats", "model", new_column_name="model_id")
    op.alter_column("chats", "model_id", nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column("chats", "model_id", nullable=False)
    op.alter_column("chats", "model_id", new_column_name="model")
