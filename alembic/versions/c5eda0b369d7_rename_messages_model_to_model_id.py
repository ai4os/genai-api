"""rename messages.model to model_id

Revision ID: c5eda0b369d7
Revises: 3c662fe5d15d
Create Date: 2026-07-31 10:13:46.655987

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "c5eda0b369d7"
down_revision: Union[str, Sequence[str], None] = "3c662fe5d15d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column("messages", "model", new_column_name="model_id")


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column("messages", "model_id", new_column_name="model")
