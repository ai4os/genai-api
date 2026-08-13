"""add agent_id to messages

Revision ID: 3c662fe5d15d
Revises: 1d0ae0ec2b02
Create Date: 2026-07-31 10:03:46.859593

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3c662fe5d15d"
down_revision: Union[str, Sequence[str], None] = "1d0ae0ec2b02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("messages", sa.Column("agent_id", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("messages", "agent_id")
