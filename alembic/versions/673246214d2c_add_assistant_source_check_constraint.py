"""add assistant source check constraint

Revision ID: 673246214d2c
Revises: fe049d72baa1
Create Date: 2026-11-12 08:53:09.005587

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "673246214d2c"
down_revision: Union[str, Sequence[str], None] = "fe049d72baa1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NAME = "ck_messages_assistant_model_xor_agent"


def upgrade() -> None:
    op.create_check_constraint(
        NAME,
        "messages",
        sa.text("role <> 'assistant' OR num_nonnulls(model_id, agent_id) = 1"),
    )


def downgrade() -> None:
    op.drop_constraint(NAME, "messages", type_="check")
