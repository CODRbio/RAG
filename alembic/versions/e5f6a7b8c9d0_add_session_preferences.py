"""add_session_preferences

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-02-27

Session-level preferences (e.g. local_db: no_local) for "query vs collection mismatch" flow.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("sessions"):
        return
    cols = {c["name"] for c in inspector.get_columns("sessions")}
    if "preferences" not in cols:
        op.add_column(
            "sessions",
            sa.Column("preferences", sa.Text(), nullable=False, server_default="{}"),
        )


def downgrade() -> None:
    op.drop_column("sessions", "preferences")
