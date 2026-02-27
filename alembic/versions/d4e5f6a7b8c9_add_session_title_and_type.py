"""add_session_title_and_type

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-02-27

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("sessions"):
        return
    cols = {c["name"] for c in inspector.get_columns("sessions")}
    if "title" not in cols:
        op.add_column("sessions", sa.Column("title", sa.Text(), nullable=False, server_default=""))
    if "session_type" not in cols:
        op.add_column("sessions", sa.Column("session_type", sa.Text(), nullable=False, server_default="chat"))


def downgrade() -> None:
    op.drop_column("sessions", "session_type")
    op.drop_column("sessions", "title")
