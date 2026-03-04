"""add_scholar_library_paper_downloaded_at

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: Literature catalog: track when PDF was downloaded.

Add downloaded_at (TEXT, nullable) to scholar_library_papers.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, Sequence[str], None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(inspector: sa.Inspector, table: str, column: str) -> bool:
    if not inspector.has_table(table):
        return False
    return column in {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("scholar_library_papers"):
        return
    if _has_column(inspector, "scholar_library_papers", "downloaded_at"):
        return
    op.add_column(
        "scholar_library_papers",
        sa.Column("downloaded_at", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("scholar_library_papers", "downloaded_at")
