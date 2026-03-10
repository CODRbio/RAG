"""add_crossref_cache_by_doi

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-03-08 12:00:00.000000

Adds crossref_cache_by_doi table for DOI-keyed Crossref lookups (backward-compatible).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "d0e1f2a3b4c5"
down_revision: Union[str, Sequence[str], None] = "c9d0e1f2a3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("crossref_cache_by_doi"):
        return
    op.create_table(
        "crossref_cache_by_doi",
        sa.Column("normalized_doi", sa.Text(), nullable=False),
        sa.Column("doi", sa.Text(), nullable=False, server_default=""),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column("authors", sa.Text(), nullable=False, server_default=""),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("venue", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("normalized_doi"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("crossref_cache_by_doi"):
        op.drop_table("crossref_cache_by_doi")
