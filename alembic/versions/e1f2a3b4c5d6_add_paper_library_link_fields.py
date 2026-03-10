"""add_paper_library_link_fields

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-03-08 14:00:00.000000

Adds persistent link between collection papers (papers) and scholar library papers
(scholar_library_papers): library_id, library_paper_id, source on papers;
collection_name, collection_paper_id on scholar_library_papers.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, Sequence[str], None] = "d0e1f2a3b4c5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(inspector: sa.Inspector, table: str, column: str) -> bool:
    if not inspector.has_table(table):
        return False
    return column in {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("papers"):
        if not _has_column(inspector, "papers", "library_id"):
            op.add_column("papers", sa.Column("library_id", sa.Integer(), nullable=True))
        if not _has_column(inspector, "papers", "library_paper_id"):
            op.add_column("papers", sa.Column("library_paper_id", sa.Integer(), nullable=True))
        if not _has_column(inspector, "papers", "source"):
            op.add_column(
                "papers",
                sa.Column("source", sa.Text(), nullable=False, server_default=""),
            )
    if inspector.has_table("scholar_library_papers"):
        if not _has_column(inspector, "scholar_library_papers", "collection_name"):
            op.add_column(
                "scholar_library_papers",
                sa.Column("collection_name", sa.Text(), nullable=False, server_default=""),
            )
        if not _has_column(inspector, "scholar_library_papers", "collection_paper_id"):
            op.add_column(
                "scholar_library_papers",
                sa.Column("collection_paper_id", sa.Text(), nullable=False, server_default=""),
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("papers"):
        if _has_column(inspector, "papers", "source"):
            op.drop_column("papers", "source")
        if _has_column(inspector, "papers", "library_paper_id"):
            op.drop_column("papers", "library_paper_id")
        if _has_column(inspector, "papers", "library_id"):
            op.drop_column("papers", "library_id")
    if inspector.has_table("scholar_library_papers"):
        if _has_column(inspector, "scholar_library_papers", "collection_paper_id"):
            op.drop_column("scholar_library_papers", "collection_paper_id")
        if _has_column(inspector, "scholar_library_papers", "collection_name"):
            op.drop_column("scholar_library_papers", "collection_name")
