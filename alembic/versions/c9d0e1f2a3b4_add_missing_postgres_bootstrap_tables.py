"""add_missing_postgres_bootstrap_tables

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-03-07 00:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "c9d0e1f2a3b4"
down_revision: Union[str, Sequence[str], None] = "b8c9d0e1f2a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    return any(col.get("name") == column_name for col in inspector.get_columns(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if inspector.has_table("canvases") and not _has_column(inspector, "canvases", "preliminary_knowledge"):
        op.add_column(
            "canvases",
            sa.Column("preliminary_knowledge", sa.Text(), nullable=False, server_default=""),
        )

    if inspector.has_table("scholar_library_papers") and not _has_column(
        inspector, "scholar_library_papers", "normalized_journal_name"
    ):
        op.add_column(
            "scholar_library_papers",
            sa.Column("normalized_journal_name", sa.Text(), nullable=False, server_default=""),
        )

    if not inspector.has_table("impact_factor_journals"):
        op.create_table(
            "impact_factor_journals",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("source_file", sa.Text(), nullable=False, server_default=""),
            sa.Column("source_version", sa.Text(), nullable=False, server_default=""),
            sa.Column("journal_name", sa.Text(), nullable=False, server_default=""),
            sa.Column("normalized_journal_name", sa.Text(), nullable=False, server_default=""),
            sa.Column("jcr_abbreviation", sa.Text(), nullable=False, server_default=""),
            sa.Column("normalized_jcr_abbreviation", sa.Text(), nullable=False, server_default=""),
            sa.Column("issn", sa.Text(), nullable=False, server_default=""),
            sa.Column("eissn", sa.Text(), nullable=False, server_default=""),
            sa.Column("category", sa.Text(), nullable=False, server_default=""),
            sa.Column("edition", sa.Text(), nullable=False, server_default=""),
            sa.Column("impact_factor", sa.Float(), nullable=True),
            sa.Column("jif_quartile", sa.Text(), nullable=False, server_default=""),
            sa.Column("jif_rank", sa.Text(), nullable=False, server_default=""),
            sa.Column("jif_5year", sa.Float(), nullable=True),
            sa.Column("publisher", sa.Text(), nullable=False, server_default=""),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("idx_ifj_normalized_name", "impact_factor_journals", ["normalized_journal_name"], unique=False)
        op.create_index("idx_ifj_normalized_abbr", "impact_factor_journals", ["normalized_jcr_abbreviation"], unique=False)
        op.create_index("idx_ifj_issn", "impact_factor_journals", ["issn"], unique=False)
        op.create_index("idx_ifj_eissn", "impact_factor_journals", ["eissn"], unique=False)

    if not inspector.has_table("impact_factor_index_meta"):
        op.create_table(
            "impact_factor_index_meta",
            sa.Column("source_file", sa.Text(), nullable=False),
            sa.Column("last_mtime", sa.Float(), nullable=False, server_default="0"),
            sa.Column("last_size", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_hash", sa.Text(), nullable=False, server_default=""),
            sa.Column("indexed_at", sa.Text(), nullable=False, server_default=""),
            sa.Column("row_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
            sa.PrimaryKeyConstraint("source_file"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if inspector.has_table("impact_factor_index_meta"):
        op.drop_table("impact_factor_index_meta")

    if inspector.has_table("impact_factor_journals"):
        index_names = {idx.get("name") for idx in inspector.get_indexes("impact_factor_journals")}
        if "idx_ifj_eissn" in index_names:
            op.drop_index("idx_ifj_eissn", table_name="impact_factor_journals")
        if "idx_ifj_issn" in index_names:
            op.drop_index("idx_ifj_issn", table_name="impact_factor_journals")
        if "idx_ifj_normalized_abbr" in index_names:
            op.drop_index("idx_ifj_normalized_abbr", table_name="impact_factor_journals")
        if "idx_ifj_normalized_name" in index_names:
            op.drop_index("idx_ifj_normalized_name", table_name="impact_factor_journals")
        op.drop_table("impact_factor_journals")

    if inspector.has_table("scholar_library_papers") and _has_column(
        inspector, "scholar_library_papers", "normalized_journal_name"
    ):
        op.drop_column("scholar_library_papers", "normalized_journal_name")

    if inspector.has_table("canvases") and _has_column(inspector, "canvases", "preliminary_knowledge"):
        op.drop_column("canvases", "preliminary_knowledge")
