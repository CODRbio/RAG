"""add_paper_uid_columns

Revision ID: g3h4i5j6k7l8
Revises: f2a3b4c5d6e7
Create Date: 2026-03-14 10:00:00.000000

Adds paper_uid TEXT column (global unique paper identifier per ref_tools spec)
to paper_metadata, scholar_library_papers, and papers tables, plus indexes.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "g3h4i5j6k7l8"
down_revision: Union[str, Sequence[str], None] = "f2a3b4c5d6e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # paper_metadata
    op.add_column(
        "paper_metadata",
        sa.Column("paper_uid", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index("idx_pm_paper_uid", "paper_metadata", ["paper_uid"])

    # scholar_library_papers
    op.add_column(
        "scholar_library_papers",
        sa.Column("paper_uid", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index("idx_scholar_lib_papers_paper_uid", "scholar_library_papers", ["paper_uid"])

    # papers
    op.add_column(
        "papers",
        sa.Column("paper_uid", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index("idx_papers_paper_uid", "papers", ["paper_uid"])


def downgrade() -> None:
    op.drop_index("idx_papers_paper_uid", table_name="papers")
    op.drop_column("papers", "paper_uid")

    op.drop_index("idx_scholar_lib_papers_paper_uid", table_name="scholar_library_papers")
    op.drop_column("scholar_library_papers", "paper_uid")

    op.drop_index("idx_pm_paper_uid", table_name="paper_metadata")
    op.drop_column("paper_metadata", "paper_uid")
