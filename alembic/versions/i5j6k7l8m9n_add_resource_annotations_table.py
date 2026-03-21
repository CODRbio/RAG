"""add_resource_annotations_table

Revision ID: i5j6k7l8m9n
Revises: h4i5j6k7l8m
Create Date: 2026-03-14 23:30:00.000000

Adds resource_annotations for Phase 3 academic assistant annotations.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "i5j6k7l8m9n"
down_revision: Union[str, Sequence[str], None] = "h4i5j6k7l8m"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "resource_annotations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        sa.Column("resource_type", sa.Text(), nullable=False, server_default=""),
        sa.Column("resource_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("paper_uid", sa.Text(), nullable=False, server_default=""),
        sa.Column("target_kind", sa.Text(), nullable=False, server_default="chunk"),
        sa.Column("target_locator_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("target_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("directive", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.Text(), nullable=False, server_default="active"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_resource_annotations_user_resource",
        "resource_annotations",
        ["user_id", "resource_type", "resource_id"],
    )
    op.create_index(
        "idx_resource_annotations_paper_uid",
        "resource_annotations",
        ["paper_uid"],
    )
    op.create_index(
        "idx_resource_annotations_target_kind",
        "resource_annotations",
        ["target_kind"],
    )


def downgrade() -> None:
    op.drop_index("idx_resource_annotations_target_kind", table_name="resource_annotations")
    op.drop_index("idx_resource_annotations_paper_uid", table_name="resource_annotations")
    op.drop_index("idx_resource_annotations_user_resource", table_name="resource_annotations")
    op.drop_table("resource_annotations")
