"""add_resource_state_tables

Revision ID: j6k7l8m9n0p
Revises: i5j6k7l8m9n
Create Date: 2026-03-15 09:10:00.000000

Adds resource_user_states/resource_tags/resource_notes for Phase 4.
Backfills archived canvases into resource_user_states.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "j6k7l8m9n0p"
down_revision: Union[str, Sequence[str], None] = "i5j6k7l8m9n"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "resource_user_states",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        sa.Column("resource_type", sa.Text(), nullable=False, server_default=""),
        sa.Column("resource_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("favorite", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("archived", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("read_status", sa.Text(), nullable=False, server_default="unread"),
        sa.Column("last_opened_at", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "resource_type",
            "resource_id",
            name="uq_resource_user_states_user_resource",
        ),
    )
    op.create_index(
        "idx_resource_user_states_user_type_archived",
        "resource_user_states",
        ["user_id", "resource_type", "archived", "updated_at"],
    )
    op.create_index(
        "idx_resource_user_states_user_type_favorite",
        "resource_user_states",
        ["user_id", "resource_type", "favorite", "updated_at"],
    )

    op.create_table(
        "resource_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        sa.Column("resource_type", sa.Text(), nullable=False, server_default=""),
        sa.Column("resource_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("tag", sa.Text(), nullable=False, server_default=""),
        sa.Column("normalized_tag", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "resource_type",
            "resource_id",
            "normalized_tag",
            name="uq_resource_tags_user_resource_tag",
        ),
    )
    op.create_index(
        "idx_resource_tags_user_resource",
        "resource_tags",
        ["user_id", "resource_type", "resource_id"],
    )

    op.create_table(
        "resource_notes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        sa.Column("resource_type", sa.Text(), nullable=False, server_default=""),
        sa.Column("resource_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("note_md", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_resource_notes_user_resource",
        "resource_notes",
        ["user_id", "resource_type", "resource_id", "updated_at"],
    )

    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            INSERT INTO resource_user_states (
                user_id, resource_type, resource_id, favorite, archived, read_status,
                last_opened_at, created_at, updated_at
            )
            SELECT
                CASE
                    WHEN user_id IS NULL OR TRIM(user_id) = '' THEN 'default'
                    ELSE user_id
                END,
                'canvas',
                id,
                0,
                1,
                'unread',
                NULL,
                COALESCE(CAST(updated_at AS TEXT), CAST(created_at AS TEXT), CAST(CURRENT_TIMESTAMP AS TEXT)),
                COALESCE(CAST(updated_at AS TEXT), CAST(created_at AS TEXT), CAST(CURRENT_TIMESTAMP AS TEXT))
            FROM canvases
            WHERE COALESCE(archived, 0) = 1
            """
        )
    )


def downgrade() -> None:
    op.drop_index("idx_resource_notes_user_resource", table_name="resource_notes")
    op.drop_table("resource_notes")

    op.drop_index("idx_resource_tags_user_resource", table_name="resource_tags")
    op.drop_table("resource_tags")

    op.drop_index("idx_resource_user_states_user_type_favorite", table_name="resource_user_states")
    op.drop_index("idx_resource_user_states_user_type_archived", table_name="resource_user_states")
    op.drop_table("resource_user_states")
