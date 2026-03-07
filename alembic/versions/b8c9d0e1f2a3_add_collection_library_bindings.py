"""add_collection_library_bindings

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-03-07 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "b8c9d0e1f2a3"
down_revision: Union[str, Sequence[str], None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("collection_library_bindings"):
        op.create_table(
            "collection_library_bindings",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
            sa.Column("collection_name", sa.Text(), nullable=False, server_default=""),
            sa.Column("library_id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
            sa.ForeignKeyConstraint(["library_id"], ["scholar_libraries.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "collection_name", name="uq_collection_library_user_collection"),
            sa.UniqueConstraint("user_id", "library_id", name="uq_collection_library_user_library"),
        )
    index_names = {idx.get("name") for idx in inspector.get_indexes("collection_library_bindings")}
    if "idx_collection_library_user_collection" not in index_names:
        op.create_index(
            "idx_collection_library_user_collection",
            "collection_library_bindings",
            ["user_id", "collection_name"],
            unique=False,
        )
    if "idx_collection_library_user_library" not in index_names:
        op.create_index(
            "idx_collection_library_user_library",
            "collection_library_bindings",
            ["user_id", "library_id"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("collection_library_bindings"):
        index_names = {idx.get("name") for idx in inspector.get_indexes("collection_library_bindings")}
        if "idx_collection_library_user_library" in index_names:
            op.drop_index("idx_collection_library_user_library", table_name="collection_library_bindings")
        if "idx_collection_library_user_collection" in index_names:
            op.drop_index("idx_collection_library_user_collection", table_name="collection_library_bindings")
        op.drop_table("collection_library_bindings")
