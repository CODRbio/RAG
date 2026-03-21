"""add_global_graph_tables

Revision ID: h4i5j6k7l8m
Revises: g3h4i5j6k7l8
Create Date: 2026-03-14 22:30:00.000000

Adds persistent graph_facts / graph_snapshots tables for GlobalGraphService.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "h4i5j6k7l8m"
down_revision: Union[str, Sequence[str], None] = "g3h4i5j6k7l8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "graph_facts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        sa.Column("scope_type", sa.Text(), nullable=False, server_default="global"),
        sa.Column("scope_key", sa.Text(), nullable=False, server_default="global"),
        sa.Column("graph_type", sa.Text(), nullable=False, server_default=""),
        sa.Column("src_node_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("src_node_type", sa.Text(), nullable=False, server_default=""),
        sa.Column("src_label", sa.Text(), nullable=False, server_default=""),
        sa.Column("relation_type", sa.Text(), nullable=False, server_default=""),
        sa.Column("dst_node_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("dst_node_type", sa.Text(), nullable=False, server_default=""),
        sa.Column("dst_label", sa.Text(), nullable=False, server_default=""),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1"),
        sa.Column("provenance_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "scope_type",
            "scope_key",
            "graph_type",
            "src_node_id",
            "relation_type",
            "dst_node_id",
            name="uq_graph_facts_scope_edge",
        ),
    )
    op.create_index(
        "idx_graph_facts_scope_graph",
        "graph_facts",
        ["user_id", "scope_type", "scope_key", "graph_type"],
    )
    op.create_index("idx_graph_facts_src", "graph_facts", ["graph_type", "src_node_id"])
    op.create_index("idx_graph_facts_dst", "graph_facts", ["graph_type", "dst_node_id"])

    op.create_table(
        "graph_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        sa.Column("scope_type", sa.Text(), nullable=False, server_default="global"),
        sa.Column("scope_key", sa.Text(), nullable=False, server_default="global"),
        sa.Column("graph_type", sa.Text(), nullable=False, server_default=""),
        sa.Column("snapshot_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.Text(), nullable=False, server_default="building"),
        sa.Column("storage_path", sa.Text(), nullable=False, server_default=""),
        sa.Column("node_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("edge_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("built_from_revision", sa.Text(), nullable=False, server_default=""),
        sa.Column("error_message", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "scope_type",
            "scope_key",
            "graph_type",
            "snapshot_version",
            name="uq_graph_snapshots_scope_version",
        ),
    )
    op.create_index(
        "idx_graph_snapshots_scope_graph",
        "graph_snapshots",
        ["user_id", "scope_type", "scope_key", "graph_type"],
    )
    op.create_index("idx_graph_snapshots_status", "graph_snapshots", ["status", "updated_at"])


def downgrade() -> None:
    op.drop_index("idx_graph_snapshots_status", table_name="graph_snapshots")
    op.drop_index("idx_graph_snapshots_scope_graph", table_name="graph_snapshots")
    op.drop_table("graph_snapshots")

    op.drop_index("idx_graph_facts_dst", table_name="graph_facts")
    op.drop_index("idx_graph_facts_src", table_name="graph_facts")
    op.drop_index("idx_graph_facts_scope_graph", table_name="graph_facts")
    op.drop_table("graph_facts")
