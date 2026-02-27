"""add_dr_checkpoints_table

Revision ID: c3d4e5f6a7b8
Revises: a1b2c3d4e5f6
Create Date: 2026-02-20 11:30:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create checkpoint table for deep-research restart."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("deep_research_checkpoints"):
        op.create_table(
            "deep_research_checkpoints",
            sa.Column("job_id", sa.Text(), nullable=False),
            sa.Column("phase", sa.Text(), nullable=False),
            sa.Column("section_title", sa.Text(), nullable=False, server_default=""),
            sa.Column("state_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.ForeignKeyConstraint(["job_id"], ["deep_research_jobs.job_id"]),
            sa.PrimaryKeyConstraint("job_id", "phase", "section_title"),
        )

    index_names = {idx.get("name") for idx in inspector.get_indexes("deep_research_checkpoints")}
    if "idx_dr_checkpoints_job_created" not in index_names:
        op.create_index(
            "idx_dr_checkpoints_job_created",
            "deep_research_checkpoints",
            ["job_id", "created_at"],
            unique=False,
        )
    if "idx_dr_checkpoints_job_phase" not in index_names:
        op.create_index(
            "idx_dr_checkpoints_job_phase",
            "deep_research_checkpoints",
            ["job_id", "phase"],
            unique=False,
        )


def downgrade() -> None:
    """Drop checkpoint table."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("deep_research_checkpoints"):
        index_names = {idx.get("name") for idx in inspector.get_indexes("deep_research_checkpoints")}
        if "idx_dr_checkpoints_job_phase" in index_names:
            op.drop_index("idx_dr_checkpoints_job_phase", table_name="deep_research_checkpoints")
        if "idx_dr_checkpoints_job_created" in index_names:
            op.drop_index("idx_dr_checkpoints_job_created", table_name="deep_research_checkpoints")
        op.drop_table("deep_research_checkpoints")
