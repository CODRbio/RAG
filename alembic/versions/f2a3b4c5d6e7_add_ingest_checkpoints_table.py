"""add_ingest_checkpoints_table

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-03-09 10:00:00.000000

Adds ingest_checkpoints table: per-file, per-stage persistent markers for the
ingest pipeline enabling resume/skip semantics and full traceability.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "f2a3b4c5d6e7"
down_revision: Union[str, Sequence[str], None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create ingest_checkpoints table."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("ingest_checkpoints"):
        op.create_table(
            "ingest_checkpoints",
            sa.Column("job_id", sa.Text(), nullable=False),
            sa.Column("file_name", sa.Text(), nullable=False),
            sa.Column("stage", sa.Text(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False, server_default="done"),
            sa.Column("detail_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.ForeignKeyConstraint(["job_id"], ["ingest_jobs.job_id"]),
            sa.PrimaryKeyConstraint("job_id", "file_name", "stage"),
        )

    index_names = {idx.get("name") for idx in inspector.get_indexes("ingest_checkpoints")}
    if "idx_ingest_ckpt_job_file" not in index_names:
        op.create_index(
            "idx_ingest_ckpt_job_file",
            "ingest_checkpoints",
            ["job_id", "file_name"],
            unique=False,
        )


def downgrade() -> None:
    """Drop ingest_checkpoints table."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("ingest_checkpoints"):
        index_names = {idx.get("name") for idx in inspector.get_indexes("ingest_checkpoints")}
        if "idx_ingest_ckpt_job_file" in index_names:
            op.drop_index("idx_ingest_ckpt_job_file", table_name="ingest_checkpoints")
        op.drop_table("ingest_checkpoints")
