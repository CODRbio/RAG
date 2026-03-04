"""add_user_id_to_core_tables

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: Multi-tenant path refactor

Add user_id to papers, ingest_jobs, deep_research_jobs, scholar_libraries.
Replace scholar_libraries unique(name) with unique(user_id, name) for SQLite.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, Sequence[str], None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(inspector: sa.Inspector, table: str, column: str) -> bool:
    if not inspector.has_table(table):
        return False
    return column in {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    is_sqlite = bind.dialect.name == "sqlite"

    # 1. papers
    if not _has_column(inspector, "papers", "user_id"):
        op.add_column(
            "papers",
            sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        )
        with op.batch_alter_table("papers", schema=None) as batch_op:
            batch_op.create_index("idx_papers_user_id", ["user_id"], unique=False)

    # 2. ingest_jobs
    if not _has_column(inspector, "ingest_jobs", "user_id"):
        op.add_column(
            "ingest_jobs",
            sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        )
        with op.batch_alter_table("ingest_jobs", schema=None) as batch_op:
            batch_op.create_index("idx_ingest_jobs_user_id", ["user_id"], unique=False)

    # 3. deep_research_jobs
    if not _has_column(inspector, "deep_research_jobs", "user_id"):
        op.add_column(
            "deep_research_jobs",
            sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        )
        with op.batch_alter_table("deep_research_jobs", schema=None) as batch_op:
            batch_op.create_index("idx_dr_jobs_user_id", ["user_id"], unique=False)

    # 4. scholar_libraries: add user_id; on SQLite replace UNIQUE(name) with UNIQUE(user_id, name) via table recreate
    if not inspector.has_table("scholar_libraries"):
        return

    if not _has_column(inspector, "scholar_libraries", "user_id"):
        op.add_column(
            "scholar_libraries",
            sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        )

    if is_sqlite:
        # SQLite: cannot drop UNIQUE(name); recreate table with UNIQUE(user_id, name)
        op.create_table(
            "scholar_libraries_new",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, server_default=""),
            sa.Column("folder_path", sa.Text(), nullable=True),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "name", name="uq_scholar_libraries_user_name"),
        )
        op.execute(
            "INSERT INTO scholar_libraries_new (id, user_id, name, description, folder_path, created_at, updated_at) "
            "SELECT id, COALESCE(user_id, 'default'), name, description, folder_path, created_at, updated_at FROM scholar_libraries"
        )
        op.drop_table("scholar_libraries")
        op.rename_table("scholar_libraries_new", "scholar_libraries")


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    with op.batch_alter_table("papers", schema=None) as batch_op:
        batch_op.drop_index("idx_papers_user_id", if_exists=True)
    op.drop_column("papers", "user_id")

    with op.batch_alter_table("ingest_jobs", schema=None) as batch_op:
        batch_op.drop_index("idx_ingest_jobs_user_id", if_exists=True)
    op.drop_column("ingest_jobs", "user_id")

    with op.batch_alter_table("deep_research_jobs", schema=None) as batch_op:
        batch_op.drop_index("idx_dr_jobs_user_id", if_exists=True)
    op.drop_column("deep_research_jobs", "user_id")

    if is_sqlite:
        op.create_table(
            "scholar_libraries_old",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, server_default=""),
            sa.Column("folder_path", sa.Text(), nullable=True),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.execute(
            "INSERT INTO scholar_libraries_old (id, name, description, folder_path, created_at, updated_at) "
            "SELECT id, name, description, folder_path, created_at, updated_at FROM scholar_libraries"
        )
        op.drop_table("scholar_libraries")
        op.rename_table("scholar_libraries_old", "scholar_libraries")
    else:
        op.drop_constraint("uq_scholar_libraries_user_name", "scholar_libraries", type_="unique")
        op.drop_column("scholar_libraries", "user_id")
