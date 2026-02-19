"""add_revoked_tokens_table

Revision ID: a1b2c3d4e5f6
Revises: 8499e1630da3
Create Date: 2026-02-19 18:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '8499e1630da3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the revoked_tokens table for JWT revocation tracking."""
    op.create_table(
        'revoked_tokens',
        sa.Column('token_hash', sa.Text(), nullable=False),
        sa.Column('expires_at', sa.Text(), nullable=False),
        sa.Column('revoked_at', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('token_hash'),
    )
    op.create_index('idx_revoked_tokens_expires_at', 'revoked_tokens', ['expires_at'], unique=False)


def downgrade() -> None:
    """Drop the revoked_tokens table."""
    op.drop_index('idx_revoked_tokens_expires_at', table_name='revoked_tokens')
    op.drop_table('revoked_tokens')
