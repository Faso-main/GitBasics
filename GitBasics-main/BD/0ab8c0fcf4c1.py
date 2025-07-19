from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0ab8c0fcf4c1'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_table('QuestAnswer',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('question', sa.String(), nullable=False),
    sa.Column('answer', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )