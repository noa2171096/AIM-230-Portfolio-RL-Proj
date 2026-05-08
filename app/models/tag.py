"""
User Tag model for custom classification categories.

Users can create custom tags which are then used in zero-shot
classification for their future image uploads.
"""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from user import User


class UserTag(Base, TimestampMixin):
    """
    Represents a custom tag created by a user.

    These tags are:
    1. Used for manual image tagging
    2. Added to zero-shot classification categories for the user's images
    3. Tracked for usage statistics

    This enables personalized ML classification based on user's tagging patterns.
    """

    __tablename__ = "user_tags"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Owner
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Tag name (normalized to lowercase)
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Usage count - how many times this tag has been applied
    usage_count: Mapped[int] = mapped_column(Integer, default=1)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="tags")

    # Constraints
    __table_args__ = (
        # Each user can only have one tag with a given name
        UniqueConstraint("user_id", "name", name="uq_user_tag_name"),
        Index("ix_user_tags_user_name", "user_id", "name"),
    )

    def __repr__(self) -> str:
        return f"<UserTag(id={self.id}, name={self.name}, usage_count={self.usage_count})>"