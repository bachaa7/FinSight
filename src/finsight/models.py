from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )
    filename: Mapped[str] = mapped_column()
    content_type: Mapped[str | None] = mapped_column()
    storage_path: Mapped[str] = mapped_column()
    text: Mapped[str] = mapped_column(Text)
    word_count: Mapped[int] = mapped_column()
    classification: Mapped[dict[str, Any]] = mapped_column(JSONB)
    entities: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
