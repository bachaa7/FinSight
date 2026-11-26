from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ClassifyRequest(BaseModel):
    """Запрос на классификацию текста."""

    text: str = Field(min_length=1, max_length=20000)


class ClassProbability(BaseModel):
    label: str
    confidence: float


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    top_3: list[ClassProbability]


class DocumentResponse(BaseModel):
    filename: str
    content_type: str | None
    word_count: int
    classification: ClassifyResponse
    id: UUID
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DocumentDetailResponse(DocumentResponse):
    text: str
