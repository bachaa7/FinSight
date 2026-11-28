from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ClassifyRequest(BaseModel):
    """Запрос на классификацию текста."""

    text: str = Field(min_length=1, max_length=20000)


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class ClassProbability(BaseModel):
    label: str
    confidence: float


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    top_3: list[ClassProbability]


class EntityResponse(BaseModel):
    text: str
    label: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class SentimentResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    score: float = Field(ge=-1, le=1)
    probabilities: dict[str, float]


class DocumentResponse(BaseModel):
    filename: str
    content_type: str | None
    word_count: int
    classification: ClassifyResponse
    id: UUID
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
    entities: list[EntityResponse] | None
    sentiment: SentimentResponse | None


class TextStatisticsResponse(BaseModel):
    char_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    sentence_count: int = Field(ge=0)


class DocumentDetailResponse(DocumentResponse):
    text: str
    statistics: TextStatisticsResponse
