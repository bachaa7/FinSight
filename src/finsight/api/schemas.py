from pydantic import BaseModel, Field


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
