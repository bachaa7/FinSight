from fastapi import FastAPI, HTTPException

from ..ml.classifier import classifier, classify_document
from .schemas import ClassifyRequest, ClassifyResponse

app = FastAPI(title="Finsight API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classes": list(classifier.model.config.id2label.values())}


@app.post("/classify")
def classify(request: ClassifyRequest) -> ClassifyResponse:
    try:
        return classify_document(request.text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
