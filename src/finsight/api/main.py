from pathlib import Path

import joblib
from fastapi import FastAPI

from .schemas import ClassifyRequest, ClassifyResponse

app = FastAPI(title="Finsight API")

# путь к модели
MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "baseline_tfidf_logreg.joblib"
)


model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classes": model.classes_.tolist()}


@app.post("/classify")
def classify(request: ClassifyRequest) -> ClassifyResponse:
    text = request.text
    probabilities = model.predict_proba([text])[0]
    best = probabilities.argmax()
    label = str(model.classes_[best])
    confidence = float(probabilities[best].round(3))
    top_positions = probabilities.argsort()[::-1][:3]
    top_3 = []
    for position in top_positions:
        top_3.append(
            {
                "label": str(model.classes_[position]),
                "confidence": float(probabilities[position].round(3)),
            }
        )

    return {"label": label, "confidence": confidence, "top_3": top_3}
