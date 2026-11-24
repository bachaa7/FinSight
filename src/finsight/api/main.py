from fastapi import FastAPI, HTTPException, UploadFile

from ..ml.classifier import classifier, classify_document
from .schemas import ClassifyRequest, ClassifyResponse, DocumentUploadResponse

app = FastAPI(title="Finsight API")

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classes": list(classifier.model.config.id2label.values())}


@app.post("/classify")
def classify(request: ClassifyRequest) -> ClassifyResponse:
    try:
        return classify_document(request.text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/documents")
def upload_document(file: UploadFile) -> DocumentUploadResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".txt"):
        raise HTTPException(status_code=415, detail="Поддерживаются только TXT-файлы")

    file_bytes = file.file.read(MAX_FILE_SIZE_BYTES + 1)
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Размер файла превышает 20 МБ")

    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=422, detail="TXT-файл должен быть в кодировке UTF-8"
        )

    try:
        classification = classify_document(text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "word_count": len(text.split()),
        "classification": classification,
    }
