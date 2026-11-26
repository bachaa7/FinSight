from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..database import get_session
from ..documents.extraction import extract_docx_text, extract_pdf_text, extract_txt_text
from ..documents.storage import delete_document_file, save_document_file
from ..ml.classifier import classifier, classify_document
from ..models import Document
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    DocumentDetailResponse,
    DocumentResponse,
)

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
def upload_document(
    file: UploadFile, session: Annotated[Session, Depends(get_session)]
) -> DocumentResponse:
    filename = file.filename or ""
    if not filename.lower().endswith((".txt", ".pdf", ".docx")):
        raise HTTPException(
            status_code=415, detail="Поддерживаются только TXT, PDF и DOCX-файлы"
        )

    file_bytes = file.file.read(MAX_FILE_SIZE_BYTES + 1)
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Размер файла превышает 20 МБ")

    try:
        if filename.lower().endswith(".txt"):
            text = extract_txt_text(file_bytes)
        elif filename.lower().endswith(".pdf"):
            text = extract_pdf_text(file_bytes)
        elif filename.lower().endswith(".docx"):
            text = extract_docx_text(file_bytes)
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=422, detail="TXT-файл должен быть в кодировке UTF-8"
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    try:
        classification = classify_document(text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    document_id = uuid4()
    extension = Path(filename).suffix.lower()
    storage_path = save_document_file(file_bytes, document_id, extension)
    document = Document(
        id=document_id,
        filename=filename,
        content_type=file.content_type,
        storage_path=storage_path,
        text=text,
        word_count=len(text.split()),
        classification=classification,
    )

    try:
        session.add(document)
        session.commit()
    except SQLAlchemyError as error:
        session.rollback()
        delete_document_file(storage_path)
        raise HTTPException(
            status_code=500, detail="Не удалось сохранить документ"
        ) from error

    session.refresh(document)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "word_count": len(text.split()),
        "classification": classification,
        "id": document.id,
        "created_at": document.created_at,
    }


@app.get("/documents", response_model=list[DocumentResponse])
def get_documents(session: Annotated[Session, Depends(get_session)]) -> list[Document]:
    statement = select(Document).order_by(Document.created_at.desc())
    documents = session.scalars(statement).all()

    return list(documents)


@app.get("/documents/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    document_id: UUID, session: Annotated[Session, Depends(get_session)]
) -> Document:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return document


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: UUID, session: Annotated[Session, Depends(get_session)]
) -> Response:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    storage_path = document.storage_path
    try:
        session.delete(document)
        session.commit()
    except SQLAlchemyError as error:
        session.rollback()
        raise HTTPException(
            status_code=500, detail="Не удалось удалить документ"
        ) from error

    try:
        delete_document_file(storage_path)
    except OSError as error:
        raise HTTPException(
            status_code=500, detail="Запись удалена, но не удалось удалить файл"
        ) from error

    return Response(status_code=204)
