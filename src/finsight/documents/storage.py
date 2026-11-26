from pathlib import Path
from uuid import UUID

UPLOAD_DIR = Path(__file__).resolve().parents[3] / "data" / "uploads"


def save_document_file(file_bytes: bytes, document_id: UUID, extension: str) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{document_id}{extension}"
    file_path = UPLOAD_DIR / stored_filename
    with file_path.open("xb") as file:
        file.write(file_bytes)

    return stored_filename


def delete_document_file(storage_path: str) -> None:
    file_path = UPLOAD_DIR / storage_path
    file_path.unlink(missing_ok=True)
