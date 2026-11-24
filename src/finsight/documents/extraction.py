from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def extract_txt_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8-sig")


def extract_pdf_text(file_bytes: bytes) -> str:
    pdf_stream = BytesIO(file_bytes)
    try:
        reader = PdfReader(pdf_stream)
        if reader.is_encrypted:
            raise ValueError("PDF защищён паролем")
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")

        text = "\n\n".join(pages_text)

        if not text.strip():
            raise ValueError("PDF не содержит извлекаемого текста")
        return text

    except PdfReadError as error:
        raise ValueError("Не удалось прочитать PDF-файл") from error
