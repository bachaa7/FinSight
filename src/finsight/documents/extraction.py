from io import BytesIO
from zipfile import BadZipFile

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError
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


def extract_docx_text(file_bytes: bytes) -> str:
    try:
        document = Document(BytesIO(file_bytes))
    except (BadZipFile, KeyError, XMLSyntaxError) as error:
        raise ValueError("Не удалось прочитать DOCX-файл") from error

    text_parts = []
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            text_parts.append(block.text)

        elif isinstance(block, Table):
            for row in block.rows:
                cell_texts = []

                for cell in row.cells:
                    cell_texts.append(cell.text)
                text_parts.append("\t".join(cell_texts))

    text = "\n\n".join(text_parts)

    if not text.strip():
        raise ValueError("DOCX не содержит извлекаемого текста")

    return text
