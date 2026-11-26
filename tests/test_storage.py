from uuid import uuid4

import pytest

from finsight.documents.storage import delete_document_file, save_document_file


def test_save_creates_directory_and_keeps_original_bytes(upload_directory):
    document_id = uuid4()
    content = b"%PDF-1.7\n\x00\xffbinary content"

    name = save_document_file(content, document_id, ".pdf")

    assert name == f"{document_id}.pdf"
    assert (upload_directory / name).read_bytes() == content


def test_save_does_not_overwrite_existing_file(upload_directory):
    document_id = uuid4()
    name = save_document_file(b"original", document_id, ".txt")

    with pytest.raises(FileExistsError):
        save_document_file(b"replacement", document_id, ".txt")

    assert (upload_directory / name).read_bytes() == b"original"


def test_delete_removes_only_selected_file(upload_directory):
    selected = save_document_file(b"selected", uuid4(), ".txt")
    other = save_document_file(b"keep", uuid4(), ".txt")

    delete_document_file(selected)

    assert not (upload_directory / selected).exists()
    assert (upload_directory / other).read_bytes() == b"keep"


def test_delete_missing_file_is_allowed(upload_directory):
    # Даже сама папка загрузок ещё не создана.
    delete_document_file(f"{uuid4()}.txt")
    assert not upload_directory.exists()
