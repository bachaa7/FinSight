from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError

from finsight.models import Document


@pytest.fixture
def stored_document():
    document_id = uuid4()
    return Document(
        id=document_id,
        filename="report.txt",
        content_type="text/plain",
        storage_path=f"{document_id}.txt",
        text="The borrower shall repay the loan.",
        word_count=6,
        classification={
            "label": "Legal",
            "confidence": 0.8,
            "top_3": [{"label": "Legal", "confidence": 0.8}],
        },
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def upload_txt(client):
    return client.post(
        "/documents",
        files={
            "file": ("report.txt", b"The borrower shall repay the loan.", "text/plain")
        },
    )


def test_upload_connects_file_and_database_record(
    api_client, api_session, upload_directory
):
    response = upload_txt(api_client)

    assert response.status_code == 200
    data = response.json()
    document_id = UUID(data["id"])
    document = api_session.add.call_args.args[0]
    assert document.id == document_id
    assert document.storage_path == f"{document_id}.txt"
    assert document.filename == data["filename"]
    assert document.word_count == data["word_count"] == 6
    assert document.text == "The borrower shall repay the loan."
    assert document.classification == data["classification"]
    assert (
        upload_directory / document.storage_path
    ).read_bytes() == document.text.encode()
    assert datetime.fromisoformat(data["created_at"]).tzinfo is not None
    assert "storage_path" not in data and "text" not in data
    api_session.commit.assert_called_once()
    api_session.refresh.assert_called_once_with(document)


def test_failed_upload_commit_removes_file(api_client, api_session, upload_directory):
    api_session.commit.side_effect = SQLAlchemyError("test database failure")

    response = upload_txt(api_client)

    assert response.status_code == 500
    assert response.json() == {"detail": "Не удалось сохранить документ"}
    api_session.rollback.assert_called_once()
    api_session.refresh.assert_not_called()
    assert list(upload_directory.iterdir()) == []


def test_failed_refresh_does_not_remove_committed_file(
    api_client, api_session, upload_directory
):
    api_session.refresh.side_effect = SQLAlchemyError("test refresh failure")

    # TestClient по умолчанию поднимает необработанное исключение сервера.
    # Важный инвариант: ошибка чтения после commit не удаляет файл записи.
    with pytest.raises(SQLAlchemyError, match="test refresh failure"):
        upload_txt(api_client)

    api_session.commit.assert_called_once()
    api_session.rollback.assert_not_called()
    assert len(list(upload_directory.iterdir())) == 1


def test_empty_document_list(api_client, api_session):
    assert api_client.get("/documents").json() == []
    api_session.commit.assert_not_called()


def test_document_list_serializes_orm_and_hides_internal_fields(
    api_client, api_session, stored_document
):
    api_session.scalars.return_value.all.return_value = [stored_document]

    response = api_client.get("/documents")

    assert response.status_code == 200
    item = response.json()[0]
    assert item["id"] == str(stored_document.id)
    assert item["classification"] == stored_document.classification
    assert "text" not in item and "storage_path" not in item


def test_document_detail_reads_saved_text_without_model(
    api_client, api_session, stored_document, classifier_module
):
    api_session.get.return_value = stored_document

    response = api_client.get(f"/documents/{stored_document.id}")

    assert response.status_code == 200
    assert response.json()["text"] == stored_document.text
    assert response.json()["id"] == str(stored_document.id)
    assert "storage_path" not in response.json()
    classifier_module.classifier.assert_not_called()
    api_session.commit.assert_not_called()


@pytest.mark.parametrize("method", ["get", "delete"])
def test_missing_document_returns_404(api_client, api_session, method):
    response = getattr(api_client, method)(f"/documents/{uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Документ не найден"}
    api_session.delete.assert_not_called()
    api_session.commit.assert_not_called()


@pytest.mark.parametrize("method", ["get", "delete"])
def test_invalid_uuid_is_rejected_before_database(api_client, api_session, method):
    response = getattr(api_client, method)("/documents/not-a-uuid")

    assert response.status_code == 422
    api_session.get.assert_not_called()


@pytest.mark.parametrize("file_exists", [True, False])
def test_delete_removes_record_and_file(
    api_client, api_session, stored_document, upload_directory, file_exists
):
    api_session.get.return_value = stored_document
    file_path = upload_directory / stored_document.storage_path
    if file_exists:
        upload_directory.mkdir()
        file_path.write_bytes(b"original file")

    response = api_client.delete(f"/documents/{stored_document.id}")

    assert response.status_code == 204
    assert response.content == b""
    assert not file_path.exists()
    api_session.delete.assert_called_once_with(stored_document)
    api_session.commit.assert_called_once()


def test_failed_delete_commit_keeps_file(
    api_client, api_session, stored_document, upload_directory
):
    api_session.get.return_value = stored_document
    api_session.commit.side_effect = SQLAlchemyError("test delete failure")
    upload_directory.mkdir()
    file_path = upload_directory / stored_document.storage_path
    file_path.write_bytes(b"keep this file")

    response = api_client.delete(f"/documents/{stored_document.id}")

    assert response.status_code == 500
    assert response.json() == {"detail": "Не удалось удалить документ"}
    assert file_path.read_bytes() == b"keep this file"
    api_session.rollback.assert_called_once()


def test_file_delete_failure_reports_partial_result(
    api_client, api_module, api_session, stored_document, monkeypatch
):
    api_session.get.return_value = stored_document

    def fail_delete(path):
        raise PermissionError("test file access failure")

    monkeypatch.setattr(api_module, "delete_document_file", fail_delete)

    response = api_client.delete(f"/documents/{stored_document.id}")

    assert response.status_code == 500
    assert response.json() == {"detail": "Запись удалена, но не удалось удалить файл"}
    api_session.commit.assert_called_once()
    # После успешного commit откат не восстановит удалённую запись.
    api_session.rollback.assert_not_called()
