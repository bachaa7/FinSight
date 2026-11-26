"""Необязательная проверка настоящего PostgreSQL, без пользовательских таблиц.

Запуск: FINSIGHT_TEST_POSTGRES=1 в окружении и обычный pytest.
Каждый тест создаёт отдельную схему внутри внешней транзакции. Все commit()
эндпоинтов остаются внутри неё благодаря SAVEPOINT; в конце всё откатывается.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema

from finsight.database import get_database_url
from finsight.models import Document


@pytest.fixture
def postgres_connection(request):
    if os.environ.get("FINSIGHT_TEST_POSTGRES") != "1":
        pytest.skip("Для настоящего PostgreSQL задайте FINSIGHT_TEST_POSTGRES=1")

    engine = create_engine(
        get_database_url(), poolclass=NullPool, connect_args={"connect_timeout": 5}
    )
    schema_name = f"finsight_test_{uuid4().hex}"
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(CreateSchema(schema_name))
            # ALTER TABLE из Alembic тоже должен работать только в тестовой
            # схеме. Имя сгенерировано нами из UUID, не приходит от клиента.
            connection.execute(text(f'SET LOCAL search_path TO "{schema_name}"'))
            connection = connection.execution_options(
                schema_translate_map={None: schema_name}
            )
            # Применяем всю цепочку в порядке ревизий: новые колонки должны
            # проверяться настоящими миграциями, а не create_all(model).
            config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
            revisions = list(ScriptDirectory.from_config(config).walk_revisions())
            context = MigrationContext.configure(connection)
            with Operations.context(context):
                for revision in reversed(revisions):
                    revision.module.upgrade()
                    if revision.revision == "3636df6d3c71" and getattr(
                        request, "param", False
                    ):
                        # Строка существует ДО добавления entities. Явно
                        # задаём старые поля, которых достаточно первой схеме.
                        connection.execute(
                            Document.__table__.insert().values(
                                id=uuid4(),
                                filename="before-ner.txt",
                                content_type="text/plain",
                                storage_path="legacy.txt",
                                text="Old document.",
                                word_count=2,
                                classification={
                                    "label": "General",
                                    "confidence": 0.7,
                                    "top_3": [{"label": "General", "confidence": 0.7}],
                                },
                            )
                        )
            yield connection
        finally:
            if transaction.is_active:
                transaction.rollback()
    engine.dispose()


@pytest.fixture
def postgres_client(api_client, api_module, postgres_connection):
    def real_test_session():
        with Session(
            bind=postgres_connection, join_transaction_mode="create_savepoint"
        ) as session:
            yield session

    api_module.app.dependency_overrides[api_module.get_session] = real_test_session
    return api_client


def test_postgres_upload_read_delete_roundtrip(postgres_client, upload_directory):
    text = "The borrower shall repay the loan."
    response = postgres_client.post(
        "/documents", files={"file": ("report.txt", text.encode(), "text/plain")}
    )
    assert response.status_code == 200
    item = response.json()
    document_id = item["id"]
    path = upload_directory / f"{document_id}.txt"
    assert path.read_bytes() == text.encode()
    assert datetime.fromisoformat(item["created_at"]).tzinfo is not None

    listing = postgres_client.get("/documents")
    assert listing.status_code == 200
    assert [document["id"] for document in listing.json()] == [document_id]
    detail = postgres_client.get(f"/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["text"] == text
    assert detail.json()["classification"] == item["classification"]
    assert detail.json()["entities"] == item["entities"] == []

    deleted = postgres_client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert not path.exists()
    assert postgres_client.get("/documents").json() == []
    assert postgres_client.get(f"/documents/{document_id}").status_code == 404
    assert postgres_client.delete(f"/documents/{document_id}").status_code == 404


def test_postgres_list_orders_newest_first(postgres_client, postgres_connection):
    ids = [uuid4(), uuid4()]
    with Session(
        bind=postgres_connection, join_transaction_mode="create_savepoint"
    ) as session:
        for document_id, year in zip(ids, [2020, 2026]):
            session.add(
                Document(
                    id=document_id,
                    filename=f"{year}.txt",
                    content_type="text/plain",
                    storage_path=f"{document_id}.txt",
                    text="Example text.",
                    word_count=2,
                    classification={
                        "label": "General",
                        "confidence": 0.7,
                        "top_3": [{"label": "General", "confidence": 0.7}],
                    },
                    created_at=datetime(year, 1, 1, tzinfo=UTC),
                )
            )
        session.commit()

    response = postgres_client.get("/documents")
    assert response.status_code == 200
    assert [document["id"] for document in response.json()] == [
        str(ids[1]),
        str(ids[0]),
    ]


@pytest.mark.parametrize("postgres_connection", [True], indirect=True)
def test_entities_migration_preserves_old_document(postgres_client):
    response = postgres_client.get("/documents")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["filename"] == "before-ner.txt"
    assert items[0]["entities"] is None
    detail = postgres_client.get(f"/documents/{items[0]['id']}")
    assert detail.json()["text"] == "Old document."
    assert detail.json()["entities"] is None


def test_postgres_entities_jsonb_roundtrip(postgres_client, entities_module):
    from types import SimpleNamespace

    value = " Apple received $5 million."
    entities_module.nlp.return_value = SimpleNamespace(
        ents=[
            SimpleNamespace(text="Apple", label_="ORG", start_char=1, end_char=6),
            SimpleNamespace(
                text="$5 million", label_="MONEY", start_char=16, end_char=26
            ),
        ]
    )
    uploaded = postgres_client.post(
        "/documents", files={"file": ("ner.txt", value.encode(), "text/plain")}
    )
    assert uploaded.status_code == 200
    item = uploaded.json()
    detail = postgres_client.get(f"/documents/{item['id']}")
    assert detail.status_code == 200
    assert detail.json()["entities"] == item["entities"]
    assert len(item["entities"]) == 2
    assert all(value[e["start"] : e["end"]] == e["text"] for e in item["entities"])
