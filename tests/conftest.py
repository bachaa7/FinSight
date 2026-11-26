import sys
from datetime import UTC, datetime
from importlib import import_module
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy.orm import Session


@pytest.fixture
def classifier_module():
    """Передать тесту наш модуль с тестовой заменой нейросети."""
    # sys.modules — хранилище импортированных модулей. Вместо настоящей
    # библиотеки Transformers помещаем туда MagicMock: её pipeline()
    # вернёт ещё один тестовый объект, не загружая библиотеку и веса.
    with patch.dict(sys.modules, {"transformers": MagicMock()}):
        # import_module обычно повторно использует модуль из sys.modules.
        # Убираем предыдущий импорт, чтобы каждый тест получил свежую модель
        # без предсказаний и истории вызовов от предыдущего теста.
        sys.modules.pop("finsight.ml.classifier", None)
        ml_package = import_module("finsight.ml")

        # При импорте Python также записывает модуль в атрибут пакета ml.
        # Сохраняем и восстанавливаем этот атрибут вместе с sys.modules.
        with patch.object(ml_package, "classifier", create=True):
            # yield передаёт модуль тесту. После теста блоки with завершатся
            # и вернут исходное состояние импортов.
            module = import_module("finsight.ml.classifier")
            # Простая тестовая токенизация: одно слово — один токен,
            # плюс два служебных. В тестах переполнения заменяем её отдельно.
            module.classifier.tokenizer.side_effect = lambda text, **kwargs: {
                "input_ids": [0] * (len(text.split()) + 2)
            }
            yield module


@pytest.fixture
def entities_module():
    """Тестировать свою обработку сущностей без загрузки spaCy и весов."""
    with patch.dict(sys.modules, {"spacy": MagicMock()}):
        sys.modules.pop("finsight.ml.entities", None)
        ml_package = import_module("finsight.ml")
        with patch.object(ml_package, "entities", create=True):
            module = import_module("finsight.ml.entities")
            # Нет сущностей — обработанный документ получает [], а не None.
            module.nlp.return_value = SimpleNamespace(ents=[])
            yield module


@pytest.fixture
def api_module(classifier_module, entities_module):
    """Импортировать API только после подмены обеих моделей."""
    classifier_module.classifier.model.config.id2label = {
        0: "Financial",
        1: "General",
        2: "Legal",
        3: "News",
        4: "Scientific",
        5: "Technical",
    }
    classifier_module.classifier.return_value = [
        [
            {"label": "Legal", "score": 0.8},
            {"label": "Financial", "score": 0.07},
            {"label": "Technical", "score": 0.05},
            {"label": "News", "score": 0.04},
            {"label": "General", "score": 0.03},
            {"label": "Scientific", "score": 0.01},
        ]
    ]
    return import_module("finsight.api.main")


@pytest.fixture
def upload_directory(tmp_path, monkeypatch):
    """Записывать файлы только во временную папку конкретного теста."""
    storage = import_module("finsight.documents.storage")
    directory = tmp_path / "uploads"
    monkeypatch.setattr(storage, "UPLOAD_DIR", directory)
    return directory


@pytest.fixture
def api_session():
    """Заменить PostgreSQL в обычных API-тестах: внешняя БД не нужна."""
    session = MagicMock(spec=Session)
    session.get.return_value = None
    session.scalars.return_value.all.return_value = []

    def refresh(document):
        # В настоящем PostgreSQL created_at заполняет DEFAULT now().
        document.created_at = datetime(2026, 1, 1, tzinfo=UTC)

    session.refresh.side_effect = refresh
    return session


@pytest.fixture
def api_client(api_module, api_session, upload_directory):
    """Настоящий HTTP-контракт FastAPI, тестовые модель/БД и временные файлы."""
    previous_overrides = api_module.app.dependency_overrides.copy()

    def override_session():
        yield api_session

    # Depends(get_session) теперь получает тестовую сессию, не читая .env.
    api_module.app.dependency_overrides[api_module.get_session] = override_session
    try:
        with TestClient(api_module.app) as client:
            yield client
    finally:
        api_module.app.dependency_overrides.clear()
        api_module.app.dependency_overrides.update(previous_overrides)


@pytest.fixture
def pdf_document_factory():
    """Создать тестовый PDF в памяти: страницы с ASCII-текстом или пустые."""

    def make_pdf(pages: list[str], password: str | None = None) -> bytes:
        writer = PdfWriter()
        # Стандартный шрифт Helvetica позволяет создать маленький PDF
        # без сторонней библиотеки генерации документов и файлов на диске.
        font = writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
        )
        for text in pages:
            page = writer.add_blank_page(width=612, height=792)
            if text:
                page[NameObject("/Resources")] = DictionaryObject(
                    {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
                )
                # Экранируем специальные символы строкового литерала PDF.
                escaped_text = (
                    text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                )
                content = DecodedStreamObject()
                content.set_data(
                    f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
                )
                # Низкоуровневая регистрация потока нужна только для
                # создания тестовых данных; рабочий код использует PdfReader.
                page[NameObject("/Contents")] = writer._add_object(content)
        if password is not None:
            writer.encrypt(password)

        stream = BytesIO()
        writer.write(stream)
        return stream.getvalue()

    # Фикстура возвращает функцию: тест сам выбирает страницы и пароль.
    return make_pdf


@pytest.fixture
def docx_document_factory():
    """Создать DOCX в памяти с абзацами и обычной таблицей между ними."""

    def make_docx(
        paragraphs: list[str],
        table_rows: list[list[str]] | None = None,
        trailing_paragraphs: list[str] | None = None,
    ) -> bytes:
        document = Document()
        for text in paragraphs:
            document.add_paragraph(text)
        if table_rows:
            table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))
            for row_index, values in enumerate(table_rows):
                for column_index, value in enumerate(values):
                    table.cell(row_index, column_index).text = value
        for text in trailing_paragraphs or []:
            document.add_paragraph(text)

        # Возвращаем байты готового документа для files= в TestClient.
        stream = BytesIO()
        document.save(stream)
        return stream.getvalue()

    return make_docx
