import sys
from importlib import import_module
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


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
            yield import_module("finsight.ml.classifier")


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
