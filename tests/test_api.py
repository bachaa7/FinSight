from importlib import import_module
from io import BytesIO
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(classifier_module):
    """Подготовить клиент настоящего API с тестовой заменой нейросети."""
    # classifier_module — общая фикстура из conftest.py. К этому моменту
    # она уже подменила Transformers и загрузила наш модуль классификатора.
    # Заполняем настройки тестовой модели: их читает эндпоинт /health.
    classifier_module.classifier.model.config.id2label = {
        0: "Financial",
        1: "General",
        2: "Legal",
        3: "News",
        4: "Scientific",
        5: "Technical",
    }

    # Импортируем API внутри фикстуры, после подмены модели. При импорте
    # в начале файла настоящая модель могла бы загрузиться ещё до теста.
    api_module = import_module("finsight.api.main")

    # TestClient отправляет запросы прямо приложению, без запуска Uvicorn
    # и без сетевого подключения. yield передаёт клиент тесту; после теста
    # блок with закрывает клиент, а classifier_module отменяет подмены.
    with TestClient(api_module.app) as client:
        yield client


def test_health(api_client, classifier_module):
    """/health возвращает статус и классы модели без запуска предсказания."""
    response = api_client.get("/health")

    assert response.status_code == 200
    # response.json() превращает JSON-тело ответа в обычный словарь Python.
    assert response.json() == {
        "status": "ok",
        "classes": [
            "Financial",
            "General",
            "Legal",
            "News",
            "Scientific",
            "Technical",
        ],
    }
    classifier_module.classifier.assert_not_called()


def test_classify(api_client, classifier_module):
    """POST-запрос проходит через схемы API и настоящую classify_document."""
    text = "The borrower shall repay the loan."
    model = classifier_module.classifier
    # pytest использует один экземпляр classifier_module в пределах теста:
    # этот model — тот же объект, который использует подготовленное API.
    # Один короткий текст даст один фрагмент и один список из шести оценок.
    model.return_value = [
        [
            {"label": "News", "score": 0.04},
            {"label": "Legal", "score": 0.80},
            {"label": "Financial", "score": 0.07},
            {"label": "Technical", "score": 0.05},
            {"label": "General", "score": 0.03},
            {"label": "Scientific", "score": 0.01},
        ]
    ]

    # json= задаёт JSON-тело запроса. Обработчик API сам получает из него
    # ClassifyRequest, вызывает классификацию и проверяет формат ответа.
    response = api_client.post("/classify", json={"text": text})

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"label", "confidence", "top_3"}
    assert data["label"] == "Legal"
    assert data["confidence"] == pytest.approx(0.80)
    assert [item["label"] for item in data["top_3"]] == [
        "Legal",
        "Financial",
        "Technical",
    ]
    assert [item["confidence"] for item in data["top_3"]] == pytest.approx(
        [0.80, 0.07, 0.05]
    )
    model.assert_called_once_with([text], top_k=None)


def test_classify_whitespace(api_client, classifier_module):
    """Пробельный текст даёт HTTP 422 с сообщением вместо серверной ошибки."""
    # Пробелы проходят min_length=1 в схеме запроса. Поэтому здесь
    # действительно проверяется цепочка: classify_document вызывает
    # ValueError, обработчик ловит её и поднимает HTTPException с кодом 422.
    response = api_client.post("/classify", json={"text": "   "})

    assert response.status_code == 422
    assert response.json() == {"detail": "Текст не должен быть пустым"}
    classifier_module.classifier.assert_not_called()


def test_upload_txt(api_client, classifier_module):
    text = "The borrower shall repay the loan."
    model = classifier_module.classifier
    model.return_value = [
        [
            {"label": "News", "score": 0.04},
            {"label": "Legal", "score": 0.80},
            {"label": "Financial", "score": 0.07},
            {"label": "Technical", "score": 0.05},
            {"label": "General", "score": 0.03},
            {"label": "Scientific", "score": 0.01},
        ]
    ]
    response = api_client.post(
        "/documents",
        files={"file": ("sample.txt", text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["filename"] == "sample.txt"
    assert data["content_type"] == "text/plain"
    assert data["word_count"] == 6

    classification = data["classification"]
    assert classification["label"] == "Legal"
    assert classification["confidence"] == pytest.approx(0.80)


def test_upload_unsupported_format(api_client, classifier_module):
    """Изображение отклоняется по расширению до обращения к классификатору."""
    # Содержимое задано байтами: здесь проверяем отклонение расширения,
    # а не чтение настоящего изображения. PDF теперь поддерживается.
    response = api_client.post(
        "/documents",
        files={"file": ("sample.png", b"Some content", "image/png")},
    )

    assert response.status_code == 415
    assert response.json() == {"detail": "Поддерживаются только TXT, PDF и DOCX-файлы"}
    classifier_module.classifier.assert_not_called()


@pytest.mark.parametrize(
    "content",
    [b"", b"   \n\t", b"\xef\xbb\xbf"],
    ids=["empty", "whitespace", "bom_only"],
)
def test_upload_empty_txt(api_client, classifier_module, content):
    """Пустое содержимое после декодирования даёт 422 без вызова модели."""
    # Проверяем пустой файл, одни пробельные символы и файл только с BOM.
    # Последний после decode("utf-8-sig") тоже превращается в пустую строку.
    response = api_client.post(
        "/documents",
        files={"file": ("empty.txt", content, "text/plain")},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Текст не должен быть пустым"}
    classifier_module.classifier.assert_not_called()


def test_upload_invalid_encoding(api_client, classifier_module):
    """Байты, недопустимые в UTF-8, отклоняются до классификации."""
    # 0xff не может быть байтом корректного UTF-8-текста.
    response = api_client.post(
        "/documents",
        files={"file": ("invalid.txt", b"\xff", "text/plain")},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "TXT-файл должен быть в кодировке UTF-8"}
    classifier_module.classifier.assert_not_called()


def test_upload_too_large(api_client, classifier_module, monkeypatch):
    """Файл на один байт больше лимита получает 413 без вызова модели."""
    # monkeypatch — встроенная фикстура pytest. Она временно меняет значение
    # и автоматически возвращает исходное после теста. Уменьшаем лимит до
    # 8 байт, чтобы не создавать и не пересылать файл размером больше 20 МиБ.
    api_module = import_module("finsight.api.main")
    monkeypatch.setattr(api_module, "MAX_FILE_SIZE_BYTES", 8)

    response = api_client.post(
        "/documents",
        files={"file": ("large.txt", b"a" * 9, "text/plain")},
    )

    assert response.status_code == 413
    # Меняем только числовую константу; текст сообщения остаётся тем,
    # который задан в обработчике для настоящего лимита приложения.
    assert response.json() == {"detail": "Размер файла превышает 20 МБ"}
    classifier_module.classifier.assert_not_called()


def test_upload_exact_size_limit(api_client, classifier_module, monkeypatch):
    """Файл ровно на границе лимита принимается и классифицируется."""
    api_module = import_module("finsight.api.main")
    monkeypatch.setattr(api_module, "MAX_FILE_SIZE_BYTES", 8)
    content = b"One two."  # Ровно 8 байт, два слова.
    model = classifier_module.classifier
    model.return_value = [
        [
            {"label": "News", "score": 0.04},
            {"label": "Legal", "score": 0.80},
            {"label": "Financial", "score": 0.07},
            {"label": "Technical", "score": 0.05},
            {"label": "General", "score": 0.03},
            {"label": "Scientific", "score": 0.01},
        ]
    ]

    response = api_client.post(
        "/documents",
        files={"file": ("limit.txt", content, "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "limit.txt"
    assert data["word_count"] == 2
    assert data["classification"]["label"] == "Legal"
    assert data["classification"]["confidence"] == pytest.approx(0.80)
    model.assert_called_once_with(["One two."], top_k=None)


def test_upload_pdf(api_client, classifier_module, pdf_document_factory):
    """Текст обеих страниц настоящего PDF доходит до классификатора."""
    content = pdf_document_factory(
        [
            "The borrower shall repay the loan.",
            "The lender shall receive interest.",
        ]
    )
    model = classifier_module.classifier
    model.return_value = [
        [
            {"label": "News", "score": 0.04},
            {"label": "Legal", "score": 0.80},
            {"label": "Financial", "score": 0.07},
            {"label": "Technical", "score": 0.05},
            {"label": "General", "score": 0.03},
            {"label": "Scientific", "score": 0.01},
        ]
    ]

    response = api_client.post(
        "/documents",
        files={"file": ("sample.pdf", content, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.pdf"
    assert data["content_type"] == "application/pdf"
    assert data["word_count"] == 11
    assert data["classification"]["label"] == "Legal"
    assert data["classification"]["confidence"] == pytest.approx(0.80)
    # split_text объединяет пробелы и переносы строк. Ожидаем текст обеих
    # страниц в исходном порядке, а не только первой страницы.
    model.assert_called_once_with(
        ["The borrower shall repay the loan. The lender shall receive interest."],
        top_k=None,
    )


@pytest.mark.parametrize(
    "document_kind, expected_detail",
    [
        ("empty", "PDF не содержит извлекаемого текста"),
        ("encrypted", "PDF защищён паролем"),
        ("broken", "Не удалось прочитать PDF-файл"),
    ],
    ids=["empty", "encrypted", "broken"],
)
def test_upload_invalid_pdf(
    api_client,
    classifier_module,
    pdf_document_factory,
    document_kind,
    expected_detail,
):
    """Ошибки извлечения PDF превращаются в HTTP 422 без вызова модели."""
    if document_kind == "empty":
        content = pdf_document_factory([""])
    elif document_kind == "encrypted":
        content = pdf_document_factory(["Secret text."], password="test-password")
    else:
        content = b"%PDF-1.7\nbroken"

    response = api_client.post(
        "/documents",
        files={"file": ("invalid.pdf", content, "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": expected_detail}
    classifier_module.classifier.assert_not_called()


@pytest.mark.parametrize(
    "paragraphs, table_rows, trailing_paragraphs, expected_text, word_count",
    [
        (
            ["The borrower shall repay the loan."],
            [["Company", "Amount"], ["Acme", "1000"]],
            ["The lender shall receive interest."],
            (
                "The borrower shall repay the loan. Company Amount Acme 1000 "
                "The lender shall receive interest."
            ),
            15,
        ),
        (
            [],
            [["Company", "Amount"], ["Acme", "1000"]],
            [],
            "Company Amount Acme 1000",
            4,
        ),
        (["", "Useful content."], None, [], "Useful content.", 2),
    ],
    ids=["paragraph_table_paragraph", "table_only", "blank_first_paragraph"],
)
def test_upload_docx(
    api_client,
    classifier_module,
    docx_document_factory,
    paragraphs,
    table_rows,
    trailing_paragraphs,
    expected_text,
    word_count,
):
    """Абзацы и ячейки DOCX доходят до классификатора в порядке документа."""
    content = docx_document_factory(paragraphs, table_rows, trailing_paragraphs)
    model = classifier_module.classifier
    model.return_value = [
        [
            {"label": "News", "score": 0.04},
            {"label": "Legal", "score": 0.80},
            {"label": "Financial", "score": 0.07},
            {"label": "Technical", "score": 0.05},
            {"label": "General", "score": 0.03},
            {"label": "Scientific", "score": 0.01},
        ]
    ]
    content_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    response = api_client.post(
        "/documents",
        files={"file": ("sample.docx", content, content_type)},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.docx"
    assert data["content_type"] == content_type
    assert data["word_count"] == word_count
    assert data["classification"]["label"] == "Legal"
    assert data["classification"]["confidence"] == pytest.approx(0.80)
    # split_text нормализует табуляции и переносы в обычные пробелы.
    # Проверяем содержание и порядок всего переданного текста.
    model.assert_called_once_with([expected_text], top_k=None)


@pytest.mark.parametrize(
    "document_kind, expected_detail",
    [
        ("empty", "DOCX не содержит извлекаемого текста"),
        ("whitespace", "DOCX не содержит извлекаемого текста"),
        ("not_zip", "Не удалось прочитать DOCX-файл"),
        ("missing_parts", "Не удалось прочитать DOCX-файл"),
        ("malformed_xml", "Не удалось прочитать DOCX-файл"),
    ],
    ids=["empty", "whitespace", "not_zip", "missing_parts", "malformed_xml"],
)
def test_upload_invalid_docx(
    api_client,
    classifier_module,
    docx_document_factory,
    document_kind,
    expected_detail,
):
    """Непригодный DOCX даёт понятный HTTP 422 до обращения к модели."""
    if document_kind == "empty":
        content = docx_document_factory([])
    elif document_kind == "whitespace":
        content = docx_document_factory(["   \n\t"])
    elif document_kind == "not_zip":
        content = b"Not a DOCX archive"
    elif document_kind == "missing_parts":
        # Это исправный ZIP, но внутри нет обязательных частей DOCX.
        stream = BytesIO()
        with ZipFile(stream, "w") as archive:
            archive.writestr("example.txt", "Not a Word document")
        content = stream.getvalue()
    else:
        # Оставляем исправный ZIP и остальные части настоящего DOCX,
        # но повреждаем XML основного текста документа.
        original = docx_document_factory(["Some text."])
        stream = BytesIO()
        with ZipFile(BytesIO(original)) as source, ZipFile(stream, "w") as target:
            for name in source.namelist():
                data = b"<broken" if name == "word/document.xml" else source.read(name)
                target.writestr(name, data)
        content = stream.getvalue()

    response = api_client.post(
        "/documents",
        files={
            "file": (
                "invalid.docx",
                content,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": expected_detail}
    classifier_module.classifier.assert_not_called()
