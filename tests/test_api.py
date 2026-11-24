from importlib import import_module

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
    """PDF отклоняется по расширению до обращения к классификатору."""
    # Содержимое задано байтами: здесь проверяем отклонение расширения,
    # а не чтение настоящего PDF-документа.
    response = api_client.post(
        "/documents",
        files={"file": ("sample.pdf", b"Some content", "application/pdf")},
    )

    assert response.status_code == 415
    assert response.json() == {"detail": "Поддерживаются только TXT-файлы"}
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
