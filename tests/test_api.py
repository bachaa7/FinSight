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
