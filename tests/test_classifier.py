import pytest

# Фикстуру classifier_module pytest находит в tests/conftest.py.
# Тесты получают её результат через одноимённый параметр без явного импорта.


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_document(classifier_module, text):
    """Пустая строка и пробельный текст отклоняются до вызова модели."""
    with pytest.raises(ValueError, match="Текст не должен быть пустым"):
        classifier_module.classify_document(text)

    # Проверяем не только исключение, но и отсутствие обращения к модели.
    classifier_module.classifier.assert_not_called()


def test_single_chunk(classifier_module):
    """Для одного фрагмента оценки сохраняются, классы сортируются по ним."""
    text = "The borrower shall repay the loan."
    # Порядок перемешан: победителя нужно определить по score,
    # а не просто взять первую запись из ответа модели.
    fragment_scores = [
        {"label": "News", "score": 0.04},
        {"label": "Legal", "score": 0.80},
        {"label": "Financial", "score": 0.07},
        {"label": "Technical", "score": 0.05},
        {"label": "General", "score": 0.03},
        {"label": "Scientific", "score": 0.01},
    ]
    model = classifier_module.classifier
    # Внешний список — результаты всех фрагментов; здесь фрагмент один.
    model.return_value = [fragment_scores]

    result = classifier_module.classify_document(text)

    assert result["label"] == "Legal"
    assert result["confidence"] == pytest.approx(0.80)
    assert [item["label"] for item in result["top_3"]] == [
        "Legal",
        "Financial",
        "Technical",
    ]
    assert [item["confidence"] for item in result["top_3"]] == pytest.approx(
        [0.80, 0.07, 0.05]
    )
    # top_k=None нужен, чтобы получить оценки всех шести классов.
    model.assert_called_once_with([text], top_k=None, truncation=False)


def test_multiple_chunks(classifier_module):
    """Оценки разных фрагментов объединяются по метке класса и усредняются."""
    # 150 слов разбиваются настоящим split_text на два фрагмента.
    text = " ".join(f"word{index}" for index in range(150))
    model = classifier_module.classifier
    model.return_value = [
        [
            # В первом фрагменте побеждает Legal.
            {"label": "Legal", "score": 0.55},
            {"label": "Financial", "score": 0.40},
            {"label": "News", "score": 0.02},
            {"label": "Technical", "score": 0.015},
            {"label": "General", "score": 0.01},
            {"label": "Scientific", "score": 0.005},
        ],
        [
            # Во втором побеждает News. Порядок классов отличается:
            # складывать оценки по позиции вместо названия нельзя.
            {"label": "News", "score": 0.60},
            {"label": "Financial", "score": 0.35},
            {"label": "Technical", "score": 0.02},
            {"label": "General", "score": 0.015},
            {"label": "Legal", "score": 0.01},
            {"label": "Scientific", "score": 0.005},
        ],
    ]

    result = classifier_module.classify_document(text)

    # По среднему побеждает Financial: (0.40 + 0.35) / 2 = 0.375.
    # Он не был победителем ни одного отдельного фрагмента.
    # Это проверяет объединение оценок, а не выбор одного фрагмента.
    assert result["label"] == "Financial"
    assert result["confidence"] == pytest.approx(0.375)
    assert [item["label"] for item in result["top_3"]] == [
        "Financial",
        "News",
        "Legal",
    ]
    # News: (0.02 + 0.60) / 2 = 0.31.
    # Legal: (0.55 + 0.01) / 2 = 0.28.
    assert [item["confidence"] for item in result["top_3"]] == pytest.approx(
        [0.375, 0.31, 0.28]
    )
    model.assert_called_once()
    assert len(model.call_args.args[0]) == 2
    assert model.call_args.kwargs["top_k"] is None
    assert model.call_args.kwargs["truncation"] is False
