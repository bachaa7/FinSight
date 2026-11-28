import pytest


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", {"char_count": 0, "word_count": 0, "sentence_count": 0}),
        ("   ", {"char_count": 3, "word_count": 0, "sentence_count": 0}),
        ("\n\t", {"char_count": 2, "word_count": 0, "sentence_count": 0}),
        ("Hello world!", {"char_count": 12, "word_count": 2, "sentence_count": 1}),
        (
            "Hello world! How are you?",
            {"char_count": 25, "word_count": 5, "sentence_count": 2},
        ),
        ("Hello world", {"char_count": 11, "word_count": 2, "sentence_count": 1}),
        ("One.\nTwo!", {"char_count": 9, "word_count": 2, "sentence_count": 2}),
        ("Привет, мир!", {"char_count": 12, "word_count": 2, "sentence_count": 1}),
        (
            "Value 3.14. Next sentence.",
            {"char_count": 26, "word_count": 4, "sentence_count": 2},
        ),
        (
            "Dr. Smith arrived. He left.",
            {"char_count": 27, "word_count": 5, "sentence_count": 2},
        ),
    ],
)
def test_statistics_with_real_sentencizer(statistics_module, text, expected):
    """Проверяем реальные правила spaCy, включая числа, сокращения и пробелы."""
    assert statistics_module.get_text_statistics(text) == expected


def test_statistics_are_calculated_from_current_saved_text(
    api_module, api_session, stored_document
):
    """Проверить обработчик и Pydantic без HTTP-клиента и внешней БД."""
    from finsight.api.schemas import DocumentDetailResponse

    stored_document.text = "Hello world! How are you?"
    stored_document.word_count = 5
    api_session.get.return_value = stored_document

    result = api_module.get_document(stored_document.id, api_session)
    data = DocumentDetailResponse.model_validate(result).model_dump(mode="json")
    assert data["statistics"] == {
        "char_count": 25,
        "word_count": 5,
        "sentence_count": 2,
    }
    assert data["statistics"]["word_count"] == data["word_count"]
    assert data["text"] == stored_document.text
    # Старый документ не обязан иметь результаты новых моделей.
    assert data["entities"] is None and data["sentiment"] is None
    api_session.add.assert_not_called()
    api_session.commit.assert_not_called()
