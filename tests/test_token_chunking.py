import pytest


@pytest.fixture
def character_tokenizer(classifier_module):
    """Имитировать много токенов даже в одном слове, без загрузки нейросети."""
    # Здесь каждый символ считается токеном. Ещё два токена имитируют
    # [CLS] и [SEP]: они тоже должны помещаться в лимит модели.
    tokenizer = classifier_module.classifier.tokenizer
    tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [0] * (len(text) + 2)}
    return tokenizer


def test_exact_token_limit_keeps_original(classifier_module, character_tokenizer):
    text = "alpha beta"
    limit = len(text) + 2

    assert classifier_module.split_chunk_by_tokens(
        text, character_tokenizer, limit
    ) == [text]


def test_word_splitting_preserves_order(classifier_module, character_tokenizer):
    text = "alpha beta gamma delta epsilon zeta eta theta"
    limit = 15

    chunks = classifier_module.split_chunk_by_tokens(text, character_tokenizer, limit)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
    assert all(
        len(character_tokenizer(chunk)["input_ids"]) <= limit for chunk in chunks
    )
    assert " ".join(chunks).split() == text.split()


@pytest.mark.parametrize("text", ["abcdefghij" * 100, "!" * 1300])
def test_text_without_spaces_keeps_every_character(
    classifier_module, character_tokenizer, text
):
    chunks = classifier_module.split_chunk_by_tokens(text, character_tokenizer)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
    assert all(len(character_tokenizer(chunk)["input_ids"]) <= 512 for chunk in chunks)
    assert "".join(chunks) == text


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_impossible_limit_stops_instead_of_recursing_forever(
    classifier_module, character_tokenizer, limit
):
    with pytest.raises(ValueError, match="Не удалось разделить текст"):
        classifier_module.split_chunk_by_tokens("abc", character_tokenizer, limit)


def test_classifier_averages_all_additional_chunks(
    classifier_module, character_tokenizer
):
    """Дополнительные части действительно доходят до модели и входят в среднее."""
    # 120 слов дают один исходный блок, но более 512 тестовых токенов.
    text = " ".join(["alpha"] * 120)
    model = classifier_module.classifier
    model.return_value = [
        [{"label": "News", "score": 0.9}, {"label": "Legal", "score": 0.1}],
        [{"label": "News", "score": 0.3}, {"label": "Legal", "score": 0.7}],
    ]

    result = classifier_module.classify_document(text)

    model.assert_called_once()
    chunks = model.call_args.args[0]
    assert len(chunks) == 2
    assert " ".join(chunks) == text
    assert all(len(character_tokenizer(chunk)["input_ids"]) <= 512 for chunk in chunks)
    assert model.call_args.kwargs["truncation"] is False
    assert result["label"] == "News"
    # Делить нужно на два новых блока, а не на один исходный.
    assert result["confidence"] == pytest.approx(0.6)
    assert [item["confidence"] for item in result["top_3"]] == pytest.approx([0.6, 0.4])
