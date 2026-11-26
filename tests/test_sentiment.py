import pytest


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_is_rejected_before_model(sentiment_module, text):
    with pytest.raises(ValueError, match="Текст не должен быть пустым"):
        sentiment_module.analyze_sentiment(text)
    sentiment_module.sentiment_analyzer.assert_not_called()


@pytest.mark.parametrize(
    "values, label, score",
    [
        ((0.8, 0.1, 0.1), "positive", 0.7),
        ((0.1, 0.8, 0.1), "negative", -0.7),
        ((0.1, 0.2, 0.7), "neutral", -0.1),
    ],
)
def test_label_confidence_and_balance(sentiment_module, values, label, score):
    positive, negative, neutral = values
    # Порядок не совпадает с порядком классов: объединение идёт по label.
    sentiment_module.sentiment_analyzer.return_value = [
        [
            {"label": "neutral", "score": neutral},
            {"label": "positive", "score": positive},
            {"label": "negative", "score": negative},
        ]
    ]
    result = sentiment_module.analyze_sentiment("Example financial text.")
    assert result["label"] == label
    assert result["confidence"] == max(values)
    assert result["score"] == pytest.approx(score)
    assert result["probabilities"] == {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
    }
    assert sentiment_module.sentiment_analyzer.call_args.kwargs["truncation"] is False


def test_multiple_chunks_are_averaged_by_label(sentiment_module):
    text = " ".join(f"word{index}" for index in range(150))
    analyzer = sentiment_module.sentiment_analyzer
    analyzer.return_value = [
        [
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.1},
            {"label": "neutral", "score": 0.1},
        ],
        [
            {"label": "negative", "score": 0.7},
            {"label": "neutral", "score": 0.2},
            {"label": "positive", "score": 0.1},
        ],
    ]
    result = sentiment_module.analyze_sentiment(text)
    chunks = analyzer.call_args.args[0]
    assert len(chunks) == 2
    assert " ".join(chunks).split() == text.split()
    assert result["probabilities"] == pytest.approx(
        {"positive": 0.45, "negative": 0.4, "neutral": 0.15}
    )
    assert result["label"] == "positive"
    assert result["confidence"] == pytest.approx(0.45)
    assert result["score"] == pytest.approx(0.05)


def test_long_chunk_uses_finbert_tokenizer_without_loading_classifier(sentiment_module):
    analyzer = sentiment_module.sentiment_analyzer
    # 120 слов — один исходный блок, но больше 512 тестовых токенов.
    analyzer.tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": [0] * (len(text) + 2)
    }
    text = " ".join(["alpha"] * 120)
    analyzer.return_value = [analyzer.return_value[0], analyzer.return_value[0]]
    result = sentiment_module.analyze_sentiment(text)
    chunks = analyzer.call_args.args[0]
    assert len(chunks) == 2
    assert " ".join(chunks) == text
    assert all(len(analyzer.tokenizer(chunk)["input_ids"]) <= 512 for chunk in chunks)
    assert result["probabilities"]["positive"] == pytest.approx(0.7)
    assert result["score"] == pytest.approx(0.6)


def test_sentiment_endpoint_does_not_write_files_or_database(
    api_client,
    api_session,
    upload_directory,
    sentiment_module,
    classifier_module,
    entities_module,
):
    response = api_client.post("/sentiment", json={"text": "Example financial text."})
    assert response.status_code == 200
    assert response.json()["label"] == "positive"
    assert response.json()["score"] == pytest.approx(0.6)
    api_session.add.assert_not_called()
    api_session.commit.assert_not_called()
    assert not upload_directory.exists()
    classifier_module.classifier.assert_not_called()
    entities_module.nlp.assert_not_called()
    sentiment_module.sentiment_analyzer.assert_called_once()


@pytest.mark.parametrize(
    "payload", [{}, {"text": ""}, {"text": "  "}, {"text": "x" * 20001}]
)
def test_invalid_sentiment_request(api_client, sentiment_module, payload):
    response = api_client.post("/sentiment", json=payload)
    assert response.status_code == 422
    sentiment_module.sentiment_analyzer.assert_not_called()


def test_sentiment_failure_prevents_upload_writes(
    api_client, api_session, sentiment_module, upload_directory
):
    sentiment_module.sentiment_analyzer.side_effect = ValueError(
        "test sentiment failure"
    )
    response = api_client.post(
        "/documents",
        files={"file": ("example.txt", b"Example financial text.", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "test sentiment failure"}
    api_session.add.assert_not_called()
    assert not upload_directory.exists()
