"""Проверяем свой код NER; качество готовых весов этими тестами не измеряется."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("label", ["ORG", "MONEY", "DATE", "PERCENT"])
def test_supported_entity_type_is_returned(entities_module, label):
    entities_module.nlp.return_value = SimpleNamespace(
        ents=[SimpleNamespace(text="value", label_=label, start_char=2, end_char=7)]
    )
    assert entities_module.extract_entities("  value") == [
        {"text": "value", "label": label, "start": 2, "end": 7}
    ]


@pytest.mark.parametrize("label", ["PERSON", "GPE", "CARDINAL", "PRODUCT"])
def test_other_entity_types_are_excluded(entities_module, label):
    entities_module.nlp.return_value = SimpleNamespace(
        ents=[SimpleNamespace(text="value", label_=label, start_char=0, end_char=5)]
    )
    assert entities_module.extract_entities("value") == []


@pytest.mark.parametrize("text", ["", "   ", "A document with no entities."])
def test_no_entities_returns_empty_list(entities_module, text):
    assert entities_module.extract_entities(text) == []
    entities_module.nlp.assert_called_once_with(text)


def test_offsets_and_repeated_mentions_preserve_original_text(entities_module):
    text = " \nApple — Apple"
    entities_module.nlp.return_value = SimpleNamespace(
        ents=[
            SimpleNamespace(text="Apple", label_="ORG", start_char=2, end_char=7),
            SimpleNamespace(text="Apple", label_="ORG", start_char=10, end_char=15),
        ]
    )
    items = entities_module.extract_entities(text)
    assert [(item["start"], item["end"]) for item in items] == [(2, 7), (10, 15)]
    assert all(text[item["start"] : item["end"]] == item["text"] for item in items)
    # Перед моделью не делаем strip/нормализацию: иначе позиции съедут.
    entities_module.nlp.assert_called_once_with(text)


def test_entities_are_saved_and_read_without_running_ner_again(
    api_client, api_session, entities_module
):
    text = " Apple received $5 million."
    entities_module.nlp.return_value = SimpleNamespace(
        ents=[
            SimpleNamespace(text="Apple", label_="ORG", start_char=1, end_char=6),
            SimpleNamespace(
                text="$5 million", label_="MONEY", start_char=16, end_char=26
            ),
        ]
    )
    response = api_client.post(
        "/documents", files={"file": ("ner.txt", text.encode(), "text/plain")}
    )
    assert response.status_code == 200
    document = api_session.add.call_args.args[0]
    assert document.entities == response.json()["entities"]
    assert len(document.entities) == 2
    assert all(text[e["start"] : e["end"]] == e["text"] for e in document.entities)
    api_session.get.return_value = document
    detail = api_client.get(f"/documents/{document.id}")
    assert detail.status_code == 200
    assert detail.json()["entities"] == document.entities
    entities_module.nlp.assert_called_once_with(text)


def test_ner_error_happens_before_file_or_database_write(
    api_client, api_session, entities_module, upload_directory
):
    entities_module.nlp.side_effect = ValueError("test NER failure")
    response = api_client.post(
        "/documents", files={"file": ("ner.txt", b"Example text.", "text/plain")}
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "test NER failure"}
    assert not upload_directory.exists()
    api_session.add.assert_not_called()
    api_session.commit.assert_not_called()


@pytest.mark.parametrize("value", [None, []])
def test_response_distinguishes_unprocessed_and_empty(api_client, api_session, value):
    from datetime import UTC, datetime
    from uuid import uuid4

    from finsight.models import Document

    item = Document(
        id=uuid4(),
        filename="old.txt",
        content_type="text/plain",
        storage_path="old.txt",
        text="Example.",
        word_count=1,
        classification={
            "label": "General",
            "confidence": 0.7,
            "top_3": [{"label": "General", "confidence": 0.7}],
        },
        entities=value,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    api_session.get.return_value = item
    result = api_client.get(f"/documents/{item.id}")
    assert result.status_code == 200
    assert result.json()["entities"] == value
