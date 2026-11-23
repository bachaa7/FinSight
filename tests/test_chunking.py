import pytest

from finsight.ml.chunking import split_text


def test_empty_text():
    chunks = split_text("")
    assert chunks == []


def test_zero_max_words():
    with pytest.raises(ValueError):
        split_text("Some text", max_words=0)


def test_short_text():
    text = "This is a short document."
    chunks = split_text(text)
    assert chunks == [text]


def test_long_text():
    words = []

    for i in range(250):
        words.append(f"word{i}")
    text = " ".join(words)
    chunks = split_text(text)
    assert len(chunks) == 3

    for chunk in chunks:
        word_count = len(chunk.split())
        assert word_count >= 60 and word_count <= 120

    restored_text = " ".join(chunks)
    restored_words = restored_text.split()
    assert restored_words == words


def test_sentence_boundary():
    first_sentence = "First " * 69 + "end."
    second_sentence = "Second " * 79 + "end."
    text = f"{first_sentence} {second_sentence}"

    chunks = split_text(text)

    # Плановая граница — 75 слов, но ближайший подходящий конец
    # предложения находится после 70-го слова. Каждое предложение
    # должно целиком попасть в свой фрагмент.
    assert chunks == [first_sentence, second_sentence]
