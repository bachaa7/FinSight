import spacy

sentence_splitter = spacy.blank("en")
sentence_splitter.add_pipe("sentencizer")


def get_text_statistics(text: str) -> dict:
    char_count = len(text)
    word_count = len(text.split())

    doc = sentence_splitter(text)
    sentence_count = 0
    for sentence in doc.sents:
        if sentence.text.strip():
            sentence_count += 1

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
    }
