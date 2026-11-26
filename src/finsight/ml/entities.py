import spacy

nlp = spacy.load("en_core_web_sm")

ENTITY_LABELS = {"ORG", "MONEY", "DATE", "PERCENT"}


def extract_entities(text: str) -> list[dict]:
    doc = nlp(text)
    entities = []
    for entity in doc.ents:
        if entity.label_ in ENTITY_LABELS:
            entities.append(
                {
                    "text": entity.text,
                    "label": entity.label_,
                    "start": entity.start_char,
                    "end": entity.end_char,
                }
            )

    return entities


if __name__ == "__main__":
    text = "Apple reported revenue of $90 billion in September 2025, an increase of 5%."
    result = extract_entities(text)
    print(result)
