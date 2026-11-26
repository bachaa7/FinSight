from pathlib import Path

from transformers import pipeline

from .chunking import split_text

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "distilbert_doc_classifier"


classifier = pipeline(
    "text-classification",
    model=str(MODEL_DIR),
)


def split_chunk_by_tokens(text: str, max_tokens: int = 512) -> list[str]:
    encoded = classifier.tokenizer(text, truncation=False, verbose=False)
    token_count = len(encoded["input_ids"])
    if token_count <= max_tokens:
        return [text]
    words = text.split()
    if len(words) > 1:
        middle = len(words) // 2
        left_text = " ".join(words[:middle])
        right_text = " ".join(words[middle:])
    else:
        if len(text) <= 1:
            raise ValueError("Не удалось разделить текст в пределах лимита токенов")
        middle = len(text) // 2
        left_text = text[:middle]
        right_text = text[middle:]

    left_chunks = split_chunk_by_tokens(left_text, max_tokens)
    right_chunks = split_chunk_by_tokens(right_text, max_tokens)
    return left_chunks + right_chunks


def classify_document(text: str) -> dict:
    chunks = split_text(text)

    if not chunks:
        raise ValueError("Текст не должен быть пустым")

    safe_chunks = []
    for chunk in chunks:
        safe_chunks.extend(split_chunk_by_tokens(chunk))
    chunks = safe_chunks

    result = classifier(chunks, top_k=None, truncation=False)

    score_totals = {}

    for chunk_scores in result:
        for i in chunk_scores:
            score_totals[i["label"]] = score_totals.get(i["label"], 0.0) + i["score"]

    score_averages = {}

    for name, score in score_totals.items():
        score_averages[name] = score / len(chunks)

    sorted_score = sorted(
        score_averages.items(), key=lambda pair: pair[1], reverse=True
    )

    best = sorted_score[0]

    label = best[0]
    confidence = best[1]
    top3 = sorted_score[:3]

    top_3 = []

    for class_name, class_score in top3:
        top_3.append({"label": class_name, "confidence": class_score})

    classification_result = {}

    classification_result["label"] = label
    classification_result["confidence"] = confidence
    classification_result["top_3"] = top_3

    return classification_result


if __name__ == "__main__":
    text = (
        "Under this Agreement, the Borrower shall repay the principal amount of the Loan, together with accrued interest, on or before the Maturity Date. "
        "The Borrower shall make each payment in United States dollars by wire transfer to the account designated by the Lender. "
        "If any payment is not received when due, the outstanding balance shall bear interest at the Default Rate until paid in full. "
        "This Agreement shall be governed by the laws of the State of New York. Under this Agreement, the Borrower shall repay the principal amount of the Loan, together with accrued interest, on or before the Maturity Date. "
        "The Borrower shall make each payment in United States dollars by wire transfer to the account designated by the Lender. "
        "If any payment is not received when due, the outstanding balance shall bear interest at the Default Rate until paid in full. "
        "This Agreement shall be governed by the laws of the State of New York."
    )

    print(classify_document(text))
