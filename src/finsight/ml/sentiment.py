from transformers import pipeline

from .chunking import split_chunk_by_tokens, split_text

sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")


def analyze_sentiment(text: str) -> dict:
    if not text.strip():
        raise ValueError("Текст не должен быть пустым")

    chunks = split_text(text)
    safe_chunks = []
    for chunk in chunks:
        safe_chunks.extend(split_chunk_by_tokens(chunk, sentiment_analyzer.tokenizer))
    chunks = safe_chunks

    scores = sentiment_analyzer(chunks, top_k=None, truncation=False)

    score_totals = {}
    for chunk_scores in scores:
        for item in chunk_scores:
            score_totals[item["label"]] = (
                score_totals.get(item["label"], 0.0) + item["score"]
            )

    probabilities = {}
    for name, total in score_totals.items():
        probabilities[name] = total / len(chunks)

    label = max(probabilities, key=probabilities.get)
    confidence = probabilities[label]
    score = probabilities["positive"] - probabilities["negative"]

    return {
        "label": label,
        "confidence": confidence,
        "score": score,
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    text = "The company reported strong revenue growth and higher profits."
    result = analyze_sentiment(text)
    print(result)
