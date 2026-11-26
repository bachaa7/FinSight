from math import ceil


def split_text(text: str, max_words: int = 120) -> list[str]:
    """Разбить текст на почти равные фрагменты по словам.

    text — исходный текст; max_words — максимальная длина фрагмента в словах.
    Пустой текст даёт пустой список. При max_words=120 текст короче 60 слов
    остаётся одним коротким фрагментом. max_words должен быть положительным.

    Если подходящий конец предложения близок к плановой границе,
    фрагмент заканчивается на нём; иначе граница остаётся плановой.
    """

    if max_words <= 0:
        raise ValueError("Значение должно быть больше 0")

    words = text.split()
    chunks = []

    if not words:
        return []

    # Позиция ПОСЛЕ слова с . ! ?; она подходит как конец среза words[start:end].
    # Например, знак после слова с индексом 4 даёт границу 5.
    sentence_ends = []

    for word_index, word in enumerate(words):
        if word.endswith((".", "!", "?")):
            sentence_ends.append(word_index + 1)

    # Сколько фрагментов нужно, чтобы каждый был не длиннее max_words.
    # 250 слов при max_words=120 требуют ceil(250 / 120) = 3 фрагмента.
    chunk_count = ceil(len(words) / max_words)

    # Фактическое начало текущего фрагмента: сначала 0, затем конец предыдущего.
    start = 0

    for chunk_index in range(chunk_count):
        # Остаток от start делим между всеми ещё не созданными фрагментами.
        # chunks_left включает текущий: для первого из трёх оно равно 3.
        remaining_words = len(words) - start

        chunks_left = chunk_count - chunk_index

        planned_size = remaining_words // chunks_left

        # Плановая граница: для 250 слов первый end будет 83.
        end = start + planned_size

        # Сколько фрагментов будет ПОСЛЕ текущего (на первом шаге из трёх — 2).
        chunks_after_current = chunk_count - chunk_index - 1

        # Сюда попадут только подходящие концы предложений.
        candidate_ends = []

        for sentence_end in sentence_ends:
            # Длина текущего фрагмента, ЕСЛИ закончить его на sentence_end.
            current_chunk_words = sentence_end - start
            # Сколько слов останется после такой возможной границы.
            words_after_boundary = len(words) - sentence_end

            # Текущий фрагмент должен иметь 60..max_words слов. Оставшихся
            # слов должно хватить для остальных фрагментов по 60..max_words.
            if (
                (current_chunk_words <= max_words)
                and (current_chunk_words >= 60)
                and (words_after_boundary >= chunks_after_current * 60)
                and (words_after_boundary <= max_words * chunks_after_current)
            ):
                candidate_ends.append(sentence_end)

        if candidate_ends:
            best = candidate_ends[0]
            best_distance = abs(best - end)
            for pos in candidate_ends:
                distance = abs(pos - end)
                if distance < best_distance:
                    best_distance = distance
                    best = pos

            end = best

        # end теперь равен выбранной границе предложения либо плановой границе.
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start = end

    return chunks


def split_chunk_by_tokens(text: str, tokenizer, max_tokens: int = 512) -> list[str]:
    encoded = tokenizer(text, truncation=False, verbose=False)
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

    left_chunks = split_chunk_by_tokens(left_text, tokenizer, max_tokens)
    right_chunks = split_chunk_by_tokens(right_text, tokenizer, max_tokens)
    return left_chunks + right_chunks
