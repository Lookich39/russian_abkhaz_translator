"""
Обучает общую SentencePiece модель на объединённом корпусе.
Общая модель, потому что оба языка используют кириллицу.
"""

import sentencepiece as spm
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ru_ab")
INPUT_FILE = os.path.join(DATA_DIR, "all_text.txt")
MODEL_PREFIX = os.path.join(DATA_DIR, "spm")

VOCAB_SIZE = 12000
CHARACTER_COVERAGE = 0.995
MODEL_TYPE = "unigram"  #unigram лучше для морфологически языков


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Обучаю SentencePiece модель...")
    print(f"  Входной файл: {INPUT_FILE}")
    print(f"  Размер словаря: {VOCAB_SIZE}")
    print(f"  Тип модели: {MODEL_TYPE}")

    spm.SentencePieceTrainer.train(
        input=INPUT_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=CHARACTER_COVERAGE,
        model_type=MODEL_TYPE,
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
        num_threads=4,
    )

    print(f"  Модель сохранена: {MODEL_PREFIX}.model")
    print(f"  Словарь сохранён: {MODEL_PREFIX}.vocab")

    # Проверка
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{MODEL_PREFIX}.model")

    test_ru = "Чемпионат Европы по теннису"
    test_ab = "Европа Ачемпионат"

    print(f"\n  Тест токенизации:")
    print(f"    RU: '{test_ru}'")
    print(f"    -> {sp.EncodeAsPieces(test_ru)}")
    print(f"    AB: '{test_ab}'")
    print(f"    -> {sp.EncodeAsPieces(test_ab)}")

    print("Готово!")


if __name__ == "__main__":
    main()