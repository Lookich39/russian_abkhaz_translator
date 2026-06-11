# scripts/prepare_ab_mono.py
"""
Подготавливает моноязычный абхазский корпус для обратного перевода.
Создаёт:
- data/ab_ru/mono.ab.txt  — сырой текст
- data/ab_ru/mono.ab.sp   — токенизированный текст (для translate_mono.py)
"""

import os
import re
import random
import sentencepiece as spm

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONO_FILE = os.path.join(BASE_DIR, "data", "raw", "ab-mono.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "ab_ru")
PARALLEL_DIR = os.path.join(BASE_DIR, "data", "ru_ab")

MONO_LIMIT = 500_000
MONO_MIN_LEN = 5
MONO_MAX_LEN = 300

MAX_PUNCT_RATIO = 0.55
MAX_DIGIT_RATIO = 0.55

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text

def char_ratio(text, predicate):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    return sum(1 for c in chars if predicate(c)) / len(chars)

def is_punct(ch):
    return ch in ".,!?;:()[]{}«»\"'`-—–…/\\|"

def is_digit(ch):
    return ch.isdigit()

def has_any_letter(text):
    return any(ch.isalpha() for ch in text)

def load_original_ab_set():
    """Загружает все ab-строки из оригинального train/valid/test."""
    all_ab = set()
    for split in ("train", "valid", "test"):
        path = os.path.join(PARALLEL_DIR, f"{split}.ab")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_ab.add(line)
    return all_ab


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MONO_FILE):
        print(f"Файл не найден: {MONO_FILE}")
        return

    print("Загружаю оригинальный ab-корпус ...")
    original_ab_set = load_original_ab_set()

    print(f"Читаю {MONO_FILE} (лимит: {MONO_LIMIT})...")
    clean_lines = []
    total_read = 0

    with open(MONO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            total_read += 1
            text = clean_text(line)
            if len(text) < MONO_MIN_LEN or len(text) > MONO_MAX_LEN:
                continue

            if not has_any_letter(text):
                continue
            
            if char_ratio(text, is_punct) > MAX_PUNCT_RATIO:
                continue

            if char_ratio(text, is_digit) > MAX_DIGIT_RATIO:
                continue

            clean_lines.append(text)

            if len(clean_lines) >= MONO_LIMIT:
                break


    print(f"  Прочитано: {total_read}")
    print(f"  После фильтрации: {len(clean_lines)}")

    unique_lines = list(dict.fromkeys(clean_lines))  # сохраняет порядок
    print(f"  После удаления дубликатов: {len(unique_lines)}")

    filtered_lines = [line for line in unique_lines if line not in original_ab_set]
    removed_leakage = len(unique_lines) - len(filtered_lines)
    print(f"  Удалено пересечений с original: {removed_leakage}")
    print(f"  Итого к переводу: {len(filtered_lines)}")

    random.shuffle(clean_lines)

    # Сохраняем текст
    mono_txt_path = os.path.join(OUTPUT_DIR, "mono.ab.txt")
    mono_sp_path = os.path.join(OUTPUT_DIR, "mono.ab.sp")

    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(BASE_DIR, "data", "ru_ab", "spm.model"))

    with open(mono_txt_path, "w", encoding="utf-8") as f_raw, \
         open(mono_sp_path, "w", encoding="utf-8") as f_sp:
        for line in filtered_lines:
            f_raw.write(line + "\n")

            pieces = sp.EncodeAsPieces(line)
            f_sp.write(" ".join(pieces) + "\n")
    
    print(f"  Сохранено: {mono_txt_path}")        
    print(f"  Токенизировано: {mono_sp_path}")
    print("Готово!")

if __name__ == "__main__":
    main()