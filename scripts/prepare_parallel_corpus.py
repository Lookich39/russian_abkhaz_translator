"""
Читает ab-ru-parallel.csv, чистит данные, делает split:
"""

import pandas as pd
import os
import re
import random
from sklearn.model_selection import train_test_split

random.seed(42)

# Настройки 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "ab-ru-parallel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "ru_ab")
os.makedirs(OUTPUT_DIR, exist_ok=True)


MAX_LEN = 200        # максимум символов в предложении
MIN_LEN = 2          # минимум символов
VALID_SIZE = 2500
TEST_SIZE = 2500

def clean_text(text):
    """Базовая очистка текста."""
    if not isinstance(text, str):
        return ""
    # убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    # убираем управляющие символы (кроме обычных пробелов)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text

def is_valid_length(text, min_len, max_len):
    return min_len <= len(text) <= max_len

def is_valid_pair(ru, ab):
    """Проверяет, что пара предложений годится для обучения."""
    if not is_valid_length(ru, MIN_LEN, MAX_LEN) or not is_valid_length(ab, MIN_LEN, MAX_LEN):
        return False

    # отношение длин не должно быть слишком большим
    ratio = len(ru) / len(ab)
    if ratio > 5.0 or ratio < 0.2:
        return False
    
    return True


def main():
    print("Читаю CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Всего строк в CSV: {len(df)}")

    # Проверяем колонки
    if 'ru' not in df.columns or 'ab' not in df.columns:
        raise ValueError(f"Ожидаются колонки 'ru' и 'ab', найдены: {list(df.columns)}")

    # Очистка
    print("Очищаю данные...")
    df = df.dropna(subset=['ru', 'ab'])
    df = df.drop_duplicates(subset=['ru', 'ab']).reset_index(drop=True)

    df['ru'] = df['ru'].apply(clean_text)
    df['ab'] = df['ab'].apply(clean_text)

    len_ru = df['ru'].str.len()
    len_ab = df['ab'].str.len()
    ratio = len_ru / len_ab

    # Маски для валидации
    mask_len = (len_ru >= MIN_LEN) & (len_ru <= MAX_LEN) & (len_ab >= MIN_LEN) & (len_ab <= MAX_LEN)
    mask_ratio = (ratio <= 5.0) & (ratio >= 0.2)

    # Фильтрация
    df = df[mask_len & mask_ratio].reset_index(drop=True)

    print(f" После очистки: {len(df)}")

    test_share = TEST_SIZE / len(df)
    train_valid_df, test_df = train_test_split(df, test_size= test_share, random_state=42, shuffle=True)

    valid_share = VALID_SIZE / len(df)
    train_df, valid_df = train_test_split(df, test_size= valid_share, random_state=42, shuffle=True)


    print(f"  Train: {len(train_df)}")
    print(f"  Valid: {len(valid_df)}")
    print(f"  Test:  {len(test_df)}")

    # Сохраняем
    for split_name, split_df in [("train", train_df),
                                  ("valid", valid_df),
                                  ("test", test_df)]:
        ru_path = os.path.join(OUTPUT_DIR, f"{split_name}.ru")
        ab_path = os.path.join(OUTPUT_DIR, f"{split_name}.ab")

        with open(ru_path, 'w', encoding='utf-8') as f:
            for text in split_df['ru']:
                f.write(text + '\n')

        with open(ab_path, 'w', encoding='utf-8') as f:
            for text in split_df['ab']:
                f.write(text + '\n')

        print(f"  Сохранено: {ru_path}, {ab_path}")

    # Также сохраняем объединённый файл для обучения SentencePiece
    combined_path = os.path.join(OUTPUT_DIR, "all_text.txt")
    with open(combined_path, 'w', encoding='utf-8') as f:
        for text in train_df['ru']:
            f.write(text + '\n')
        for text in train_df['ab']:
            f.write(text + '\n')
    print(f"  Объединённый текст для SPM: {combined_path}")
    print("Готово!")


if __name__ == "__main__":
    main()