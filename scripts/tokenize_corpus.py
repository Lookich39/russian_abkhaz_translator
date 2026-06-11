"""
Токенизирует все файлы данных с помощью обученной SentencePiece модели.
"""

import sentencepiece as spm
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ru_ab")
SPM_MODEL = os.path.join(DATA_DIR, "spm.model")

FILES_TO_TOKENIZE = [
    ("train.ru", "train.ru.sp"),
    ("train.ab", "train.ab.sp"),
    ("valid.ru", "valid.ru.sp"),
    ("valid.ab", "valid.ab.sp"),
    ("test.ru",  "test.ru.sp"),
    ("test.ab",  "test.ab.sp"),
]


def tokenize_file(sp, input_path, output_path):
    """Токенизирует файл построчно."""
    count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if line:
                pieces = sp.EncodeAsPieces(line)
                fout.write(' '.join(pieces) + '\n')
            else:
                fout.write('\n')
            count += 1
    return count


def main():
    print("Загружаю SentencePiece модель...")
    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_MODEL)

    for input_name, output_name in FILES_TO_TOKENIZE:
        input_path = os.path.join(DATA_DIR, input_name)
        output_path = os.path.join(DATA_DIR, output_name)

        count = tokenize_file(sp, input_path, output_path)
        print(f"  {input_name} -> {output_name} ({count} строк)")

    print("Готово!")


if __name__ == "__main__":
    main()