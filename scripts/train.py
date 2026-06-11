# scripts/train.py
"""
Обёртка для запуска полного пайплайна обучения.
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\nОШИБКА: {desc}")
        sys.exit(1)
    print(f"\n  ✓ {desc}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ab_ru", "final", "all"],
                        default="all")
    args = parser.parse_args()

    stages = {
        "ab_ru": [
            ("python scripts/prepare_parallel_corpus.py", "Подготовка параллельных данных"),
            ("python scripts/train_spm.py", "Обучение SentencePiece"),
            ("python scripts/tokenize_corpus.py", "Токенизация параллельного корпуса"),
            ("onmt_build_vocab -config config/ab_ru/build_vocab.yaml -n_sample -1", "Построение базового словаря"),
            ("onmt_train -config config/ab_ru/train.yaml", "Обучение обратной модели"),
        ],

        "final": [
            ("python scripts/prepare_ab_mono.py", "Подготовка моно-корпуса"),
            ("python scripts/translate_mono.py", "Перевод моно ab→ru"),
            ("python scripts/filter_synthetic.py", "Фильтрация synthetic-корпуса"),
            ("python scripts/merge_data.py", "Объединение данных"),
            ("onmt_build_vocab -config config/final/build_vocab.yaml -n_sample -1",
             "Словарь финальной модели"),
            ("onmt_train -config config/final/train.yaml",
             "Обучение финальной модели"),
        ],
    }

    if args.stage == "all":
        order = ["ab_ru", "final"]
    else:
        order = [args.stage]

    for stage in order:
        print(f"\n{'#'*60}")
        print(f"  ЭТАП: {stage}")
        print(f"{'#'*60}")
        for cmd, desc in stages[stage]:
            run(cmd, desc)

    print(f"\n{'='*60}")
    print("  ВСЁ ЗАВЕРШЕНО!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()