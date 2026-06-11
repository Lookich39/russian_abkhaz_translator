"""
Фильтрация synthetic-корпуса после обратного перевода.
"""

import os
import re
import json
from collections import Counter, defaultdict
from itertools import zip_longest

import sentencepiece as spm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT_DIR = os.path.join(BASE_DIR, "data", "ab_ru")
SPM_MODEL = os.path.join(BASE_DIR, "data", "ru_ab", "spm.model")

SYNTH_AB_IN = os.path.join(BT_DIR, "synth.ab.sp")
SYNTH_RU_IN = os.path.join(BT_DIR, "synth.ru.sp")

SYNTH_AB_OUT = os.path.join(BT_DIR, "synth.filtered.ab.sp")
SYNTH_RU_OUT = os.path.join(BT_DIR, "synth.filtered.ru.sp")

REPORT_PATH = os.path.join(BT_DIR, "filter_report.json")

#  Пороговые значения 
MIN_TOKENS = 2
MAX_TOKENS = 120

LENGTH_RATIO_MIN = 0.35   # len(ru) / len(ab)
LENGTH_RATIO_MAX = 3.00

DROP_IF_UNK = True

MAX_REPEAT_RATIO = 0.40   # если один токен занимает > 40%
MAX_SAME_TOKEN_RUN = 4    # если подряд идёт один и тот же токен >= 4 раз

MAX_PUNCT_RATIO = 0.55
MAX_DIGIT_RATIO = 0.55


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def max_repeat_ratio(tokens):
    if not tokens:
        return 1.0
    counts = Counter(tokens)
    return max(counts.values()) / len(tokens)


def longest_same_token_run(tokens):
    if not tokens:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def get_filter_reason(ab_sp, ru_sp, sp):
    ab_sp = ab_sp.strip()
    ru_sp = ru_sp.strip()

    if not ab_sp or not ru_sp:
        return "empty"

    ab_tokens = ab_sp.split()
    ru_tokens = ru_sp.split()

    if len(ab_tokens) < MIN_TOKENS or len(ru_tokens) < MIN_TOKENS:
        return "too_short"

    if len(ab_tokens) > MAX_TOKENS or len(ru_tokens) > MAX_TOKENS:
        return "too_long"

    if DROP_IF_UNK and ("<unk>" in ab_tokens or "<unk>" in ru_tokens):
        return "contains_unk"

    ratio = len(ru_tokens) / max(len(ab_tokens), 1)
    if ratio < LENGTH_RATIO_MIN or ratio > LENGTH_RATIO_MAX:
        return "bad_length_ratio"

    if max_repeat_ratio(ru_tokens) > MAX_REPEAT_RATIO:
        return "high_repeat_ratio_ru"

    if longest_same_token_run(ru_tokens) >= MAX_SAME_TOKEN_RUN:
        return "long_same_token_run_ru"

    # Детокенизация
    ab_text = sp.DecodePieces(ab_tokens)
    ru_text = sp.DecodePieces(ru_tokens)

    ab_norm = normalize_text(ab_text)
    ru_norm = normalize_text(ru_text)

    if not ru_norm:
        return "empty_after_decode"

    # Если перевод совпал с исходником — скорее всего модель скопировала вход
    if ab_norm == ru_norm:
        return "copy_of_source"

    return None


def main():
    os.makedirs(BT_DIR, exist_ok=True)

    if not os.path.exists(SYNTH_AB_IN):
        raise FileNotFoundError(f"Не найден файл: {SYNTH_AB_IN}")
    if not os.path.exists(SYNTH_RU_IN):
        raise FileNotFoundError(f"Не найден файл: {SYNTH_RU_IN}")
    if not os.path.exists(SPM_MODEL):
        raise FileNotFoundError(f"Не найден SPM: {SPM_MODEL}")

    print("Загружаю SentencePiece...")
    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_MODEL)

    stats = Counter()
    examples = defaultdict(list)

    total_in = 0
    total_kept = 0

    with open(SYNTH_AB_IN, "r", encoding="utf-8") as fab, \
         open(SYNTH_RU_IN, "r", encoding="utf-8") as fru, \
         open(SYNTH_AB_OUT, "w", encoding="utf-8") as fout_ab, \
         open(SYNTH_RU_OUT, "w", encoding="utf-8") as fout_ru:

        for idx, (ab_line, ru_line) in enumerate(zip_longest(fab, fru), start=1):
            if ab_line is None or ru_line is None:
                raise RuntimeError(
                    "Файлы synth.ab.sp и synth.ru.sp имеют разное число строк"
                )

            ab_sp = ab_line.strip()
            ru_sp = ru_line.strip()
            total_in += 1

            reason = get_filter_reason(
                ab_sp=ab_sp,
                ru_sp=ru_sp,
                sp=sp,
            )

            if reason is None:
                fout_ab.write(ab_sp + "\n")
                fout_ru.write(ru_sp + "\n")
                total_kept += 1
            else:
                stats[reason] += 1
                if len(examples[reason]) < 3:
                    try:
                        ab_text = sp.DecodePieces(ab_sp.split()) if ab_sp else ""
                        ru_text = sp.DecodePieces(ru_sp.split()) if ru_sp else ""
                    except Exception:
                        ab_text, ru_text = ab_sp, ru_sp

                    examples[reason].append({
                        "ab": ab_text[:200],
                        "ru": ru_text[:200],
                    })

            if idx % 50000 == 0:
                print(f"  Обработано: {idx}")

    total_removed = total_in - total_kept


    print("\nФильтрация завершена.")
    print(f"  Вход:      {total_in}")
    print(f"  Оставлено: {total_kept}")
    print(f"  Удалено:   {total_removed}")
    print(f"  Файл ab:   {SYNTH_AB_OUT}")
    print(f"  Файл ru:   {SYNTH_RU_OUT}")
    print(f"  Отчёт:     {REPORT_PATH}")

    if stats:
        print("\nУдалено по причинам:")
        for reason, count in stats.most_common():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()