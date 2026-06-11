"""
Объединяет оригинальный параллельный корпус с синтетическим
"""

import os
import random

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_DIR = os.path.join(BASE_DIR, "data", "ru_ab")
DATA_DIR = os.path.join(BASE_DIR, "data", "ab_ru")
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")

# Сколько раз повторить оригинальные данные
ORIG_UPSAMPLE = 3

# Максимум синтетических строк (None = все)
MAX_SYNTH = None


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)

    # 1. Оригинальные данные
    orig_ru = read_lines(os.path.join(ORIG_DIR, "train.ru.sp"))
    orig_ab = read_lines(os.path.join(ORIG_DIR, "train.ab.sp"))

    assert len(orig_ru) == len(orig_ab), "Оригинальные файлы разной длины!"
    print(f"Оригинальный корпус: {len(orig_ru)} строк")

    # Upsampling оригинала
    if ORIG_UPSAMPLE > 1:
        orig_ru = orig_ru * ORIG_UPSAMPLE
        orig_ab = orig_ab * ORIG_UPSAMPLE
        print(f"  После upsampling x{ORIG_UPSAMPLE}: {len(orig_ru)} строк")

    # 2. Синтетические данные
    synth_ru_path = os.path.join(DATA_DIR, "synth.filtered.ru.sp")
    synth_ab_path = os.path.join(DATA_DIR, "synth.filtered.ab.sp")

    if not os.path.exists(synth_ru_path):
        print(f"ВНИМАНИЕ: {synth_ru_path} не найден!")
        print("Сначала запустите translate_mono.py")
        return

    synth_ru = read_lines(synth_ru_path)
    synth_ab = read_lines(synth_ab_path)

    assert len(synth_ru) == len(synth_ab), "Синтетические файлы разной длины!"
    print(f"Синтетический корпус: {len(synth_ru)} строк")

    if MAX_SYNTH and len(synth_ru) > MAX_SYNTH:
        indices = random.sample(range(len(synth_ru)), MAX_SYNTH)
        synth_ru = [synth_ru[i] for i in indices]
        synth_ab = [synth_ab[i] for i in indices]
        print(f"  Ограничено до: {len(synth_ru)} строк")

    # 3. Объединение
    merged_ru = orig_ru + synth_ru
    merged_ab = orig_ab + synth_ab

    # Перемешиваем
    combined = list(zip(merged_ru, merged_ab))
    random.shuffle(combined)
    merged_ru, merged_ab = zip(*combined)

    print(f"\nИтого train: {len(merged_ru)} строк")
    print(f"  Оригинал:    {len(orig_ru)}")
    print(f"  Синтетика:   {len(synth_ru)}")
    print(f"  Соотношение: {len(orig_ru)/len(merged_ru)*100:.1f}% / "
          f"{len(synth_ru)/len(merged_ru)*100:.1f}%")

    # 4. Сохранение
    write_lines(os.path.join(FINAL_DIR, "train.ru.sp"), merged_ru)
    write_lines(os.path.join(FINAL_DIR, "train.ab.sp"), merged_ab)

    # Valid — копируем оригинальный без изменений
    for lang in ("ru", "ab"):
        src = os.path.join(ORIG_DIR, f"valid.{lang}.sp")
        dst = os.path.join(FINAL_DIR, f"valid.{lang}.sp")
        write_lines(dst, read_lines(src))

    print(f"\nСохранено в {FINAL_DIR}/:")
    print(f"  train.ru.sp: {count_lines(os.path.join(FINAL_DIR, 'train.ru.sp'))}")
    print(f"  train.ab.sp: {count_lines(os.path.join(FINAL_DIR, 'train.ab.sp'))}")
    print(f"  valid.ru.sp: {count_lines(os.path.join(FINAL_DIR, 'valid.ru.sp'))}")
    print(f"  valid.ab.sp: {count_lines(os.path.join(FINAL_DIR, 'valid.ab.sp'))}")
    print("Готово!")


if __name__ == "__main__":
    main()