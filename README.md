# Russian → Abkhaz Machine Translation Pipeline

🏆 **Результат на соревновании Yandex Data Dojo 2026:** 35 место из 320 участников (Итоговая метрика: **BLEU 11.6**)

Это проект для машинного перевода **с русского языка на абхазский** с использованием **OpenNMT-py**, **SentencePiece** и **Back-Translation**.

Проект выполнен **в рамках соревнования Yandex Data Dojo 2026** и реализует полный pipeline для обучения и применения модели перевода:
- подготовку параллельного корпуса,
- обучение обратной модели **ab → ru**,
- генерацию синтетического корпуса из моноязычных абхазских текстов,
- обучение финальной модели **ru → ab**,
- запуск готового решения через `solution.py` в формате соревнования.

---

## Возможности

- Перевод **с русского на абхазский**.
- Обучение обратной модели **с абхазского на русский**.
- Поддержка **Back-Translation** для расширения train-датасета.
- Общая токенизация через **SentencePiece**.
- Фильтрация synthetic-корпуса после генерации.
- Объединение original + synthetic корпуса для финального обучения.
- Дообучение финальной модели только на gold-данных (**gold-only finetuning**).
- Усреднение последних чекпоинтов.
- Поддержка запуска на **GPU**.
- Поддержка формата соревнования:
  - вход: `input.pickle`
  - выход: `output.json`

---

## Установка

Клонируйте репозиторий:

```bash
git clone https://github.com/Lookich39/russian_abkhaz_translator.git
cd ru_ab_mt
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

## Структура проекта

```text
project/
├── solution.py
├── README.md
├── requirements.txt
│
├── config/
│   ├── ab_ru/
│   │   ├── build_vocab.yaml
│   │   └── train.yaml
│   └── final/
│       ├── build_vocab.yaml
│       └── train.yaml
│       
│
├── scripts/
│   ├── train.py
│   ├── prepare_parallel_corpus.py
│   ├── train_spm.py
│   ├── tokenize_corpus.py
│   ├── prepare_ab_mono.py
│   ├── translate_mono.py
│   ├── filter_synthetic.py        # optional but recommended
│   └── merge_data.py
│   
│   
│
├── data/
│   ├── raw/
│   │   ├── ab-ru-parallel.csv
│   │   └── ab-mono.txt
│   │
│   ├── ru_ab/
│   │   ├── train.ru
│   │   ├── train.ab
│   │   ├── valid.ru
│   │   ├── valid.ab
│   │   ├── test.ru
│   │   ├── test.ab
│   │   ├── train.ru.sp
│   │   ├── train.ab.sp
│   │   ├── valid.ru.sp
│   │   ├── valid.ab.sp
│   │   ├── test.ru.sp
│   │   ├── test.ab.sp
│   │   ├── all_text.txt
│   │   ├── spm.model
│   │   ├── spm.vocab
│   │   └── vocab.src
│   │
│   ├── ab_ru/
│   │   ├── mono.ab.txt
│   │   ├── mono.ab.sp
│   │   ├── synth.ab.sp
│   │   ├── synth.ru.sp
│   │   ├── synth.filtered.ab.sp
│   │   └── synth.filtered.ru.sp
│   │   
│   │
│   └── final/
│       ├── train.ru.sp
│       ├── train.ab.sp
│       ├── valid.ru.sp
│       ├── valid.ab.sp
│       └── vocab.src
│
└── models/
    ├── ab_ru/
    │   └── ab_ru_step_*.pt
    └── final/
        └── final_step_*.pt
```

---

## Использование
## Подготовка

Поместите исходные данные в папку `data/raw/`:

```text
data/raw/ab-ru-parallel.csv
data/raw/ab-mono.txt
```

### Формат `ab-ru-parallel.csv`

```csv
ru,ab
Чемпионат Европы по теннису ..., ...
Иисус даже отдал за людей жизнь ..., ...
```

### Формат `ab-mono.txt`

Один абхазский текст на строку.

---

## Запуск обучения

### Полный pipeline

```bash
python scripts/train.py --stage all
```

Это выполнит:

- подготовку параллельного корпуса,
- обучение SentencePiece,
- токенизацию,
- построение словаря,
- обучение обратной модели **ab → ru**,
- подготовку моноязычного корпуса,
- перевод моно-корпуса,
- фильтрацию synthetic-корпуса,
- объединение original + synthetic,
- построение словаря для финальной модели,
- обучение финальной модели **ru → ab**.

---

### Только обратная модель

```bash
python scripts/train.py --stage ab_ru
```

---

### Только финальная модель

```bash
python scripts/train.py --stage final
```

---

## Поэтапный запуск

### Этап 1. Обратная модель `ab → ru`

```bash
python scripts/prepare_parallel_corpus.py
python scripts/train_spm.py
python scripts/tokenize_corpus.py
onmt_build_vocab -config config/ab_ru/build_vocab.yaml -n_sample -1
onmt_train -config config/ab_ru/train.yaml
```

### Этап 2. Финальная модель `ru → ab`

```bash
python scripts/prepare_ab_mono.py
python scripts/translate_mono.py
python scripts/filter_synthetic.py
python scripts/merge_data.py
onmt_build_vocab -config config/final/build_vocab.yaml -n_sample -1
onmt_train -config config/final/train.yaml
```

---


## Инференс

Финальный inference выполняется через `solution.py`.

### Вход

Файл:

```text
input.pickle
```

Формат:

```python
[
    {
        "rid": int,
        "src": str,
    },
]
```

### Выход

Файл:

```text
output.json
```

Формат:

```json
[
  {
    "rid": 1,
    "translation": "..."
  }
]
```

### Запуск

```bash
python solution.py
```

---

## Минимальный набор файлов для платформы

Если модель уже обучена и нужно перевести текст, достаточно оставить:

```text
project/
├── solution.py
├── requirements.txt
├── models/
│   └── final/
│       └── final_step_*.pt
└── data/
    └── ru_ab/
        └── spm.model
```

После запуска `solution.py` будет создан файл:

```text
output.json
```

---


## Основные команды

### Полный запуск
```bash
python scripts/train.py --stage all
```

### Обучение обратной модели
```bash
python scripts/train.py --stage ab_ru
```

### Обучение финальной модели
```bash
python scripts/train.py --stage final
```

### Инференс
```bash
python solution.py
```

---

## License

Copyright © 2026 Alexey Kudryavtsev. See LICENSE for details.


---



