"""
Переводит моноязычные абхазские тексты на русский
с помощью обратной модели.
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import glob
import time
import inspect
import tempfile

import sentencepiece as spm
import torch

from onmt.translate import Translator, GNMTGlobalScorer
from onmt.model_builder import load_test_model
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser as OnmtArgumentParser


# Патч torch.load
_original_torch_load = torch.load


def _patched_torch_load(f, map_location=None, pickle_module=None,
                        weights_only=None, mmap=None, **kwargs):
    return _original_torch_load(f, map_location=map_location,
                                weights_only=False, **kwargs)


torch.load = _patched_torch_load


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ab_ru")
SPM_MODEL = os.path.join(BASE_DIR, "data", "ru_ab", "spm.model")
MODELS_DIR = os.path.join(BASE_DIR, "models", "ab_ru")

CHUNK_SIZE = 15000
BEAM_SIZE = 1
BATCH_SIZE = 64
MAX_LENGTH = 200


def find_best_ab_ru_model():
    pattern = os.path.join(MODELS_DIR, "*_step_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        raise FileNotFoundError(
            f"Обратная модель не найдена: {pattern}\n"
            "Сначала обучите обратную модель."
        )

    def _step(p):
        try:
            return int(
                os.path.basename(p)
                .replace(".pt", "")
                .split("_step_")[-1]
            )
        except ValueError:
            return 0

    return max(checkpoints, key=_step)


def load_translator(model_path, gpu_id):
    parser = OnmtArgumentParser()
    opts.translate_opts(parser)

    args = [
        "-model", model_path,
        "-src", "dummy",
        "-beam_size", str(BEAM_SIZE),
        "-max_length", str(MAX_LENGTH),
        "-batch_size", str(BATCH_SIZE),
    ]
    if gpu_id >= 0 and torch.cuda.is_available():
        args += ["-gpu", str(gpu_id)]

    opt = parser.parse_args(args)
    fields, model, model_opt = load_test_model(opt)
    scorer = GNMTGlobalScorer.from_opt(opt)
    dummy_out = open(os.devnull, "w", encoding="utf-8")

    translator = Translator.from_opt(
        model, fields, opt, model_opt,
        global_scorer=scorer,
        out_file=dummy_out,
        report_align=False,
        report_score=False,
        logger=None,
    )
    
    sig = set(inspect.signature(translator.translate).parameters.keys())
    return translator, sig, dummy_out


def translate_chunk(translator, sig, dummy_out, ab_lines_tokenized, gpu_id):
    """Переводит уже токенизированные абхазские строки."""
    src_fd, src_path = tempfile.mkstemp(suffix=".src.txt")
    out_path = src_path + ".out.txt"

    try:
        with os.fdopen(src_fd, "w", encoding="utf-8") as f:
            for line in ab_lines_tokenized:
                f.write(line + "\n")

        out_f = open(out_path, "w", encoding="utf-8")
        translator.out_file = out_f

        try:
            kwargs = {}
            if "src" in sig: kwargs["src"] = src_path
            if "tgt" in sig: kwargs["tgt"] = None
            if "batch_size" in sig: kwargs["batch_size"] = BATCH_SIZE
            if "batch_type" in sig: kwargs["batch_type"] = "sents"
            if "attn_debug" in sig: kwargs["attn_debug"] = False
            if "align_debug" in sig: kwargs["align_debug"] = False
            translator.translate(**kwargs)
        finally:
            out_f.close()
            translator.out_file = dummy_out

        with open(out_path, "r", encoding="utf-8") as f:
            ru_sp_lines = [line.strip() for line in f]

        return ru_sp_lines

    finally:
        for p in (src_path, out_path):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def main():
    gpu_id = 0 if torch.cuda.is_available() else -1

    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_MODEL)

    model_path = find_best_ab_ru_model()
    translator, sig, dummy_out = load_translator(model_path, gpu_id)

    # Читаем уже токенизированные абхазские строки
    mono_sp_path = os.path.join(DATA_DIR, "mono.ab.sp")
    if not os.path.exists(mono_sp_path):
        print(f"Файл не найден: {mono_sp_path}")
        print("Сначала запустите prepare_ab_mono.py")
        return

    with open(mono_sp_path, "r", encoding="utf-8") as f:
        mono_lines = [line.strip() for line in f if line.strip()]

    total = len(mono_lines)
    print(f"Моноязычных строк: {total}")

    all_ru_sp = []
    all_ab_sp = []
    t0 = time.time()

    for i in range(0, total, CHUNK_SIZE):
        chunk = mono_lines[i:i + CHUNK_SIZE]
        chunk_num = i // CHUNK_SIZE + 1
        total_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE

        print(f"  Чанк {chunk_num}/{total_chunks} "
              f"({len(chunk)} строк)...", end=" ", flush=True)

        ct = time.time()
        ru_sp_lines = translate_chunk(
            translator, sig, dummy_out, chunk, gpu_id
        )

        ab_sp_lines = chunk

        for ab_sp, ru_sp in zip(ab_sp_lines, ru_sp_lines):
            if ru_sp.strip() and ab_sp.strip():
                all_ab_sp.append(ab_sp)
                all_ru_sp.append(ru_sp)

        elapsed_chunk = time.time() - ct
        print(f"{elapsed_chunk:.1f}с")

    elapsed_total = time.time() - t0
    print(f"\nВсего переведено: {len(all_ru_sp)} строк "
          f"за {elapsed_total:.0f}с ({elapsed_total/60:.1f} мин)")

    synth_ru = os.path.join(DATA_DIR, "synth.ru.sp")
    synth_ab = os.path.join(DATA_DIR, "synth.ab.sp")

    with open(synth_ru, "w", encoding="utf-8") as f:
        for line in all_ru_sp:
            f.write(line + "\n")

    with open(synth_ab, "w", encoding="utf-8") as f:
        for line in all_ab_sp:
            f.write(line + "\n")

    print(f"Синтетический корпус:")
    print(f"  {synth_ru} ({len(all_ru_sp)} строк)")
    print(f"  {synth_ab} ({len(all_ab_sp)} строк)")
    print("Готово!")


if __name__ == "__main__":
    main()