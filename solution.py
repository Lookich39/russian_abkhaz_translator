"""
Переводчик русский -> абхазский.
Вход:  input.pickle  — [{"rid": int, "src": str}, ...]
Выход: output.json   — [{"rid": int, "translation": str}, ...]
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import glob
import time
import tempfile
import inspect
import pickle
import json

import sentencepiece as spm
import torch

from onmt.translate import Translator, GNMTGlobalScorer
from onmt.model_builder import load_test_model
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser as OnmtArgumentParser

# Патч torch.load для PyTorch 2.6+
_original_torch_load = torch.load


def _patched_torch_load(f, map_location=None, pickle_module=None,
                        weights_only=None, mmap=None, **kwargs):
    return _original_torch_load(
        f, map_location=map_location, weights_only=False, **kwargs
    )


torch.load = _patched_torch_load


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPM_PATH = os.path.join(BASE_DIR, "data", "ru_ab", "spm.model")
MODELS_DIR = os.path.join(BASE_DIR, "models", "final")
INPUT_PATH = os.path.join(BASE_DIR, "example-input.pickle")
OUTPUT_PATH = os.path.join(BASE_DIR, "example-output.json")


class RuAbTranslator:

    def __init__(self, model_path=None, spm_model_path=None, gpu_id=0):
        if model_path is None:
            model_path = self._find_best_model(MODELS_DIR)
        if spm_model_path is None:
            spm_model_path = SPM_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        if not os.path.exists(spm_model_path):
            raise FileNotFoundError(f"SPM не найдена: {spm_model_path}")

        self.model_path = os.path.abspath(model_path)
        self.gpu_id = gpu_id
        self.device = torch.device(
            f"cuda:{gpu_id}"
            if gpu_id >= 0 and torch.cuda.is_available()
            else "cpu"
        )

        print(f"Загружаю SentencePiece: {spm_model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_model_path)

        print(f"Загружаю модель:  {self.model_path}")
        print(f"Устройство:       {self.device}")

        self._load_translator()
        print("Переводчик готов!\n")

    def _find_best_model(self, models_dir):
        pattern = os.path.join(models_dir, "*_step_*.pt")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            raise FileNotFoundError(
                f"Чекпоинты не найдены: {pattern}"
            )

        def _step(p):
            try:
                name = os.path.basename(p).replace(".pt", "")
                return int(name.split("_step_")[-1])
            except (ValueError, IndexError):
                return 0

        best = max(checkpoints, key=_step)
        print(f"Автовыбор чекпоинта: {best}")
        return best

    def _load_translator(self):
        parser = OnmtArgumentParser()
        opts.translate_opts(parser)

        translate_args = [
            "-model", self.model_path,
            "-src", "dummy",
            "-beam_size", "8",
            "-max_length", "200",
            "-batch_size", "32",
        ]
        if self.gpu_id >= 0 and torch.cuda.is_available():
            translate_args += ["-gpu", str(self.gpu_id)]

        opt = parser.parse_args(translate_args)
        self.opt = opt

        fields, model, model_opt = load_test_model(opt)
        self.fields = fields
        self.model_opt = model_opt

        scorer = GNMTGlobalScorer.from_opt(opt)
        self._dummy_out = open(os.devnull, "w", encoding="utf-8")

        self.translator = Translator.from_opt(
            model, fields, opt, model_opt,
            global_scorer=scorer,
            out_file=self._dummy_out,
            report_align=False,
            report_score=False,
            logger=None,
        )

        self._translate_sig = set(
            inspect.signature(self.translator.translate).parameters.keys()
        )

    def _tokenize(self, texts):
        return [
            " ".join(self.sp.EncodeAsPieces(t.strip()))
            for t in texts
        ]

    def _detokenize(self, lines):
        return [
            self.sp.DecodePieces(line.split())
            for line in lines
        ]

    def translate_batch(self, texts, beam_size=4, max_length=200,
                        batch_size=32):
        if not texts:
            return []

        tokenized = self._tokenize(texts)

        src_fd, src_path = tempfile.mkstemp(suffix=".src.txt")
        out_path = src_path + ".out.txt"

        try:
            with os.fdopen(src_fd, "w", encoding="utf-8") as f:
                for line in tokenized:
                    f.write(line + "\n")

            out_f = open(out_path, "w", encoding="utf-8")
            self.translator.out_file = out_f

            try:
                kwargs = {}
                if "src" in self._translate_sig:
                    kwargs["src"] = src_path
                if "tgt" in self._translate_sig:
                    kwargs["tgt"] = None
                if "batch_size" in self._translate_sig:
                    kwargs["batch_size"] = batch_size
                if "batch_type" in self._translate_sig:
                    kwargs["batch_type"] = "sents"
                if "attn_debug" in self._translate_sig:
                    kwargs["attn_debug"] = False
                if "align_debug" in self._translate_sig:
                    kwargs["align_debug"] = False

                self.translator.translate(**kwargs)

            finally:
                out_f.close()
                self.translator.out_file = self._dummy_out

            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise RuntimeError(f"Файл перевода пуст: {out_path}")

            with open(out_path, "r", encoding="utf-8") as f:
                raw = [line.strip() for line in f]

            return self._detokenize(raw)

        finally:
            for p in (src_path, out_path):
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    def translate(self, text):
        result = self.translate_batch([text])
        return result[0] if result else ""


def main():
    print(f"Читаю {INPUT_PATH} ...")
    with open(INPUT_PATH, "rb") as f:
        rows = pickle.load(f)

    print(f"Всего примеров: {len(rows)}")

    rids = [row["rid"] for row in rows]
    texts = [row["src"] for row in rows]

    gpu_id = 0 if torch.cuda.is_available() else -1
    translator = RuAbTranslator(gpu_id=gpu_id)

    print("Начинаю перевод...")
    t0 = time.time()
    translations = translator.translate_batch(texts)
    elapsed = time.time() - t0

    print(f"Перевод завершён за {elapsed:.1f} с ({elapsed / 60:.1f} мин)")

    results = [
        {"rid": rid, "translation": translation}
        for rid, translation in zip(rids, translations)
    ]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Результат: {OUTPUT_PATH}")

    print("\nПримеры:")
    for i in range(min(5, len(results))):
        print(f"  rid:         {results[i]['rid']}")
        print(f"  src:         {texts[i]}")
        print(f"  translation: {results[i]['translation']}\n")


if __name__ == "__main__":
    main()