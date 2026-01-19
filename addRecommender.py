
import argparse
import os
import re
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import random

# При желании приглушить логи TF:
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

random.seed(42)
np.random.seed(42)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

# ----- колоноки и S/I/R -----
CANONICAL_COLS = {
    "diagnosis": ["диагноз", "diagnosis", "diag", "dx"],
    "microbe": ["микроорганизм", "микроорганизмы", "микроорганизм(ы)", "microbe", "organism", "pathogen"],
    "antibiotic": ["антибиотик", "antibiotic", "ab", "drug"],
    "sensitivity": ["чувствительность", "sensitivity", "ast", "результат", "результат_чувствительности"],
}

SENS_MAP = {
    "r": 0, "резистентен": 0, "резистентный": 0, "устойчив": 0, "нечувствителен": 0, "resistant": 0, "susceptibility:resistant": 0,
    "i": 1, "intermediate": 1, "умеренно чувствителен": 1, "промежуточная": 1, "пограничная": 1,
    "s": 2, "susceptible": 2, "чувствителен": 2, "высокочувствителен": 2, "susceptibility:susceptible": 2,
}

def normalize_colname(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip().lower().replace("ё", "е")
    return re.sub(r"[\s\-\(\)\[\]{}]+", "", name)

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_cols = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        cand_norm = normalize_colname(cand)
        if cand_norm in norm_cols:
            return norm_cols[cand_norm]
    for cand in candidates:
        cand_norm = normalize_colname(cand)
        for n, orig in norm_cols.items():
            if cand_norm in n:
                return orig
    return None

def map_sensitivity(value) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip().lower().replace("ё", "е").replace(".", "")
    s = re.sub(r"\s+", " ", s)
    if s in SENS_MAP:
        return SENS_MAP[s]
    for token in [" s ", " i ", " r "]:
        if token in f" {s} ":
            return SENS_MAP[token.strip()]
    if "резист" in s or "устойчив" in s or "resist" in s:
        return 0
    if "промеж" in s or "умерен" in s or "intermed" in s:
        return 1
    if "чувств" in s or "suscept" in s:
        return 2
    try:
        num = float(s.replace(",", "."))
        if num <= 0.5:
            return 0
        elif 0.5 < num < 1.5:
            return 1
        else:
            return 2
    except:
        return None

def build_dataset(df: pd.DataFrame, diag_col: str, micro_col: str, ab_col: str, sens_col: str) -> pd.DataFrame:
    work = df[[diag_col, micro_col, ab_col, sens_col]].copy()
    work = work.dropna(subset=[diag_col, micro_col, ab_col, sens_col])
    for c in [diag_col, micro_col, ab_col]:
        work[c] = work[c].astype(str).str.strip()
    work["sens_cls"] = work[sens_col].apply(map_sensitivity)
    work = work[work["sens_cls"].notna()].copy()
    work["sens_cls"] = work["sens_cls"].astype(int)
    work = work.drop_duplicates(subset=[diag_col, micro_col, ab_col, "sens_cls"]).reset_index(drop=True)
    return work

# ---------- сериализуемый стандартайзер ----------
@register_keras_serializable(package="custom")
def custom_standardize(x):
    lowercase = tf.strings.lower(x)
    lowercase = tf.strings.regex_replace(lowercase, "ё", "е")
    cleaned = tf.strings.regex_replace(lowercase, r"[^a-zа-я0-9\s\-\+/#]", " ")
    cleaned = tf.strings.regex_replace(cleaned, r"\s+", " ")
    return cleaned

# ---------- сериализуемая модель ----------
@register_keras_serializable(package="custom")
class TextModel(keras.Model):
    def __init__(self, vocab_size=30000, seq_len=64, emb_dim=64, hidden_dim=128, **kwargs):
        super().__init__(**kwargs) 
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)

        self.vectorizer = layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.seq_len,
            standardize=custom_standardize, 
        )
        self.embedding = layers.Embedding(self.vocab_size, self.emb_dim, mask_zero=True)
        self.pool = layers.GlobalAveragePooling1D()
        self.hidden = layers.Dense(self.hidden_dim, activation="relu")
        self.out = layers.Dense(3, activation="softmax")

    def adapt(self, text_ds):
        self.vectorizer.adapt(text_ds)

    def call(self, inputs, training=False):
        x = self.vectorizer(inputs)
        x = self.embedding(x)
        x = self.pool(x)
        x = self.hidden(x)
        return self.out(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "emb_dim": self.emb_dim,
            "hidden_dim": self.hidden_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ---------- Рекомендер ----------
class Recommender:
    def __init__(self, model_dir: str = "./saved_model_ab",
                 vocab_size: int = 30000, seq_len: int = 64, emb_dim: int = 64, hidden_dim: int = 128):
        self.model_dir = model_dir
        self.model: Optional[keras.Model] = None
        self.meta: Dict = {
            "cols": {}, "vocab_size": vocab_size, "seq_len": seq_len, "emb_dim": emb_dim, "hidden_dim": hidden_dim
        }
        self._antibiotics: List[str] = []
        self._col_diag = None
        self._col_micro = None
        self._col_ab = None
        self._col_sens = None
        self._df = None

    def _resolve_columns(self, df: pd.DataFrame):
        diag_col = find_column(df, CANONICAL_COLS["diagnosis"])
        micro_col = find_column(df, CANONICAL_COLS["microbe"])
        ab_col   = find_column(df, CANONICAL_COLS["antibiotic"])
        sens_col = find_column(df, CANONICAL_COLS["sensitivity"])
        missing = []
        if not diag_col:  missing.append("Диагноз")
        if not micro_col: missing.append("Микроорганизм")
        if not ab_col:    missing.append("Антибиотик")
        if not sens_col:  missing.append("Чувствительность")
        if missing:
            raise ValueError(f"Не удалось обнаружить колонки в Excel: {', '.join(missing)}")
        self._col_diag, self._col_micro, self._col_ab, self._col_sens = diag_col, micro_col, ab_col, sens_col
        self.meta["cols"] = {"diagnosis": diag_col, "microbe": micro_col, "antibiotic": ab_col, "sensitivity": sens_col}

    def load_data(self, excel_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl", dtype=str)
        if isinstance(df, dict):
            df = list(df.values())[0]
        self._resolve_columns(df)
        work = build_dataset(df, self._col_diag, self._col_micro, self._col_ab, self._col_sens)
        self._antibiotics = sorted(work[self._col_ab].unique().tolist())
        self._df = work
        return work

    @staticmethod
    def _make_text_row(diag: str, micro: str, ab: str) -> str:
        return f"[DIAG] {diag} [MICROBE] {micro} [ANTIBIOTIC] {ab}"

    def _build_texts_and_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        texts = [self._make_text_row(d, m, a) for d, m, a in zip(df[self._col_diag], df[self._col_micro], df[self._col_ab])]
        y = df["sens_cls"].values.astype("int32")
        return np.array(texts), y

    def train(self, epochs: int = 15, batch_size: int = 256, validation_split: float = 0.15, learning_rate: float = 2e-3) -> keras.callbacks.History:
        if self._df is None:
            raise RuntimeError("Сначала вызовите load_data().")
        X_texts, y = self._build_texts_and_labels(self._df)
        ds = tf.data.Dataset.from_tensor_slices((X_texts, y)).shuffle(len(X_texts), seed=42)

        model = TextModel(vocab_size=self.meta["vocab_size"], seq_len=self.meta["seq_len"],
                          emb_dim=self.meta["emb_dim"], hidden_dim=self.meta["hidden_dim"])

        # адаптация словаря
        text_only_ds = ds.map(lambda x, y: x)
        model.adapt(text_only_ds.batch(512))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        callbacks = [keras.callbacks.EarlyStopping(monitor="val_acc", patience=3, mode="max", restore_best_weights=True)]

        # Keras 3 / optree: строки как dtype=object
        X_np = np.array(X_texts, dtype=object)
        y_np = y.astype("int32")

        history = model.fit(X_np, y_np, epochs=epochs, batch_size=batch_size,
                            validation_split=validation_split, callbacks=callbacks, verbose=2)
        self.model = model
        return history

    def save(self):
        if self.model is None:
            raise RuntimeError("Модель ещё не обучена.")
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "model.keras")
        self.model.save(model_path, include_optimizer=True)
        with open(os.path.join(self.model_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.model_dir, "antibiotics.txt"), "w", encoding="utf-8") as f:
            for ab in self._antibiotics:
                f.write(ab + "\n")

    def load(self):
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Не найден каталог модели: {self.model_dir}")
        model_path = os.path.join(self.model_dir, "model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Не найден файл модели: {model_path}")
        self.model = keras.models.load_model(model_path,
                                             custom_objects={"TextModel": TextModel, "custom_standardize": custom_standardize})
        with open(os.path.join(self.model_dir, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        ab_path = os.path.join(self.model_dir, "antibiotics.txt")
        if os.path.exists(ab_path):
            with open(ab_path, "r", encoding="utf-8") as f:
                self._antibiotics = [line.strip() for line in f if line.strip()]

    def available_antibiotics(self) -> List[str]:
        return list(self._antibiotics)

    def _score_antibiotics(self, diag: str, microbe: str, antibiotics: List[str]) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Модель не загружена/не обучена. Вызовите train() или load().")
        texts = [self._make_text_row(diag, microbe, ab) for ab in antibiotics]
        probs = self.model.predict(np.array(texts, dtype=object), verbose=0)
        p_r, p_i, p_s = probs[:, 0], probs[:, 1], probs[:, 2]
        expected = 0.0 * p_r + 1.0 * p_i + 2.0 * p_s
        return pd.DataFrame({"Антибиотик": antibiotics, "P(R)": p_r, "P(I)": p_i, "P(S)": p_s,
                             "Ожидаемый_балл": expected}).sort_values("Ожидаемый_балл", ascending=False).reset_index(drop=True)

    def recommend(self, diag: str, microbe: str, top_k: int = 20) -> pd.DataFrame:
        if not self._antibiotics:
            raise RuntimeError("Список антибиотиков пуст. Загрузите данные через load_data() или файл antibiotics.txt.")
        return self._score_antibiotics(diag=diag, microbe=microbe, antibiotics=self._antibiotics).head(top_k)

# ---------- интерактив ----------
def interactive_cli(rec: "Recommender", excel_path: str, sheet_name: Optional[str], topk: int, recs_out: str):
    try:
        rec.load()
    except Exception as e:
        print("Не удалось загрузить модель. Сначала обучите её без --interactive.")
        print(str(e))
        return

    try:
        rec.load_data(excel_path, sheet_name=sheet_name)
    except Exception as e:
        print("Ошибка чтения Excel. Проверьте путь и лист.")
        print(str(e))
        return

    print("\nИнтерактивный режим. Пустая строка — выход.\n")
    while True:
        diag = input("Диагноз: ").strip()
        if not diag:
            print("Выход.")
            break
        micro = input("Микроорганизм: ").strip()
        if not micro:
            print("Выход.")
            break
        try:
            recs = rec.recommend(diаг=diag, microbe=micro, top_k=topk)
        except TypeError:
            # опечатка diag vs диаг -> исправим
            recs = rec.recommend(diag=diag, microbe=micro, top_k=topk)
        except Exception as e:
            print("Ошибка рекомендации:", e)
            continue
        print("\nТоп-", min(topk, len(recs)), " для случая:", sep="")
        print("Диагноз:", diag)
        print("Микроорганизм:", micro)
        print(recs.to_string(index=False, formatters={
            "P(R)": lambda x: f"{x:.3f}", "P(I)": lambda x: f"{x:.3f}",
            "P(S)": lambda x: f"{x:.3f}", "Ожидаемый_балл": lambda x: f"{x:.3f}",
        }))
        try:
            recs.to_excel(recs_out, index=False)
            print(f"\nСохранено: {recs_out}\n" + "-" * 60)
        except Exception as e:
            print("Не удалось сохранить Excel:", e)
            print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="Обучение и рекомендации антибиотиков по диагнозу и микроорганизму.")
    parser.add_argument("--excel", type=str, required=True, help="Путь к Excel-файлу с данными.")
    parser.add_argument("--sheet", type=str, default=None, help="Имя листа Excel (если None — первый лист).")
    parser.add_argument("--model_dir", type=str, default="./saved_model_ab", help="Каталог для сохранения/загрузки модели.")
    parser.add_argument("--recs_out", type=str, default="./recommendations_example.xlsx", help="Куда сохранить рекомендации.")
    parser.add_argument("--train_epochs", type=int, default=15, help="Число эпох обучения.")
    parser.add_argument("--batch_size", type=int, default=256, help="Размер батча.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Доля валидации.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Скорость обучения.")
    parser.add_argument("--just_predict", action="store_true", help="Только загрузить модель и сделать пример рекомендации.")
    parser.add_argument("--interactive", action="store_true", help="Интерактивный режим: ввод с клавиатуры.")
    parser.add_argument("--topk", type=int, default=20, help="Сколько лучших антибиотиков показать.")
    args = parser.parse_args()

    rec = Recommender(model_dir=args.model_dir)

    if args.interactive:
        interactive_cli(rec, args.excel, args.sheet, args.topk, args.recs_out)
        return

    if args.just_predict:
        rec.load()
        df = rec.load_data(args.excel, sheet_name=args.sheet)
        sample_row = df.iloc[0]
        diag = str(sample_row[rec.meta["cols"]["diagnosis"]])
        micro = str(sample_row[rec.meta["cols"]["microbe"]])
        recs = rec.recommend(diag=diag, microbe=micro, top_k=args.topk)
        print("Пример рекомендации для:\nДиагноз:", diag, "\nМикроорганизм:", micro)
        print(recs.head(args.topk))
        recs.to_excel(args.recs_out, index=False)
        print(f"Рекомендации сохранены в: {args.recs_out}")
        return

    # Обучение → сохранение → пример
    df = rec.load_data(args.excel, sheet_name=args.sheet)
    print(f"Загружено записей после очистки: {len(df)}")
    rec.train(epochs=args.train_epochs, batch_size=args.batch_size, validation_split=args.val_split, learning_rate=args.lr)
    rec.save()
    print(f"Модель сохранена в: {args.model_dir}")

    sample_row = df.iloc[0]
    diag = str(sample_row[rec.meta["cols"]["diagnosis"]])
    micro = str(sample_row[rec.meta["cols"]["microbe"]])
    recs = rec.recommend(diag=diag, microbe=micro, top_k=args.topk)
    print("Пример рекомендации для:\nДиагноз:", diag, "\nМикроорганизм:", micro)
    print(recs.head(args.topk))
    recs.to_excel(args.recs_out, index=False)
    print(f"Рекомендации сохранены в: {args.recs_out}")

if __name__ == "__main__":
    main()
