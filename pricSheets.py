# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

# >>> УКАЖИТЕ СВОЙ ПУТЬ К ИСХОДНОМУ ФАЙЛУ <<<
XLSX_PATH = r"C:\Users\Mila\Desktop\курсач\Диагнозы_20_24_чувствительность_к_АБ_2 (1).xlsx"
# >>> КУДА СОХРАНИТЬ «ПЛОСКИЙ» ФАЙЛ <<<
OUT_PATH  = r"C:\Users\Mila\Desktop\курсач\prepared_all.xlsx"

def norm(s):
    return re.sub(r"\s+", "", str(s).strip().lower().replace("ё", "е"))

DIAG_CANDS   = {norm(x) for x in ["Диагноз","diag","diagnosis","dx"]}
MICRO_CANDS  = {norm(x) for x in ["Микроорганизм","Микроорганизмы","microbe","organism","pathogen"]}
VALID_SIR = {"ч","и","р","s","i","r","Ч","И","Р"}

def find_col(df, cands):
    m = {norm(c): c for c in df.columns}
    for k, v in m.items():
        if k in cands:
            return v
    for k, v in m.items():
        for c in cands:
            if c in k:
                return v
    return None

def looks_like_sensitivity_col(series):
    s = series.dropna().astype(str).str.strip()
    if len(s) == 0:
        return False
    ok = s.apply(lambda x: x in VALID_SIR)
    return ok.mean() >= 0.6

def map_sir(x):
    if pd.isna(x):
        return np.nan
    t = str(x).strip().lower()
    return {"ч":"S","и":"I","р":"R","s":"S","i":"I","r":"R"}.get(t, np.nan)

all_parts = []

xls = pd.ExcelFile(XLSX_PATH, engine="openpyxl")
for sheet in xls.sheet_names:
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet, engine="openpyxl", dtype=str)
    if df.empty:
        continue

    col_diag  = find_col(df, DIAG_CANDS)
    col_micro = find_col(df, MICRO_CANDS)
    if not col_diag or not col_micro:
        continue

    antibiotic_cols = []
    for c in df.columns:
        if c in [col_diag, col_micro]:
            continue
        if looks_like_sensitivity_col(df[c]):
            antibiotic_cols.append(c)

    if not antibiotic_cols:
        continue

    part = df.melt(
        id_vars=[col_micro, col_diag],
        value_vars=antibiotic_cols,
        var_name="Антибиотик",
        value_name="Чувствительность"
    )
    part["Чувствительность"] = part["Чувствительность"].map(map_sir)
    part = part.dropna(subset=["Чувствительность"])
    part = part.rename(columns={col_diag:"Диагноз", col_micro:"Микроорганизм"})
    part["Лист"] = sheet
    all_parts.append(part)

if not all_parts:
    raise SystemExit("Не найдено ни одного листа с (Диагноз/Микроорганизм + АБ-колонки).")

result = pd.concat(all_parts, ignore_index=True)
result = result.dropna(subset=["Диагноз","Микроорганизм"])

result.to_excel(OUT_PATH, index=False)
print("✅ Готово:", OUT_PATH)
print("Строк:", len(result), " | Листы:", sorted(set(result["Лист"]))[:10], "...")
print(result.head(10))
