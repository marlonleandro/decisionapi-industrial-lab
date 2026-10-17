
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

predictions_file = (
    ROOT / "results/runs/v1_test/predictions.csv"
)

threshold_file = (
    ROOT / "results/selected_threshold.json"
)

with threshold_file.open(encoding="utf-8") as f:
    config = json.load(f)

threshold = float(config["selection"]["threshold"])

df = pd.read_csv(predictions_file)

required = {"image", "ground_truth", "probability"}
if not required.issubset(df.columns):
    raise ValueError("Faltan columnas requeridas")

# Evitar evaluar un test incompleto o con duplicados
if len(df) != 60 or df["image"].duplicated().any():
    raise ValueError("Test incompleto o con imágenes duplicadas")

df["probability"] = pd.to_numeric(
    df["probability"], errors="raise"
)

if df["probability"].isna().any():
    raise ValueError("Existen probabilidades vacías")

if not df["probability"].between(0, 1).all():
    raise ValueError("Probabilidades fuera de rango")

if set(df["ground_truth"]) != {"normal", "anomaly"}:
    raise ValueError("Ground truth inválido")

y_true = (df["ground_truth"] == "anomaly").astype(int)
y_pred = (df["probability"] >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

metrics = {
    "dataset": "VisA pcb1",
    "split": "test",
    "model": "gpt-6-luna",
    "prompt_version": "v1",
    "threshold": threshold,
    "samples": len(df),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "fpr": float(fp / (fp + tn)) if fp + tn else None,
    "fnr": float(fn / (fn + tp)) if fn + tp else None,
    "tp": int(tp),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn)
}

print("\n=== FINAL TEST — OPENAI DECISIONS API ===")
print(json.dumps(metrics, indent=2))

output = ROOT / "results/final_test_metrics.json"
output.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8"
)

print(f"\nResultados guardados en: {output}")
