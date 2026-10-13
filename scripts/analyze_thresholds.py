
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
INPUT = ROOT / "results" / "predictions_v2.csv"
OUTPUT = ROOT / "results" / "threshold_analysis_v2.csv"

THRESHOLDS = [round(x / 100, 2) for x in range(5, 96, 5)]

def main():
    df = pd.read_csv(INPUT)

    # Excluir errores de API y respuestas no evaluables
    df = df[
        df["ground_truth"].isin(["normal", "anomaly"])
    ].copy()

    df["probability"] = pd.to_numeric(
        df["probability"], errors="coerce"
    )
    df = df.dropna(subset=["probability"])

    y_true = (df["ground_truth"] == "anomaly").astype(int)
    results = []

    for threshold in THRESHOLDS:
        y_pred = (df["probability"] >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "fpr": fp / (fp + tn) if fp + tn else 0,
            "fnr": fn / (fn + tp) if fn + tp else 0,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT, index=False)

    print("\n=== Threshold Analysis ===\n")
    print(result_df.to_string(index=False))
    print(f"\nSaved: {OUTPUT}")

    print("\n=== Lowest False Negative Rate ===")
    min_fnr = result_df["fnr"].min()
    candidates = result_df[result_df["fnr"] == min_fnr]
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
