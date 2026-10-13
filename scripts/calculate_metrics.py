from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

INPUT_FILE = (
    ROOT
    / "results"
    / "predictions_v2.csv"
)


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    df = df[
        df["prediction"].isin(
            [
                "normal",
                "anomaly"
            ]
        )
    ]

    y_true = (
        df["ground_truth"]
        == "anomaly"
    ).astype(int)

    y_pred = (
        df["prediction"]
        == "anomaly"
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred
        ).ravel()
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp)
        else 0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0
    )

    print()
    print(
        "=== OpenAI Decisions "
        "PCB Benchmark ==="
    )

    print(
        f"Samples   : {len(df)}"
    )

    print(
        f"Accuracy  : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1        : "
        f"{f1:.4f}"
    )

    print()
    print(
        f"TP: {tp}"
    )

    print(
        f"TN: {tn}"
    )

    print(
        f"FP: {fp}"
    )

    print(
        f"FN: {fn}"
    )

    print()
    print(
        "False Negative Rate: "
        f"{false_negative_rate:.4f}"
    )

    print(
        "False Positive Rate: "
        f"{false_positive_rate:.4f}"
    )

    print()
    print(
        "Average latency: "
        f"{df['latency_ms'].mean():.2f} ms"
    )


if __name__ == "__main__":
    main()