
from pathlib import Path
import json
import time
import argparse

import pandas as pd
import torch

from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "results/yolo/pcb1_cls_v1/weights/best.pt"
)

DATASET = ROOT / "data/yolo_pcb1"
OUTPUT = ROOT / "results/yolo/pcb1_cls_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50
    )
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold fuera de rango")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(MODEL_PATH))

    print("Classes:", model.names)

    # Localizar la clase anomaly sin asumir
    # que siempre tendrá índice 0.
    anomaly_ids = [
        k for k, v in model.names.items()
        if v.lower() == "anomaly"
    ]

    if len(anomaly_ids) != 1:
        raise ValueError(
            "No se encontró una clase anomaly única"
        )

    anomaly_id = anomaly_ids[0]

    rows = []

    for ground_truth in ["normal", "anomaly"]:
        folder = DATASET / args.split / ground_truth

        images = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in [
                ".jpg", ".jpeg", ".png", ".webp"
            ]
        ])

        for image in images:
            start = time.perf_counter()

            result = model.predict(
                source=str(image),
                imgsz=320,
                device=device,
                verbose=False
            )[0]

            if device != "cpu":
                torch.cuda.synchronize()

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            probability = float(
                result.probs.data[anomaly_id].item()
            )

            prediction = (
                "anomaly"
                if probability >= args.threshold
                else "normal"
            )

            rows.append({
                "image": image.name,
                "ground_truth": ground_truth,
                "probability": probability,
                "prediction": prediction,
                "latency_ms": round(latency_ms, 3)
            })

    df = pd.DataFrame(rows)

    y_true = (df.ground_truth == "anomaly").astype(int)
    y_pred = (df.prediction == "anomaly").astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "YOLO11n-cls",
        "split": args.split,
        "threshold": args.threshold,
        "samples": len(df),
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
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "avg_latency_ms": float(
            df.latency_ms.mean()
        )
    }

    output_csv = OUTPUT / f"predictions_{args.split}.csv"
    output_json = OUTPUT / f"metrics_{args.split}.json"

    df.to_csv(output_csv, index=False)

    output_json.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("\n=== YOLO PCB EVALUATION ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nPredictions: {output_csv}")


if __name__ == "__main__":
    main()
