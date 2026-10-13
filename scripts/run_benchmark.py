from __future__ import annotations

import csv
import json
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)


from src.decisions.client import (
    inspect_pcb_visible_defect
)


INPUT_CSV = (
    ROOT
    / "data"
    / "benchmark_pcb1.csv"
)

OUTPUT_CSV = (
    ROOT
    / "results"
    / "predictions_v2.csv"
)

RAW_DIR = (
    ROOT
    / "results"
    / "raw_v2"
)


# SOLO PARA EL PRIMER BENCHMARK
TEMP_THRESHOLD = 0.50


def classify(
    probability: float
) -> str:

    if probability >= TEMP_THRESHOLD:
        return "anomaly"

    return "normal"


def main():

    load_dotenv(
        ROOT / ".env"
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with INPUT_CSV.open(
        encoding="utf-8"
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    results = []

    total = len(rows)

    for index, row in enumerate(
        rows,
        start=1
    ):

        image_path = Path(
            row["image"]
        )

        ground_truth = row[
            "ground_truth"
        ]

        print()
        print(
            f"[{index}/{total}] "
            f"{image_path.name}"
        )

        print(
            f"Ground truth: "
            f"{ground_truth}"
        )

        start = time.perf_counter()

        try:

            decision = (
                inspect_pcb_visible_defect(
                    image_path,
                    prompt_version="v2"
                )
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            latency_ms = round(
                elapsed * 1000,
                2
            )

            answer = (
                decision.answers[0]
            )

            if answer.type == "refusal":

                print(
                    "REFUSAL"
                )

                results.append({
                    "image": str(image_path),
                    "ground_truth": ground_truth,
                    "probability": "",
                    "prediction": "refusal",
                    "threshold": TEMP_THRESHOLD,
                    "latency_ms": latency_ms,
                    "correct": False,
                    "error": "refusal"
                })

                continue

            probability = float(
                answer.probability
            )

            prediction = classify(
                probability
            )

            correct = (
                prediction
                == ground_truth
            )

            print(
                f"P(defect): "
                f"{probability:.4f}"
            )

            print(
                f"Prediction: "
                f"{prediction}"
            )

            print(
                f"Correct: "
                f"{correct}"
            )

            print(
                f"Latency: "
                f"{latency_ms} ms"
            )

            raw_file = (
                RAW_DIR
                / f"{index:03d}_"
                  f"{image_path.stem}.json"
            )

            raw_file.write_text(
                json.dumps(
                    decision.model_dump(
                        mode="json"
                    ),
                    indent=2,
                    ensure_ascii=False
                ),
                encoding="utf-8"
            )

            results.append({

                "image":
                    str(image_path),

                "ground_truth":
                    ground_truth,

                "probability":
                    probability,

                "prediction":
                    prediction,

                "threshold":
                    TEMP_THRESHOLD,

                "latency_ms":
                    latency_ms,

                "correct":
                    correct,

                "error":
                    ""
            })

        except Exception as ex:

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"ERROR: {ex}"
            )

            results.append({

                "image":
                    str(image_path),

                "ground_truth":
                    ground_truth,

                "probability":
                    "",

                "prediction":
                    "error",

                "threshold":
                    TEMP_THRESHOLD,

                "latency_ms":
                    round(
                        elapsed * 1000,
                        2
                    ),

                "correct":
                    False,

                "error":
                    str(ex)
            })

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "image",
            "ground_truth",
            "probability",
            "prediction",
            "threshold",
            "latency_ms",
            "correct",
            "error"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print(
        "Benchmark terminado."
    )

    print(
        f"Resultados: "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()