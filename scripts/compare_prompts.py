
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

v1 = pd.read_csv(ROOT / "results/predictions.csv")
v2 = pd.read_csv(ROOT / "results/predictions_v2.csv")

# Vincular usando la ruta exacta de la imagen.
# No depender del orden de ejecución.
df = v1.merge(
    v2,
    on=["image", "ground_truth"],
    suffixes=("_v1", "_v2"),
    validate="one_to_one"
)

df["v1_correct"] = (
    df["prediction_v1"] == df["ground_truth"]
)

df["v2_correct"] = (
    df["prediction_v2"] == df["ground_truth"]
)

df["change"] = "UNCHANGED"

df.loc[
    ~df["v1_correct"] & df["v2_correct"],
    "change"
] = "IMPROVED"

df.loc[
    df["v1_correct"] & ~df["v2_correct"],
    "change"
] = "WORSENED"

output = ROOT / "results/prompt_comparison.csv"

df.to_csv(output, index=False)

print("\n=== V1 vs V2 ===\n")

print(
    df[[
        "image",
        "ground_truth",
        "probability_v1",
        "probability_v2",
        "prediction_v1",
        "prediction_v2",
        "change"
    ]].to_string(index=False)
)

print("\n=== SUMMARY ===")
print(df["change"].value_counts())

print(f"\nSaved: {output}")
