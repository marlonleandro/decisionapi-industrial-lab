
from pathlib import Path
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results/predictions.csv"
OUTPUT = ROOT / "results/false_negatives"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# Falsos negativos del umbral original 0.50
fn = df[
    (df["ground_truth"] == "anomaly") &
    (df["probability"] < 0.50)
].copy()

fn = fn.sort_values("probability")

print("\n=== FALSE NEGATIVES ===\n")

for i, row in enumerate(fn.itertuples(), start=1):
    image = Path(row.image)
    dest = OUTPUT / f"FN_{i:02d}_{image.name}"

    if image.exists():
        shutil.copy2(image, dest)

    print(
        f"{image.name} | "
        f"P(defect)={row.probability:.4f} | "
        f"found={image.exists()}"
    )

print(f"\nTotal: {len(fn)}")
print(f"Exported to: {OUTPUT}")
