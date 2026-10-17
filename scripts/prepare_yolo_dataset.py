
from pathlib import Path
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/pcb1_splits.csv"
DEST = ROOT / "data/yolo_pcb1"

df = pd.read_csv(MANIFEST)

required = {"image", "ground_truth", "split"}
if not required.issubset(df.columns):
    raise ValueError(
        f"Columnas requeridas: {required}. "
        f"Encontradas: {set(df.columns)}"
    )

# Verificar que ningún archivo se repita entre splits
if df["image"].duplicated().any():
    raise ValueError("Hay imágenes duplicadas en el manifest")

for row in df.itertuples(index=False):
    if row.split not in ("train", "validation", "test"):
        raise ValueError(f"Split inválido: {row.split}")

    if row.ground_truth not in ("normal", "anomaly"):
        raise ValueError("Etiqueta inválida")

    src = Path(row.image)
    if not src.is_absolute():
        src = ROOT / src

    if not src.exists():
        raise FileNotFoundError(src)

    split = "val" if row.split == "validation" else row.split
    folder = DEST / split / row.ground_truth
    folder.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, folder / src.name)

print(f"Dataset preparado: {DEST}")
print(df.groupby(["split", "ground_truth"]).size())
