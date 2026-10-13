from __future__ import annotations

import csv
import random
from pathlib import Path


RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/benchmark_pcb1.csv")

NORMAL_COUNT = 10
ANOMALY_COUNT = 10
SEED = 42


def find_pcb1_root() -> Path:
    for path in RAW_DIR.rglob("pcb1"):
        normal = path / "Data" / "Images" / "Normal"
        anomaly = path / "Data" / "Images" / "Anomaly"

        if normal.exists() and anomaly.exists():
            return path

    raise FileNotFoundError("No se encontró pcb1")


def get_images(folder: Path):
    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in valid_extensions
        ]
    )


def sample_images(
    images,
    count,
    seed
):
    rng = random.Random(seed)

    if len(images) < count:
        raise ValueError(
            f"No existen suficientes imágenes: "
            f"{len(images)} disponibles"
        )

    return rng.sample(images, count)


def main():
    pcb1_root = find_pcb1_root()

    normal_dir = (
        pcb1_root
        / "Data"
        / "Images"
        / "Normal"
    )

    anomaly_dir = (
        pcb1_root
        / "Data"
        / "Images"
        / "Anomaly"
    )

    normal_images = get_images(normal_dir)
    anomaly_images = get_images(anomaly_dir)

    selected_normal = sample_images(
        normal_images,
        NORMAL_COUNT,
        SEED
    )

    selected_anomaly = sample_images(
        anomaly_images,
        ANOMALY_COUNT,
        SEED + 1
    )

    rows = []

    for image in selected_normal:
        rows.append({
            "image": str(image),
            "category": "pcb1",
            "ground_truth": "normal"
        })

    for image in selected_anomaly:
        rows.append({
            "image": str(image),
            "category": "pcb1",
            "ground_truth": "anomaly"
        })

    random.Random(SEED).shuffle(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "category",
                "ground_truth"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Benchmark creado: {OUTPUT_FILE}"
    )

    print(
        f"Total imágenes: {len(rows)}"
    )

    print(
        f"Normal: {NORMAL_COUNT}"
    )

    print(
        f"Anomaly: {ANOMALY_COUNT}"
    )


if __name__ == "__main__":
    main()