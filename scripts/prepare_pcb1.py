from __future__ import annotations

import json
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
SAMPLES_DIR = Path("data/samples")


def find_pcb1_root() -> Path:
    candidates = []
    for path in RAW_DIR.rglob("pcb1"):
        if (
            path.is_dir()
            and (path / "Data" / "Images" / "Normal").exists()
            and (path / "Data" / "Images" / "Anomaly").exists()
        ):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Could not find pcb1/Data/Images/{Normal,Anomaly}. "
            "Run scripts/download_visa.py first."
        )

    return candidates[0]


def first_image(folder: Path) -> Path:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"}
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix in extensions
    )
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")
    return images[0]


def copy_sample(src: Path, base_name: str) -> Path:
    dst = SAMPLES_DIR / f"{base_name}{src.suffix}"
    shutil.copy2(src, dst)
    return dst


if __name__ == "__main__":
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    pcb1 = find_pcb1_root()
    normal_src = first_image(pcb1 / "Data" / "Images" / "Normal")
    anomaly_src = first_image(pcb1 / "Data" / "Images" / "Anomaly")

    normal_dst = copy_sample(normal_src, "pcb1_normal")
    anomaly_dst = copy_sample(anomaly_src, "pcb1_anomaly")

    metadata = {
        "category": "pcb1",
        "samples": [
            {
                "ground_truth": "normal",
                "source": str(normal_src),
                "sample": str(normal_dst),
            },
            {
                "ground_truth": "anomaly",
                "source": str(anomaly_src),
                "sample": str(anomaly_dst),
            },
        ],
    }

    (SAMPLES_DIR / "samples.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Prepared:")
    print(f"  NORMAL : {normal_dst}")
    print(f"  ANOMALY: {anomaly_dst}")
