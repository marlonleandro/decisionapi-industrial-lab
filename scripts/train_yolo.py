
from pathlib import Path
import json
import time
import torch

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "yolo_pcb1"
OUTPUT = ROOT / "results" / "yolo"

MODEL_NAME = "yolo11n-cls.pt"


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no disponible. "
            "Verifica PyTorch y el driver NVIDIA."
        )

    print("=" * 50)
    print("INDUSTRIAL PCB INSPECTION - YOLO")
    print("=" * 50)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Dataset: {DATASET}")

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {DATASET}"
        )

    model = YOLO(MODEL_NAME)

    start = time.perf_counter()

    model.train(
        data=str(DATASET),
        epochs=50,
        imgsz=320,
        batch=16,
        device=0,
        patience=12,
        seed=42,
        deterministic=True,
        workers=2,
        project=str(OUTPUT),
        name="pcb1_cls_v1",
        exist_ok=False,
        plots=True,
        save=True,
        verbose=True,
    )

    elapsed = time.perf_counter() - start

    run_dir = Path(model.trainer.save_dir)
    best_model = run_dir / "weights" / "best.pt"

    metadata = {
        "model": MODEL_NAME,
        "dataset": "VisA pcb1",
        "epochs_max": 50,
        "imgsz": 320,
        "batch": 16,
        "seed": 42,
        "gpu": torch.cuda.get_device_name(0),
        "training_seconds": round(elapsed, 2),
        "best_model": str(best_model),
        "run_dir": str(run_dir),
    }

    output_file = run_dir / "experiment.json"

    output_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\nENTRENAMIENTO FINALIZADO")
    print(f"Tiempo: {elapsed:.2f} segundos")
    print(f"Mejor modelo: {best_model}")
    print(f"Resultados: {run_dir}")


if __name__ == "__main__":
    main()
