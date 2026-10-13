from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.decisions.client import inspect_pcb_visible_defect


def serialize_model(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Cannot serialize object of type {type(obj)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--ground-truth",
        choices=["normal", "anomaly"],
        default=None,
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    decision = inspect_pcb_visible_defect(args.image)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image": str(args.image),
        "ground_truth": args.ground_truth,
        "model": decision.model,
        "answers": [
            serialize_model(answer)
            for answer in decision.answers
        ],
        "usage": serialize_model(decision.usage),
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"{args.image.stem}_{stamp}.json"
    out_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved result: {out_file}")


if __name__ == "__main__":
    main()
