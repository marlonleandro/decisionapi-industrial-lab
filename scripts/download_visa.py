from __future__ import annotations

import tarfile
from pathlib import Path

import requests

URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
RAW_DIR = Path("data/raw")
ARCHIVE = RAW_DIR / "VisA_20220922.tar"


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE.exists():
        print(f"[skip] Archive already exists: {ARCHIVE}")
        return

    print(f"Downloading VisA from:\n{URL}")
    with requests.get(URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with ARCHIVE.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(
                        f"\r{downloaded / 1024**2:.1f} MiB / "
                        f"{total / 1024**2:.1f} MiB ({pct:.1f}%)",
                        end="",
                    )
        print()

    print(f"Saved: {ARCHIVE}")


def extract():
    marker = RAW_DIR / ".visa_extracted"

    if marker.exists():
        print("[skip] Dataset already extracted.")
        return

    print("Extracting archive...")
    with tarfile.open(ARCHIVE, "r") as tar:
        tar.extractall(RAW_DIR, filter="data")

    marker.write_text("ok", encoding="utf-8")
    print(f"Extracted under: {RAW_DIR}")


if __name__ == "__main__":
    download()
    extract()
