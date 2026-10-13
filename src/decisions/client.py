from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(
            f"Unsupported or unknown image MIME type: {mime_type}. "
            "Use JPEG, PNG or WEBP."
        )

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def inspect_pcb_visible_defect(image_path: str | Path):
    client = OpenAI()
    image_url = image_to_data_url(image_path)

    return client.decisions.create(
        model="gpt-6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Inspect this printed circuit board (PCB) as part of "
                            "an industrial visual quality-control process."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                    },
                ],
            }
        ],
        questions=[
            {
                "type": "predicate",
                "name": "visible_defect",
                "instructions": (
                    "Does this PCB show a visible manufacturing anomaly or defect? "
                    "Look for observable abnormalities such as missing or misplaced "
                    "components, surface damage, scratches, cracks, unusual spots, "
                    "or other visible structural inconsistencies. "
                    "Ignore ordinary lighting variation, shadows, reflections, "
                    "perspective differences, and normal component variation."
                ),
            }
        ],
    )
