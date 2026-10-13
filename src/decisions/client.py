from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


PROMPT_V1 = (
            "Does this PCB show a visible manufacturing anomaly or defect? "
            "Look for observable abnormalities such as missing or misplaced "
            "components, surface damage, scratches, cracks, unusual spots, "
            "or other visible structural inconsistencies. "
            "Ignore ordinary lighting variation, shadows, reflections, "
            "perspective differences, and normal component variation."
        )

PROMPT_V2 = """
        Determine whether this HC-SR04 ultrasonic sensor module
        contains any visible manufacturing anomaly or defect.

        Expected configuration:
        - Four connector pins labeled Vcc, Trig, Echo and Gnd.
        - All four pins must be physically present.
        - Pins should be straight and properly aligned.
        - No visibly broken, bent or missing pins.
        - No abnormal solder joints or damaged PCB traces.
        - No missing or displaced components.
        - No unusual physical damage or contamination.

        Pay special attention to:
        1. Missing connector pins.
        2. Bent or misaligned pins.
        3. Physical damage around solder joints.
        4. Scratches or damage on the PCB.
        5. Deformation of ultrasonic transducers.

        Ignore normal lighting differences, shadows,
        reflections and harmless cosmetic variation.

        A visually evident deviation from the expected
        configuration counts as a defect.

        Do not infer hidden electrical defects from
        visual appearance alone.
        """


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


def inspect_pcb_visible_defect(
        image_path: str | Path,
        prompt_version: str = "v1"
    ):
    client = OpenAI()
    image_url = image_to_data_url(image_path)

    prompts = {
        "v1": PROMPT_V1,
        "v2": PROMPT_V2
    }

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
                "instructions": prompts[prompt_version],
            }
        ],
    )
