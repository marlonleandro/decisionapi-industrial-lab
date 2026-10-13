# OpenAI Decisions Industrial Vision Lab

Laboratorio reproducible para evaluar **OpenAI Decisions API** sobre imágenes industriales
del dataset **VisA (Visual Anomaly)**, comenzando con la categoría `pcb1`.

## Sprint 1

Objetivo:

1. Preparar el entorno Python.
2. Descargar el dataset VisA desde AWS Open Data.
3. Extraer una PCB normal y una PCB anómala de `pcb1`.
4. Enviar una imagen real a OpenAI Decisions API.
5. Obtener la probabilidad de defecto visible.
6. Guardar la respuesta cruda en `results/raw/`.

## Requisitos

- Python 3.11+ recomendado
- OpenAI Python SDK 3.26.0+
- Una API key de OpenAI con acceso a Decisions API

## Instalación

### Windows / PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edita `.env`:

```env
OPENAI_API_KEY=sk-...
```

Nunca subas `.env` a GitHub.

## Paso 1 — Descargar VisA

```powershell
python scripts/download_visa.py
```

El archivo oficial se descarga desde:

`https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`

El dataset no se versiona en Git.

## Paso 2 — Preparar dos imágenes de pcb1

```powershell
python scripts/prepare_pcb1.py
```

El script busca automáticamente una imagen `Normal` y una `Anomaly` y crea:

```text
data/samples/
├── pcb1_normal.<ext>
├── pcb1_anomaly.<ext>
└── samples.json
```

## Paso 3 — Ejecutar la primera decisión

Primero con una PCB normal:

```powershell
python scripts/inspect_one.py data/samples/pcb1_normal.JPG --ground-truth normal
```

Y después con una PCB anómala:

```powershell
python scripts/inspect_one.py data/samples/pcb1_anomaly.JPG --ground-truth anomaly
```

> Si la extensión real es `.jpg` o `.png`, usa el nombre creado por `prepare_pcb1.py`.

La pregunta utilizada en Sprint 1 es intencionalmente binaria:

> ¿Esta PCB presenta un defecto de fabricación visible?

La respuesta incluye una probabilidad entre 0 y 1.

## Salida

Ejemplo conceptual:

```json
{
  "image": "data/samples/pcb1_anomaly.JPG",
  "ground_truth": "anomaly",
  "model": "gpt-6-luna",
  "answer": {
    "type": "predicate",
    "name": "visible_defect",
    "probability": 0.94
  }
}
```

No se debe interpretar `0.50`, `0.80` u otro valor como threshold definitivo.
Los umbrales se calibrarán posteriormente usando el ground truth de VisA.

## Fuente del dataset

Visual Anomaly (VisA), Amazon Science / AWS Open Data.

- 10,821 imágenes.
- 9,621 normales.
- 1,200 anómalas.
- 12 categorías.
- `pcb1` contiene 1,004 imágenes normales y 100 anómalas.
- Licencia del dataset: CC BY 4.0.

## Próximo sprint

Automatizar varias imágenes, registrar latencia/resultados y comenzar el benchmark
contra el ground truth.
