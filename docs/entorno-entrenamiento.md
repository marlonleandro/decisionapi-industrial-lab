# Entorno RTX 5050 y entrenamiento verificado

La conversación reportó primero “NO GPU” desde Python mientras NVIDIA detectaba la tarjeta. El estado actual corrige esa discrepancia: el entorno `.venv` reconoce CUDA y la RTX 5050 Laptop GPU. No se reconstruyó el proceso exacto de instalación anterior ni se modificaron paquetes/driver hoy.

| Componente | Evidencia actual |
|---|---|
| Windows / Python | Python 3.11.9, 64 bits |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| Memoria reportada | 8.151 MiB |
| Driver | 610.74 |
| CUDA UMD del reporte inicial NVIDIA | 13.3, dato aportado por el usuario |
| PyTorch | 2.11.0+cu128 |
| Runtime CUDA de PyTorch | 12.8 |
| CUDA disponible | True; inferencias val y test ejecutadas |
| Ultralytics | 8.4.177 |
| OpenAI SDK | 3.29.0 |

El dato de CUDA del driver y `torch.version.cuda` describen capas distintas y no tienen que ser idénticos. La combinación funcional se verificó ejecutando inferencia, además de consultar disponibilidad. `environment.json`, `environment_test.json` y `packages_snapshot.json` preservan versiones y transformaciones reales.

## Entrenamiento conservado

Modelo preentrenado `yolo11n-cls.pt`, clasificación en 2 clases. `train_yolo.py` y `args.yaml`: máximo 50 épocas, imgsz=320, batch=16, device=0, patience=12, seed=42, deterministic=True, workers=2, optimizer=auto, AMP. El registro adjunto a la conversación identifica AdamW lr=0,001667, early stopping tras 12 épocas sin mejora, 21 épocas totales y mejor época 9. `results.csv` confirma 21 filas y accuracy_top1=0,95 en la época 9. El tiempo de `experiment.json` es 68,54 segundos (incluye el trabajo medido por el script; no es solo tiempo de GPU). Top5=1 es poco informativo con dos clases.

El registro indica 1.533.666 parámetros antes de fusión y 1.528.586 tras fusión. `weights/yolo26n.pt` se descargó para la comprobación AMP y no sustituye al YOLO11n-cls entrenado, según ese registro. Los pesos decisivos son `results/yolo/pcb1_cls_v1/weights/best.pt`.

SHA-256 best.pt: `31b2b128e312217b5c253f2910f57788536c4557f40431a5fcf76cda8703178d`.

La inferencia de validación de hoy reproduce exactamente las 40 probabilidades históricas (máxima diferencia absoluta 0). Clase 0=anomaly, clase 1=normal; el evaluador busca el nombre en lugar de asumir el índice. Las transformaciones reales son Resize(320), CenterCrop(320×320), ToTensor y Normalize(mean=0,std=1), registradas en entorno. La [documentación de clasificación Ultralytics](https://docs.ultralytics.com/tasks/classify) describe los recortes de entrenamiento/inferencia; pueden ocultar regiones importantes. No se cambió esa transformación después de ver test.

`requirements.txt` original no incluye torch/torchvision/ultralytics. Se añadió `requirements-yolo-observed.txt` como referencia de versiones observadas y se conservó el original. No equivale a un lock de todas las dependencias; la instantánea completa figura en packages_snapshot.json. Reentrenar no garantiza pesos idénticos incluso con deterministic=True en otra plataforma/versión.
