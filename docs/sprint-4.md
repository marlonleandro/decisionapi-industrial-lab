# Sprint 4 — evaluación independiente VisA pcb1

## Requisitos previos
- Haber terminado Sprint 3 con `src/decisions/client.py` que soporte `prompt_version="v1"`.
- Estar en la raíz del proyecto original.
- Tener `data/benchmark_pcb1.csv` con las 20 imágenes originales y dataset VisA descargado.
- Dependencias originales, y `scikit-learn` para calibración.

## Instalación de estos archivos
Copiar `scripts/*.py` del paquete a `scripts/` del proyecto original sin borrar los archivos existentes.

## Ejecución

```bash
python scripts/prepare_dataset_splits.py
python scripts/run_benchmark_v4.py --split validation --prompt-version v1 --max-images 2
python scripts/run_benchmark_v4.py --split validation --prompt-version v1
python scripts/validate_and_freeze.py
python scripts/run_benchmark_v4.py --split test --prompt-version v1 --allow-test
```

**No ejecutes el test hasta revisar el rendimiento de validación y aceptar la política de umbral.**

## Advertencias metodológicas
- Las particiones del laboratorio son locales y determinísticas, no la partición oficial propuesta para entrenamiento. Para comparar contra YOLO, NO debe usarse en el entrenamiento ninguna imagen incluida en `validation`, `test` o en exploración. Prepara el conjunto YOLO con `train` solamente y/o redefine particiones antes de cualquier nueva evaluación.
- `pcb1` ofrece solo 100 anomalías; 10 ya se usaron en exploración, 20 en validación y 30 en test, por lo que quedan 40 para entrenar. Esto limita fuertemente el entrenamiento supervisado. Más adelante convendrá extender el benchmark a otras categorías.
- La muestra del test es pequeña: presentar IC y no afirmar aptitud para producción.
- Usar exactamente el mismo criterio del ground truth en todos los métodos.
- Para el reporte, no cambiar el prompt después de ver resultados del test.
- El script de corrida usa la política de reintentos predeterminada del SDK OpenAI; los fallos se registran y se pueden reintentar ejecutando nuevamente.
- `results/selected_threshold.json` queda congelado antes del test.
- Al ejecutar `--max-images` los checkpoints cuentan para el experimento completo; no se repiten llamadas.
