# OpenAI Decisions Industrial Vision Lab

Laboratorio reproducible que compara la **OpenAI Decisions API (GPT-6-luna)** frente a
**YOLO11n-cls** en una tarea de inspección visual de defectos sobre placas PCB del dataset
**VisA**, categoría `pcb1`. La tarea es una **clasificación binaria por imagen**
(`normal` vs `anomaly`, con `anomaly` como clase positiva): ambos métodos reciben la foto
completa y devuelven una probabilidad; ninguno localiza el defecto ni certifica el
funcionamiento eléctrico de la placa.

## Notebooks

La evidencia se explica en dos cuadernos complementarios. Ambos son **de lectura**: no
llaman a la API, no reentrenan YOLO y no modifican resultados; solo leen CSV/JSON/PNG ya
conservados en el repositorio, por lo que se ejecutan sin GPU, sin credenciales y sin coste.

| Notebook | Para qué sirve |
|---|---|
| [laboratorio-decisions-api-vs-yolo-friendly.ipynb](notebooks/laboratorio-decisions-api-vs-yolo-friendly.ipynb) | **Versión divulgativa**, pensada para profesionales de TI. Explica el problema, los gráficos y la decisión en lenguaje llano, y cierra con una ruta de reproducción. |
| [laboratorio-decisions-api-vs-yolo.ipynb](notebooks/laboratorio-decisions-api-vs-yolo.ipynb) | **Versión técnica**. Detalla arquitectura, pipeline de scripts, particiones, trazabilidad, congelamiento de umbrales, verificación de métricas y diagnóstico de errores. |

## Resultado en un minuto

Test pareado sobre **60 imágenes** (30 normales / 30 anómalas), con umbrales congelados
antes de evaluar:

| Método | Umbral | Accuracy | Precision | Recall | F1 | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---|
| YOLO11n-cls | 0,50 | **93,33%** | 90,62% | **96,67%** | 0,9355 | 29/27/3/1 |
| Decisions API V1 | 0,20 | 81,67% | 85,19% | 76,67% | 0,8070 | 23/26/4/7 |

YOLO detectó 29 de 30 anomalías y dejó escapar 1; la API detectó 23 y dejó escapar 7, sin
generar menos falsas alarmas (3 frente a 4). La diferencia es de **+11,67 puntos de accuracy**
y **+20 puntos de recall**. En las 9 discordancias de acierto, YOLO acertó 8 y la API 1
(McNemar exacto bilateral, p = 0,0391).

**Conclusión del laboratorio:** YOLO es el candidato para iniciar un piloto de inspección
visual en este escenario. La beta de Decisions API con GPT-6-luna no mostró madurez
suficiente para actuar como único filtro automático en la configuración evaluada. Es un
resultado local, sobre una muestra pequeña y balanceada, y no demuestra superioridad
universal ni aptitud para producción.

## Qué se hizo

1. **Dataset y particiones propias.** VisA `pcb1` (1.004 normales / 100 anómalas) se separó
   en exploración (20), entrenamiento (240), validación (40) y test (60), con semilla 42 y
   manifiesto reproducible en [`data/pcb1_splits.csv`](data/pcb1_splits.csv). No es el split
   oficial de VisA y no se verificó separación por lote o pieza física.
2. **Exploración de prompts V1 vs V2** sobre 20 imágenes. V2 describía un módulo HC-SR04 y
   detectó 7/10 anomalías frente a 6/10 de V1, pero generó 6 falsas alarmas frente a ninguna
   y bajó los aciertos del 80% al 55%. Se conservó **V1** para el benchmark.
3. **Benchmark de la API** con [`scripts/run_benchmark_v4.py`](scripts/run_benchmark_v4.py):
   un checkpoint JSON por imagen con respuesta cruda, probabilidad, latencia, estado y modelo.
   Pregunta tipo `predicate` `visible_defect` enviada a `gpt-6-luna`.
4. **Entrenamiento de YOLO11n-cls** con 240 imágenes (200 normales / 40 anómalas),
   `imgsz=320`, `batch=16`, máximo 50 épocas, paciencia 12 y semilla 42. La corrida conservada
   terminó en 21 épocas y eligió la época 9; pesos en
   [`results/yolo/pcb1_cls_v1/weights/best.pt`](results/yolo/pcb1_cls_v1/weights/best.pt).
5. **Congelamiento de umbrales en validación**, con grilla 0,05–0,95, FPR máximo del 10%,
   prioridad a recall y desempate por F1 y FPR: **0,20 para la API** y **0,50 para YOLO**,
   fijados antes de evaluar test.
6. **Comparación pareada en test** y auditoría de integridad con
   [`scripts/offline_audit_compare.py`](scripts/offline_audit_compare.py), que verifica datos
   y hashes, infiere validación, congela el umbral y evalúa test una sola vez.
7. **Diagnóstico de errores con máscaras** de VisA, usadas solo para revisar las regiones
   anotadas y nunca como entrada a los modelos. Ningún falso negativo fue común a ambos
   métodos; tres falsas alarmas sí lo fueron.
8. **Medición de latencia** con protocolos distintos: YOLO 27,34 ms de media incluyendo la
   primera llamada (19,61 ms excluyéndola) y API 1.062,13 ms de registro histórico. No
   permiten afirmar un factor de aceleración comparable ni un coste monetario verificado.

## Estructura del repositorio

```text
notebooks/   Notebooks divulgativo y técnico
docs/        Arquitectura, metodología, prompts, entorno, reproducción y limitaciones
scripts/     Pipeline: descarga, splits, benchmark API, entrenamiento y auditoría
src/         Cliente de Decisions API
data/        Manifiesto de particiones y muestras (el dataset no se versiona)
results/     Predicciones, checkpoints, umbrales congelados y auditoría
reports/     Reporte comparativo y revisiones
weights/     Pesos base
```

## Reproducción

### Ruta A — Leer o recalcular los notebooks

Requiere Python 3.11.9 con NumPy, pandas y Matplotlib ([`requirements.txt`](requirements.txt))
y un kernel de Jupyter o VS Code. No requiere GPU, pesos ni API key.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Abre cualquiera de los dos notebooks, selecciona el kernel y ejecuta las celdas en orden
desde la raíz del proyecto o desde `notebooks/`. La primera celda localiza la raíz subiendo
por el árbol de directorios. Las comprobaciones deben reproducir TP=29, TN=27, FP=3, FN=1
para YOLO y TP=23, TN=26, FP=4, FN=7 para la API.


### Ruta B — Repetir inferencia o entrenamiento

Requiere descargar VisA desde su [repositorio oficial](https://github.com/amazon-science/spot-diff),
GPU CUDA para entrenar y credenciales de OpenAI para la API (las llamadas pueden generar
gasto). Los comandos y el orden operativo están en la
[guía de reproducción](docs/reproduccion.md); la metodología y las particiones, en
[dataset y metodología](docs/dataset-metodologia.md) y el
[entorno documentado](docs/entorno-entrenamiento.md).

Para experimentos nuevos usa directorios de salida y versiones de prompt nuevas: la caché
histórica de la API identifica imágenes por ruta, no por contenido ni por hash del prompt.

## Limitaciones

- Una sola categoría, un solo entrenamiento y 60 fotos finales balanceadas; no se midió la
  prevalencia real de una planta ni el rendimiento por lote.
- Preparación distinta: YOLO aprendió de 240 imágenes etiquetadas y la API se evaluó sin
  ajuste específico. Se comparan esos sistemas preparados, no todas las variantes de ambas
  tecnologías.
- Validación usada para elegir pesos y umbrales, por lo que su resultado puede ser optimista;
  el test ya había sido inspeccionado y debe retirarse de futuros ajustes.
- Faltan mediciones comparables de tiempo extremo a extremo, costes verificables, estabilidad
  ante repeticiones y un flujo de supervisión industrial.

El detalle completo está en
[limitaciones y próximos pasos](docs/limitaciones-proximos-pasos.md).

## Evidencia y documentación

- [Reporte comparativo](reports/comparacion-20261013.md)
- [Arquitectura](docs/arquitectura.md)
- [Dataset y metodología](docs/dataset-metodologia.md)
- [Prompts V1/V2](docs/prompts.md)
- [Reproducción](docs/reproduccion.md)
- [Limitaciones y próximos pasos](docs/limitaciones-proximos-pasos.md)
- [Auditoría de integridad](results/audit_20261013/audit.json) y
  [verificación](results/audit_20261013/verification.json)

## Dataset y licencia

**Visual Anomaly (VisA)**, Amazon Science / AWS Open Data: 10.821 imágenes, 12 categorías;
`pcb1` contiene 1.004 normales y 100 anómalas. Licencia **CC BY 4.0**; requiere atribución.
El dataset no se versiona en Git.

## Créditos

Elaborado por Marlon Leandro.
