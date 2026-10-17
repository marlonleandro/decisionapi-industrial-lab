# Revisión del repositorio antes de publicar en GitHub

Fecha: 15 de octubre de 2026, America/Lima. Repositorio revisado: `DecisionAPI`. Estado de referencia: HEAD `96d3b44`, con cambios locales previos y archivos nuevos sin seguimiento.

## Decisión recomendada

Publicar un repositorio con código, documentación, los dos notebooks, manifiestos, resultados auditados, respuestas históricas y el peso entrenado `best.pt`. Mantener fuera de Git el entorno Python, las credenciales, VisA original, sus copias para YOLO, los pesos descargables, el checkpoint final y los mosaicos de entrenamiento.

Las exclusiones nuevas evitan incorporar aproximadamente **102,46 MiB** de copias y artefactos prescindibles para la publicación, además de la nota de entrega local. El entorno y VisA, que ya estaban ignorados, ocupan aproximadamente **8.809,74 MiB**. Excluirlos no libera espacio en disco: evita añadirlos a Git.

**No se eliminó, movió ni quitó del índice ningún archivo. No se hizo commit ni push.** Solo se amplió `.gitignore` y se añadieron este informe y su inventario. Los cambios previos de `requirements.txt` y los resultados del laboratorio se conservaron.

## Qué se revisó

- Inventario por tamaño de todos los archivos del árbol de trabajo, agrupando las dependencias de `.venv` y los datos originales para facilitar la lectura; se excluyó el contenido interno de `.git` del cálculo de espacio.
- Archivos seguidos por Git, archivos nuevos y reglas de exclusión realmente aplicadas por Git.
- Código Python, rutas de entrada/salida, referencias en Markdown y notebooks, manifiestos y requisitos de reproducción.
- Comparación SHA-256 de las **340 imágenes copiadas para YOLO** frente a sus originales: todas coinciden.
- Hashes de `best.pt` y `last.pt`: **son diferentes**, aunque tengan el mismo tamaño.
- Esquema de 100 checkpoints de API y 20 respuestas V2. Las respuestas de los checkpoints conservan `answers`, `model` y `usage`; no contienen el cuerpo de entrada de la imagen en ese objeto.
- Revisión por patrones de credenciales de 199 archivos de texto publicables antes de crear estos informes y 198 blobs de texto del historial Git local disponible. No se encontraron coincidencias. No se abrió `.env` ni se incluyeron valores de credenciales en las salidas.
- Nombres de archivos sensibles en el historial local: se encontró únicamente `.env.example`, el ejemplo esperado. Esta comprobación no consulta el estado remoto ni sustituye una auditoría completa de secretos con patrones adicionales.

El [inventario CSV](revision-repositorio-20261015.csv) contiene 585 filas con ruta, tamaño, estado Git, decisión, motivo e indicación de pertenencia al inventario histórico de hashes. Enumera individualmente 566 archivos del proyecto y agrupa `.venv` y las carpetas de datos originales; también registra `scr/`. Excluye sus dos propios archivos de informe para evitar una referencia circular. Los tamaños son bytes lógicos, no espacio físico asignado en disco.

## 1. Candidatos para eliminar o mantener solo en local

| Ruta | Tamaño observado | Recomendación | Condición o consecuencia |
|---|---:|---|---|
| `scr/` | Vacía | Eliminar | No contiene código. No confundir con `src/`, que sí es esencial. Git no versiona directorios vacíos. |
| `LEEME-entrega.md` | Menos de 1 KiB | Eliminar o dejar solo en local | Es una nota de copia/entrega con una ruta de esta máquina. No interviene en el laboratorio. Ya quedó ignorada. |
| `__pycache__/`, `*.pyc`, `*.cache` | Pequeño y variable | Eliminar cuando se quiera limpiar | Cachés regenerables. No confundir `*.cache` con los checkpoints JSON de respuestas de la API. |
| `data/yolo_pcb1/` | **86,58 MiB**, 340 imágenes y 2 cachés | Mantener solo en local; eliminación regenerable | Las 340 copias coinciden con sus originales. Se reconstruye con `prepare_yolo_dataset.py`. Borrarla impide entrenar y ejecutar la auditoría completa hasta regenerarla. El notebook friendly puede seguir recalculando sus gráficos. |
| `yolo11n-cls.pt` | **5,52 MiB** | Mantener solo en local o descargar de nuevo | Es el modelo base preentrenado; no es el peso ajustado del experimento. Conservar una copia/hash si se exige reproducir exactamente el entrenamiento de referencia. |
| `weights/yolo26n.pt` | **5,29 MiB** | Mantener solo en local o archivar | Artefacto auxiliar de comprobación AMP, según la documentación. No se utiliza como clasificador final del laboratorio. |
| `results/yolo/pcb1_cls_v1/weights/last.pt` | **3,04 MiB** | Archivar fuera de Git si interesa reanudar; prescindible para publicar | Es el último estado de entrenamiento. No es idéntico a `best.pt` y no debe sustituirlo. Eliminarlo pierde ese checkpoint concreto. |
| Siete `train_batch*.jpg` y `val_batch*.jpg` en la corrida YOLO | **2,03 MiB** en total | Archivar u omitir de Git | Mosaicos de diagnóstico; no se leen para calcular las métricas ni para inferir. No son las matrices de confusión ni la figura de errores del reporte. |
| `data/raw/VisA_20220922.tar` | **1.840,44 MiB** | Eliminación opcional y condicionada para ahorrar disco | El dataset está extraído y hay un marcador de extracción, pero no se contrastó todo el TAR contra todos sus archivos extraídos. Antes de borrarlo, verificar extracción completa o disponer de respaldo/descarga. Ya estaba fuera de Git. |

**Importante para la conservación histórica:** los diez artefactos de pesos descargables, `last.pt` y mosaicos figuran en `results/audit_20261013/original_hashes.json`. Omitirlos de Git es compatible con publicar la evidencia principal, pero borrarlos sin respaldo impide verificar de nuevo la totalidad de aquella instantánea de 179 archivos. No editar los hashes para ocultar su ausencia. La reproducción de métricas y la verificación íntegra del archivo histórico son objetivos distintos.

No se identificó ningún módulo de `src/` que pueda eliminarse sin afectar el cliente o su estructura de importación.

## 2. Archivos grandes que no son basura

| Ruta | Tamaño observado | Motivo para conservar |
|---|---:|---|
| `.venv/` | **5.137,72 MiB**, 47.585 archivos | Entorno funcional actual. No se publica, pero eliminarlo obliga a reinstalar y verificar dependencias/CUDA. No es una limpieza necesaria para GitHub. |
| `data/raw/`, con el TAR | **3.672,03 MiB**, 12.040 archivos | Fuente de imágenes, máscaras, etiquetas y datos para otras categorías. Conservar localmente para inferencia y diagnóstico. Ya está ignorado, salvo `.gitkeep`. |
| `data/raw/pcb1/` | **283,23 MiB**, 1.205 archivos | Datos originales de la categoría evaluada, necesarios para reproducir inferencias, hashes y diagnóstico de máscaras. |
| `results/yolo/pcb1_cls_v1/weights/best.pt` | **3,04 MiB** | Peso exacto ganador. Se lee en `evaluate_yolo.py`, la auditoría y la documentación de reproducción. Debe seguir disponible en Git o en un artefacto externo publicado con hash y descarga documentada. Actualmente se conserva publicable en Git. |
| Notebook técnico original | **4,78 MiB**, 47 celdas, 10 imágenes embebidas | Anexo técnico enlazado desde la versión friendly. Parte de su tamaño son gráficos y salidas guardadas, útiles para leerlo sin ejecutar. |
| Notebook friendly | **0,48 MiB**, 28 celdas, 7 imágenes embebidas | Entregable principal para profesionales de TI. Conservar salidas: evita exigir ejecución solo para leer los gráficos. |
| `results/audit_20261013/test_false_negative_masks.png` | **2,97 MiB** | Diagnóstico visual enlazado desde el reporte y mostrado en el notebook técnico. No confundir con mosaicos temporales de entrenamiento. |

El resto de las categorías de VisA no participa en esta comparación de `pcb1`, pero es una reserva para el siguiente laboratorio. No se recomienda borrarlo automáticamente: ya está fuera de Git y la documentación plantea ampliar categorías.

El notebook técnico podría publicarse con salidas reducidas para disminuir el tamaño del repositorio, conservando aparte una copia ejecutada. Es una decisión editorial opcional, no un requisito ni una razón para eliminarlo. No se cambiaron sus salidas en esta revisión.

## 3. Evidencia pequeña que debemos subir

Conservar estos archivos aunque sean generados. Su tamaño es pequeño y aportan más valor que el espacio que ahorraríamos al eliminarlos.

| Grupo | Archivos o rutas | Razón |
|---|---|---|
| Manifiestos | `data/benchmark_pcb1.csv`, `data/pcb1_splits.csv` | Identifican exactamente exploración, entrenamiento, validación y test. |
| Resultados finales | `results/final_test_metrics.json`, `results/selected_threshold.json` | Métricas API y umbral fijado antes de test. |
| Comparación auditada | Todo `results/audit_20261013/` | Predicciones pareadas, métricas, intervalos, política, hashes, máscaras y entorno. Evitar una regla que ignore `results/` completo. |
| Respuestas API | `results/runs/v1_validation/`, `results/runs/v1_test/` | 100 checkpoints y 2 CSV, unos **0,089 MiB**. Permiten comprobar respuestas reales y reutilizar llamadas exitosas. Borrarlos puede provocar gasto al repetir el benchmark. |
| Exploración histórica | `results/predictions*.csv`, `prompt_comparison.csv`, `threshold_analysis*.csv`, `results/raw_v2/` | Sustentan la comparación de prompts y son entradas del notebook técnico. Las 20 respuestas V2 ocupan unos **8 KiB**. |
| Corrida YOLO resumida | `args.yaml`, `experiment.json`, `results.csv`, `metrics_val.json`, `predictions_val.csv`, `results.png`, `confusion_matrix*.png` | Configuración y evolución del entrenamiento y validación. Conservar junto con `best.pt`. |
| Ejemplos | `data/samples/`, `results/false_negatives/` | Ejemplos de inspección/errores utilizados por documentación o auditoría, ya seguidos por Git. Ocupan unos **1,53 MiB** en conjunto. |
| Código | `src/`, `scripts/` | Cliente, preparación, entrenamiento, evaluación, auditoría y fuentes del notebook. |
| Dependencias | `requirements.txt`, `requirements-yolo-observed.txt` | No son duplicados exactos: el segundo es una referencia del entorno observado y está enlazado por la documentación. |
| Plantilla de entorno | `.env.example` | Facilita configurar la variable requerida sin publicar su valor real. |

No deduplicar respuestas JSON solo porque tengan bytes idénticos: dos imágenes pueden haber recibido legítimamente la misma respuesta. Su asociación con el caso evaluado tiene valor de evidencia.

## 4. Archivos históricos que se pueden reorganizar después

- **`scripts/document_verified_state.py`:** generador de una sola continuación, con textos extensos y la ruta antigua `docs/continuacion-20261013`. No es una utilidad general idempotente. Se podría mover a una carpeta de herramientas históricas, conservando enlaces y procedencia. Hoy está referenciado por el notebook técnico y `docs/reproduccion.md`; no conviene borrarlo directamente.
- **Scripts de exploración:** `inspect_one.py`, `run_benchmark.py`, `calculate_metrics.py`, `analyze_thresholds.py`, `compare_prompts.py` y `export_false_negatives.py` no forman el camino principal del resultado final, pero explican fases históricas y varios aparecen en el notebook. Mantenerlos o archivarlos como grupo documentado. No se ejecutaron: algunos hacen llamadas o reemplazan resultados.
- **`docs/sprint-1.md`, `docs/sprint-4.md`:** documentación histórica de etapas distintas, no copias del reporte final. Se pueden agrupar bajo documentación histórica con enlaces actualizados.
- **`docs/procedencia.md`:** conserva límites de recuperación y origen de la evidencia. Contiene un identificador de conversación y una ruta local. Para una edición pública, conviene preparar una versión de procedencia centrada en datos/experimento, guardando los detalles de conversación fuera del repo si no aportan al lector. No eliminar sin reemplazar su función ni resolver referencias.
- **`scripts/build_friendly_notebook.py`:** no es imprescindible para leer el notebook generado, pero es su fuente de mantenimiento y cuesta unos 36 KiB. Recomiendo conservarlo.

## 5. Hallazgos que deben resolverse antes de presentar el repo como listo

1. **Once enlaces Markdown rotos — corregidos el 15 de octubre de 2026.** La revisión detectó siete enlaces con la estructura antigua en `README-comparacion.md`, uno en `docs/README.md` y tres en el reporte comparativo. Se actualizaron para apuntar a los documentos existentes directamente en `docs/`, conservando las referencias útiles. Se verificaron los destinos locales de los enlaces en los documentos Markdown y en ambos notebooks.
2. **README principal desactualizado.** Explica solo Sprint 1. Conviene convertirlo en la entrada al notebook friendly, al reporte y a la guía de reproducción; conservar el texto de Sprint 1 como historial si interesa.
3. **Dos afirmaciones de documentación desactualizadas.** `docs/arquitectura.md` agrupa los notebooks con una carpeta vacía como si no aportaran resultados ejecutados; los notebooks hoy contienen salidas. `docs/entorno-entrenamiento.md` dice que `requirements.txt` no contiene Torch/Ultralytics, pero el archivo local actual sí los incluye. Son problemas de actualización, no motivos para borrar archivos.
4. **Instalación específica de CUDA.** El `requirements.txt` actual fija ruedas `+cu128`; conviene documentar su fuente de instalación y separar, si se desea, dependencias de lectura del notebook de las necesarias para GPU/API. No eliminar el entorno funcional hasta demostrar que una instalación limpia funciona.
5. **Reproducción desde un clon limpio.** Excluir `data/yolo_pcb1/` exige prepararlo desde las imágenes originales antes de ejecutar la auditoría, que comprueba las 340 copias. La guía debe indicarlo expresamente. Mantener `best.pt` evita añadir un obstáculo innecesario.
6. **Historial y privacidad.** Algunas salidas y configuraciones registran rutas absolutas del equipo. No se encontraron patrones de claves en los textos revisados; eso no implica ausencia de toda información privada. Conservar la evidencia histórica y preparar una representación pública si se quiere ocultar rutas o identificadores.
7. **Licencias y atribución.** Los ejemplos y figuras reutilizan VisA. La documentación ya atribuye el dataset y CC BY 4.0; conviene hacer visible esa atribución desde la portada. No se encontró una licencia propia del código en la raíz; definirla es una decisión del propietario y no se inventó una en esta revisión.

Los demás puntos quedan identificados y pendientes. La corrección posterior de enlaces no alteró las métricas ni los archivos de evidencia históricos.

## 6. Cambios aplicados a `.gitignore`

Se conservaron las exclusiones de `.venv`, cachés Python, `.env` y datos originales. Se añadieron:

- Variantes de entorno (`.env.*`), manteniendo publicable `.env.example`.
- `data/yolo_pcb1/` y los cachés de Ultralytics.
- Pesos `.pt` de la raíz y la carpeta auxiliar `weights/`.
- Solo `last.pt` y mosaicos `train_batch*.jpg`/`val_batch*.jpg` dentro de las corridas YOLO.
- Checkpoints temporales del editor de notebooks, algunos archivos de sistema/editor y la nota de entrega local.

**No se excluyeron** `best.pt`, los notebooks, los manifiestos, los checkpoints JSON de la API ni los informes de auditoría. Las reglas se comprobaron con `git check-ignore`; ningún archivo ya seguido por Git coincide con las nuevas exclusiones. Agregar una regla ignore no retira archivos que ya estuvieran versionados, pero los artefactos pesados recién excluidos aún no estaban en el índice.

Antes de añadir los informes de esta revisión, el conjunto publicable quedó en **209 archivos y aproximadamente 13,54 MiB**, contando archivos seguidos y nuevos no ignorados. Los propios informes añaden un pequeño incremento. Esto mide el contenido actual del árbol de trabajo, no el tamaño comprimido de Git ni el historial remoto.

## 7. Orden propuesto para una limpieza posterior

1. Eliminar la carpeta vacía `scr/` y la nota `LEEME-entrega.md` si ya no se necesitan.
2. Si se quiere ahorrar disco, eliminar cachés y `data/yolo_pcb1/` tras aceptar que habrá que regenerarlo para entrenar o auditar imágenes. Las copias ya fueron verificadas.
3. Respaldar los diez artefactos inventariados que se omiten de Git si se desea conservar la instantánea histórica completa. Después se pueden retirar de la carpeta de trabajo.
4. Decidir sobre el TAR de VisA únicamente después de comprobar extracción o disponer de respaldo. Mantener los originales de `pcb1` para reproducir inferencias.
5. Actualizar portada y notas de reproducción antes de publicar; los once enlaces rotos ya fueron corregidos. Reorganizar archivos históricos en un cambio separado y revisar las referencias.
6. Revisar el conjunto exacto a añadir a Git. La evidencia debe permitir recalcular las cifras del notebook friendly desde sus CSV/JSON sin GPU ni llamadas nuevas.

La mayor mejora para GitHub ya está preparada con las exclusiones. La mayor recuperación de disco sería el TAR o el entorno Python, pero ambos requieren una decisión operativa: no deben eliminarse como si fueran basura indistintamente.
