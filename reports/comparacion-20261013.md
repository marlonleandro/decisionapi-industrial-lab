# Comparación verificada Decisions API vs YOLO11n-cls

Fecha: 13 de octubre de 2026, zona America/Lima. Proyecto original: `D:\Proyectos\IA\Lab\DecisionAPI`. Evidencia numérica: `results/audit_20261013/`. Estado: evaluación terminada con umbrales fijados en validación; no se ejecutaron llamadas pagadas ni nuevo entrenamiento.

En el test de 60 imágenes de VisA pcb1, YOLO11n-cls obtuvo **93,33% accuracy**, **96,67% recall** y **F1=0,9355**; Decisions API V1 obtuvo **81,67%**, **76,67%** y **F1=0,8070**. YOLO produjo 1 FN y 3 FP; API, 7 FN y 4 FP. Es una comparación de un modelo entrenado específicamente y una API sin ajuste específico, sobre una muestra local pequeña y balanceada.

## Métricas verificadas por fase

| Método / conjunto | N | Umbral | Accuracy | Precision | Recall | F1 | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---|
| API_exploration_v1 | 20 | 0.50 | 80.00% | 100.00% | 60.00% | 0.7500 | 6/10/0/4 |
| API_exploration_v2 | 20 | 0.50 | 55.00% | 53.85% | 70.00% | 0.6087 | 7/4/6/3 |
| API_validation_v1 | 40 | 0.20 | 97.50% | 100.00% | 95.00% | 0.9744 | 19/20/0/1 |
| API_test_v1 | 60 | 0.20 | 81.67% | 85.19% | 76.67% | 0.8070 | 23/26/4/7 |
| YOLO_validation | 40 | 0.50 | 95.00% | 95.00% | 95.00% | 0.9500 | 19/19/1/1 |
| YOLO_test | 60 | 0.50 | 93.33% | 90.62% | 96.67% | 0.9355 | 29/27/3/1 |

Exploración V1/V2 se recalculó desde CSV históricos de 20 imágenes; su versión se atribuye por archivo/código y no se pudo recuperar el texto histórico enviado. V1 a 0,50 tenía accuracy 80%, recall 60%, sin FP; V2 tuvo accuracy 55%, recall 70%, seis FP. El umbral exploratorio no se eligió en test. `threshold_analysis*.csv` contiene barridos de exploración, no calibración independiente; V1 a 0,15 daba recall 80% y FPR10% en esas 20 imágenes y no sustituyó al umbral posterior de validación.

API validation se verificó contra 40 checkpoints y respuesta cruda: t=0,20, TP19/TN20/FP0/FN1. API test se recalculó sin nuevas llamadas y coincide con `final_test_metrics.json`. YOLO histórico validation a t=0,50 coincide exactamente con la inferencia nueva en GPU. Se evaluaron 19 umbrales de validación con la misma política API y se conservó 0,50; se congeló a las 11:20:52 (Lima), antes de terminar test a las 11:21:09.

## Intervalos e inferencia pareada

| Método test | IC95% accuracy (Wilson) | IC95% recall | FPR / IC95% |
|---|---|---|---|
| YOLO | 84,07–97,38% | 83,33–99,41% | 10,00% / 3,46–25,62% |
| API V1 | 70,08–89,44% | 59,07–88,21% | 13,33% / 5,31–29,68% |

En nueve discordancias de acierto, YOLO acierta ocho que API falla; API acierta una que YOLO falla. McNemar exacto bilateral: p=0,0390625. Este análisis exploratorio está condicionado por un único experimento, muestras pequeñas y elección del sistema; no justifica una afirmación universal de superioridad o aptitud de producción. Las 60 filas pareadas están en `paired_test.csv`. La mejora de accuracy observada es 11,67 puntos porcentuales y la de recall, 20 puntos.

La restricción FPR≤10% fue empírica en validación: API pasó de 0% validation a 13,33% test; YOLO de 5% a 10%. No garantiza una tasa poblacional ni una cota estadística del 10%.

## Máscaras y errores

Se procesaron las 100 máscaras anómalas de pcb1: 10 exploración, 40 train, 20 validation, 30 test. Todas tienen dimensiones coincidentes con la imagen y píxeles positivos. Para estadísticas se usó mask>0, conservando IDs/etiquetas. Las máscaras solo se analizaron tras evaluación y no se pasaron a ninguno de los modelos. El [mapa oficial](https://raw.githubusercontent.com/amazon-science/spot-diff/main/utils/id2class.py) distingue Bent/Melt/Scratch/Missing.

En test, el área anómala mediana ocupa 0.2166% de la imagen; rango 0.0929–7.8977%. Proyectando las máscaras con nearest a Resize(320)+CenterCrop, 0 de 30 anomalías pierden todos sus píxeles positivos. Es un diagnóstico aproximado: el procesamiento real de RGB usa bilinear y la máscara no mide evidencia contextual visible. Todos los FN test de ambos métodos conservan el 100% de los píxeles positivos proyectados; el análisis no respalda atribuir estos errores a la pérdida de la región anotada por el recorte. Cinco de los siete FN API incluyen missing y dos solo melt. El FN YOLO 056 es missing y API lo detectó: no hubo FN común entre métodos. Tres FP normales fueron comunes (0684, 0776, 0811); API tuvo además 0734. La figura confirma módulos HC-SR04 en los casos inspeccionados, coherentes con el contexto de V2; el empeoramiento de V2 no demuestra un desajuste de dominio.

| Método | Imagen | Error | Probabilidad | Etiquetas VisA | Área máscara | Retención aproximada crop |
|---|---|---|---:|---|---:|---:|
| API_test_v1 | 044.JPG | FN | 0.16000 | missing | 7.8977% | 100.0% |
| API_test_v1 | 046.JPG | FN | 0.19000 | missing | 6.3314% | 100.0% |
| API_test_v1 | 048.JPG | FN | 0.09000 | missing | 0.1086% | 100.0% |
| API_test_v1 | 051.JPG | FN | 0.13000 | melt,missing | 0.2886% | 100.0% |
| API_test_v1 | 054.JPG | FN | 0.12000 | missing | 3.9441% | 100.0% |
| API_test_v1 | 060.JPG | FN | 0.06000 | melt | 0.0929% | 100.0% |
| API_test_v1 | 062.JPG | FN | 0.16000 | melt | 0.1243% | 100.0% |
| API_test_v1 | 0684.JPG | FP | 0.40000 | normal | — | — |
| API_test_v1 | 0734.JPG | FP | 0.57000 | normal | — | — |
| API_test_v1 | 0776.JPG | FP | 0.20000 | normal | — | — |
| API_test_v1 | 0811.JPG | FP | 0.30000 | normal | — | — |
| YOLO_test | 056.JPG | FN | 0.38511 | missing | 1.9482% | 100.0% |
| YOLO_test | 0684.JPG | FP | 0.77167 | normal | — | — |
| YOLO_test | 0776.JPG | FP | 0.52022 | normal | — | — |
| YOLO_test | 0811.JPG | FP | 0.75619 | normal | — | — |


Los FN de exploración V1 a 0,50 son 047 (p=0,07), 058 (0,11), 097 (0,18), 012 (0,18), coherentes con `results/false_negatives/`. El inventario completo añade FP/FN de V2, API validation/test y YOLO val/test en `error_inventory.csv`; las métricas geométricas se encuentran en `mask_statistics.csv`. Ver también [figura de FN test](../results/audit_20261013/test_false_negative_masks.png); los contornos rojos son ground truth, no localización producida por los modelos.

## Entorno y latencia

RTX 5050 Laptop GPU, driver 610.74, memoria 8.151 MiB, Python 3.11.9, PyTorch 2.11.0+cu128/CUDA 12.8, Ultralytics 8.4.177, SDK OpenAI 3.29.0. CUDA se confirmó con inferencia. Entrenamiento histórico: 240 imágenes, 21 épocas de un máximo de 50, mejor época 9, imgsz 320 / batch 16 / patience 12 / seed 42, 68,54 segundos según experiment.json. No se reentrenó.

Latencia test observada: YOLO media 27.34ms incluyendo primera llamada, 19.61ms excluyéndola; API media histórica 1062.13ms. No se calculó un factor de aceleración: son mediciones diferentes, con red/servicio en API y arranque/local en YOLO. No hay coste monetario verificado.

## Integridad, límites y planes

Se verificaron semilla 42, ausencia de duplicados por ruta/contenido entre splits, exclusión de las 20 imágenes exploratorias y las 340 copias. La anomalía 000 de la prueba inicial aparece en validation; el test no contiene las muestras iniciales y no comparte datos con train. Esto limita la independencia de validación y queda explícito, sin redefinir splits retrospectivamente.

Hash best.pt: `31b2b128e312217b5c253f2910f57788536c4557f40431a5fcf76cda8703178d`. Hash manifest: `fce4468a8bafcf8915f8bf656cd2ba437d214af2d958697fa9e914859c6a19d2`. `yolo_threshold_frozen.json` conserva hashes, política, grilla, score de validación y tiempo de congelación; `original_hashes.json` permite comprobar que se conservaron los archivos anteriores. El README original sigue intacto; [README-comparacion.md](../README-comparacion.md) y [docs](../docs/README.md) son las guías actuales.

La conversación accesible contiene solo cinco intercambios recientes y omite textos de asistentes/turnos previos. Esta documentación reúne los temas solicitados con evidencia real, pero no se presenta como exportación completa del chat. No se inventaron métricas de fases ausentes. Ver [procedencia](../docs/procedencia.md).

Pendiente: categorías/lotes nuevos, test nuevo antes de ajustes, revisión manual de errores, evaluación de imagen completa/resolución/tiles, más semillas, prompt con contexto de dominio confirmado, calibración probabilística y costes FP/FN, latencia comparable y revisión humana. Ninguno de esos planes tiene resultados ejecutados atribuidos. Ver [limitaciones y próximos pasos](../docs/limitaciones-proximos-pasos.md) y [reproducción](../docs/reproduccion.md).
