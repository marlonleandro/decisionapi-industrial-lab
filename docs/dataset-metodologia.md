# Dataset, particiones y metodología

[VisA, Amazon Science](https://github.com/amazon-science/spot-diff) reúne 10.821 imágenes de 12 categorías. La categoría pcb1 tiene 1.004 normales y 100 anómalas, cantidades comprobadas localmente. El dataset se distribuye bajo CC BY 4.0; debe conservarse su atribución. Fuente descargable: [archivo VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar). La descarga/extracción ya existe y no se repitió.

| Conjunto | Normal | Anomaly | Total | Uso |
|---|---:|---:|---:|---|
| Exploración | 10 | 10 | 20 | Comparar V1/V2 y estudiar errores |
| Train | 200 | 40 | 240 | Ajustar YOLO |
| Validation / val | 20 | 20 | 40 | Selección de checkpoint y umbral |
| Test | 30 | 30 | 60 | Comparación final sin ajuste posterior |
| No asignado | 744 | 0 | 744 | Reserva normal; no un nuevo test balanceado |

`benchmark_pcb1.csv` usa listas ordenadas, muestreo de normales con semilla 42 y anomalías con 43, y mezcla con 42. `pcb1_splits.csv` usa un único generador `random.Random(42)`: normales primero, luego anomalías; excluye exploración; ordena, mezcla y asigna validation, test, train en ese orden; finalmente mezcla filas. La auditoría reprodujo las 340 filas exactamente, comprobó exclusión de las 20 de exploración, ausencia de solapamientos por ruta y contenido SHA-256 y coincidencia de las 340 copias YOLO con originales. No se verificaron duplicados visualmente similares ni grupos de lote/captura.

La partición es local: no debe presentarse como el split oficial de VisA. Train está desbalanceado 5:1 y val/test balanceados 1:1. Precision y accuracy dependen de esa prevalencia. Quedan solo 40 anomalías de entrenamiento.

La prueba inicial preparó `Normal/0000.JPG` (no asignada) y `Anomaly/000.JPG` (incluida en validation). Sus llamadas originales no se recuperaron, por lo que se registra exposición potencial previa. El test no contiene esas dos muestras. No se cambiaron las particiones para ocultar esta salvedad.

Regla de decisión común: `anomaly` si `probability >= threshold`; las probabilidades de los métodos tienen orígenes distintos. La selección usa grilla 0,05 a 0,95, paso 0,05; exige FPR empírico ≤0,10 y maximiza recall, luego F1, luego minimiza FPR. Un empate total conserva el primer umbral en orden ascendente. API quedó en 0,20; YOLO en 0,50. No se usaron probabilidades YOLO test para elegir su umbral. La calibración es selección de punto operativo; no se ajustó ni demostró calibración probabilística con isotonic/Platt o diagramas de fiabilidad.

TP/TN/FP/FN se calculan con anomalía positiva. Accuracy=(TP+TN)/N; precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2TP/(2TP+FP+FN); FPR=FP/(FP+TN); FNR=FN/(FN+TP). Los nuevos reportes añaden intervalos de Wilson al 95% para accuracy, recall y FPR. Estos intervalos binomiales no incluyen incertidumbre de selección de checkpoint/umbral ni dependencia entre imágenes. McNemar exacto compara aciertos de ambos métodos sobre las mismas imágenes; se interpreta como análisis exploratorio.

Las máscaras originales contienen IDs 0–4, no solo 0/255. Positivo se define como `mask > 0`. [Mapa oficial de clases](https://raw.githubusercontent.com/amazon-science/spot-diff/main/utils/id2class.py): 1 Bent, 2 Melt, 3 Scratch, 4 Missing. `image_anno.csv` conserva etiquetas de defecto, a veces múltiples. No existen máscaras guardadas para imágenes normales. Se verificaron las 100 máscaras anómalas asignadas a exploración/train/validation/test; no se usaron para entrenamiento o selección.
