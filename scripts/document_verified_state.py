from pathlib import Path
import ast, csv, hashlib, importlib.metadata as md, json, statistics, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(sys.argv[1]).resolve()
OUT=ROOT/'results/audit_20261013'
def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: f.write(value)
def js(p,value): put(p,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def csvout(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def metric(rows,t):
    tp=tn=fp=fn=0
    for r in rows:
        pos=r['ground_truth']=='anomaly'; pred=float(r['probability'])>=t
        if pos and pred: tp+=1
        elif pos: fn+=1
        elif pred: fp+=1
        else: tn+=1
    return dict(samples=len(rows),threshold=t,accuracy=(tp+tn)/len(rows),precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),fpr=fp/(fp+tn),fnr=fn/(fn+tp),tp=tp,tn=tn,fp=fp,fn=fn)

annotations={r['image']:r for r in read_csv(ROOT/'data/raw/pcb1/image_anno.csv')}
manifest=read_csv(ROOT/'data/pcb1_splits.csv')
explore=read_csv(ROOT/'data/benchmark_pcb1.csv')
masks=[]
for r in explore+manifest:
    if r['ground_truth']!='anomaly': continue
    a=annotations[r['image'].removeprefix('data/raw/')]
    mask_path=ROOT/'data/raw'/a['mask']
    mask=np.array(Image.open(mask_path)); image=Image.open(ROOT/r['image'])
    assert mask.ndim==2 and mask.shape== (image.height,image.width)
    positive=mask>0; ys,xs=np.where(positive); assert len(xs)>0
    width,height=image.size
    scale=320/min(width,height)
    nw,nh=(320,int(320*height/width)) if width<=height else (int(320*width/height),320)
    resized=Image.fromarray(positive.astype('uint8')).resize((nw,nh),Image.Resampling.NEAREST)
    x=int(round((nw-320)/2)); y=int(round((nh-320)/2))
    projected=np.array(resized); cropped=np.array(resized.crop((x,y,x+320,y+320)))
    masks.append(dict(image=r['image'],split=r.get('split','exploration'),mask=a['mask'],mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),annotation_labels=a['label'],pixel_ids=','.join(map(str,np.unique(mask[positive]).tolist())),width=width,height=height,anomaly_pixels=int(positive.sum()),area_fraction=float(positive.mean()),bbox_xmin=int(xs.min()),bbox_ymin=int(ys.min()),bbox_xmax=int(xs.max()),bbox_ymax=int(ys.max()),projected_positive_pixels=int(projected.sum()),center_crop_positive_pixels=int(cropped.sum()),approx_retained_fraction=float(cropped.sum()/projected.sum()) if projected.sum() else None))
csvout(OUT/'mask_statistics.csv',masks)
maskby={r['image']:r for r in masks}
configs=[('API_exploration_v1',ROOT/'results/predictions.csv',.5),('API_exploration_v2',ROOT/'results/predictions_v2.csv',.5),('API_validation_v1',ROOT/'results/runs/v1_validation/predictions.csv',.2),('API_test_v1',ROOT/'results/runs/v1_test/predictions.csv',.2),('YOLO_validation',OUT/'yolo_predictions_validation.csv',.5),('YOLO_test',OUT/'yolo_predictions_test.csv',.5)]
errors=[]; experiments={}
for name,path,t in configs:
    rows=read_csv(path); experiments[name]=metric(rows,t)
    for r in rows:
        pred=float(r['probability'])>=t; pos=r['ground_truth']=='anomaly'
        if pred!=pos:
            m=maskby.get(r['image'],{})
            errors.append(dict(experiment=name,image=r['image'],error='FN' if pos else 'FP',probability=float(r['probability']),threshold=t,annotation_labels=m.get('annotation_labels','normal'),mask_area_fraction=m.get('area_fraction',''),approx_crop_retained_fraction=m.get('approx_retained_fraction','')))
csvout(OUT/'error_inventory.csv',errors)
js(OUT/'historical_metrics_recomputed.json',experiments)
summaries={}
for split in ['exploration','train','validation','test']:
    group=[m for m in masks if m['split']==split]
    summaries[split]=dict(anomalies=len(group),area_fraction_min=min(m['area_fraction'] for m in group),area_fraction_median=statistics.median(m['area_fraction'] for m in group),area_fraction_max=max(m['area_fraction'] for m in group),approx_crop_total_loss=sum(m['approx_retained_fraction']==0 for m in group))
js(OUT/'mask_summary.json',summaries)
js(OUT/'packages_snapshot.json',{d.metadata['Name']:d.version for d in sorted(md.distributions(),key=lambda d:d.metadata['Name'].lower())})

# Analytical figure: original images with ground-truth contours, not model localization.
chosen=[r for r in errors if r['experiment'] in ['API_test_v1','YOLO_test'] and r['error']=='FN']
unique=list(dict.fromkeys(r['image'] for r in chosen))
fig,axes=plt.subplots((len(unique)+2)//3,3,figsize=(12,4*((len(unique)+2)//3)),squeeze=False)
for ax in axes.flat: ax.axis('off')
for ax,key in zip(axes.flat,unique):
    image=Image.open(ROOT/key); m=maskby[key]
    ax.imshow(image); mask=np.array(Image.open(ROOT/'data/raw'/m['mask']))>0
    ax.contour(mask,levels=[.5],colors=['red'],linewidths=.9)
    names=', '.join(r['experiment'].replace('_test_v1','').replace('_test','') for r in chosen if r['image']==key)
    ax.set_title(f'{Path(key).name} | FN: {names}\n{m["annotation_labels"]}; area {100*m["area_fraction"]:.3f}%',fontsize=9)
fig.suptitle('Test false negatives: red contours are VisA ground truth',fontsize=14)
fig.tight_layout(); fig.savefig(OUT/'test_false_negative_masks.png',dpi=150); plt.close(fig)

tree=ast.parse((ROOT/'src/decisions/client.py').read_text(encoding='utf-8'))
prompts={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in ['PROMPT_V1','PROMPT_V2']}
js(OUT/'prompt_snapshot.json',{k:{'text':v,'sha256_utf8':hashlib.sha256(v.encode()).hexdigest()} for k,v in prompts.items()})
audit=json.loads((OUT/'audit.json').read_text()); comparison=json.loads((OUT/'comparison_test.json').read_text()); env=json.loads((OUT/'environment.json').read_text()); freeze=json.loads((OUT/'yolo_threshold_frozen.json').read_text())
docs=ROOT/'docs/continuacion-20261013'
index='''# DecisionAPI-vs-YOLO — estado verificado al 13 de octubre de 2026

Proyecto local: `D:\\Proyectos\\IA\\Lab\\DecisionAPI`. Esta continuación conserva los scripts, pesos, CSV y reportes anteriores. El README original permanece como registro del Sprint 1; esta guía reúne el estado actual.

Se auditó el repositorio, se reprodujo exactamente la validación YOLO y se congeló su umbral en **0,50** antes de inferir test. En las mismas 60 imágenes independientes de entrenamiento, YOLO alcanzó **93,33% accuracy / 96,67% recall** y Decisions API V1 **81,67% / 76,67%**. Son resultados de laboratorio; ver limitaciones.

## Documentación

- [Arquitectura y alcance](docs/continuacion-20261013/arquitectura.md)
- [Dataset, particiones y metodología](docs/continuacion-20261013/dataset-metodologia.md)
- [Prompts V1/V2 exactos](docs/continuacion-20261013/prompts.md)
- [Entorno RTX 5050 y entrenamiento](docs/continuacion-20261013/entorno-entrenamiento.md)
- [Reproducción sin nuevas llamadas a la API](docs/continuacion-20261013/reproduccion.md)
- [Limitaciones y próximos pasos](docs/continuacion-20261013/limitaciones-proximos-pasos.md)
- [Procedencia y alcance del chat recuperado](docs/continuacion-20261013/procedencia.md)
- [Reporte completo con errores y máscaras](reports/comparacion-20261013.md)
- [Evidencia estructurada](results/audit_20261013/audit.json)

## Archivos nuevos de evaluación

`scripts/offline_audit_compare.py` separa `audit`, `freeze` y `test`, valida hashes, y rechaza sustituir salidas existentes. Las salidas están en `results/audit_20261013/`; se reutilizaron los checkpoints de Decisions API. No se ejecutaron llamadas pagadas ni un nuevo entrenamiento.

El directorio contiene hashes originales, hashes de las 340 imágenes, entorno y dependencias, grilla de umbrales, umbral congelado, predicciones YOLO, comparación pareada, intervalos Wilson, métricas históricas recalculadas, inventario de errores, máscaras y figura diagnóstica. Las máscaras se usaron después de la evaluación para diagnóstico, nunca como entrada al modelo.
'''
put(ROOT/'README-comparacion.md',index)
put(docs/'arquitectura.md','''# Arquitectura y alcance

El problema es clasificación binaria por imagen: `normal` frente a `anomaly` en VisA pcb1. La clase positiva es anomalía. YOLO11n-cls es un clasificador, y no produce cajas ni máscaras. Decisions API responde al predicado visual `visible_defect`; ninguno de los dos métodos certifica funcionamiento eléctrico.

```mermaid
flowchart LR
  A[VisA pcb1 y anotaciones] --> B[Manifiestos reproducibles]
  B --> E[Exploracion V1 y V2]
  B --> T[Train: 240 imagenes]
  T --> Y[YOLO11n-cls entrenado]
  B --> V[Validation: 40 imagenes]
  V --> C[Seleccion de umbral por metodo]
  Y --> C
  C --> F[Umbrales congelados y hashes]
  B --> S[Test: 60 imagenes]
  F --> P[Comparacion pareada]
  S --> P
  P --> R[Metricas y diagnostico con mascaras]
```

En `src/decisions/client.py`, la imagen JPEG/PNG/WebP se convierte en data URL; el cliente envía contexto y una pregunta `predicate` a `client.decisions.create(model="gpt-6-luna", ...)`. Extrae una probabilidad, no una explicación ni una localización. La [referencia oficial de Decisions API](https://developers.openai.com/api/reference/resources/decisions/methods/create) describe este contrato.

`run_benchmark_v4.py` separa validation/test y conserva respuestas crudas, latencia, estado, modelo y timestamp. El identificador de checkpoint es SHA-256 de la ruta, truncado a 16 caracteres. Los éxitos se reutilizan y los errores pueden reintentarse. Ese caché histórico no vincula la identidad del prompt o los bytes de la imagen: para experimentos nuevos se requieren nuevos directorios/versiones y hashes. Los hashes agregados hoy documentan el estado actual; no prueban retroactivamente el texto exacto enviado en todas las llamadas anteriores.

Para YOLO, `prepare_yolo_dataset.py` copia cada imagen del manifest a `train/val/test` y `normal/anomaly`. `train_yolo.py` ajusta los pesos preentrenados; `best.pt` se selecciona con validación. `offline_audit_compare.py` comprueba esa selección con inferencia de validación, congela el umbral y compara test con los CSV de API sin importar el cliente ni cargar `.env`.

No hay interfaz de producción, cola de revisión humana ni reglas PASS/MANUAL_REVIEW/REJECT implementadas en la evidencia revisada. Son próximos pasos. Las carpetas `notebooks/` y `scr/` no constituyen evidencia de resultados ejecutados.
''')
put(docs/'dataset-metodologia.md','''# Dataset, particiones y metodología

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
''')
put(docs/'prompts.md','''# Prompts preservados V1/V2

Fuente principal: `src/decisions/client.py` actual. Los CSV de exploración no registran versión explícita y no se recuperaron las respuestas originales del chat: la atribución V1/V2 se basa en los nombres de archivo y el código conservado. No puede demostrarse retroactivamente que el texto actual de V1 coincide byte por byte con el enviado en exploración. Los checkpoints v4 sí registran `prompt_version=v1`, pero no guardan instrucciones. Se conservó hoy una instantánea con hashes en `results/audit_20261013/prompt_snapshot.json`.

Contexto común: “Inspect this printed circuit board (PCB) as part of an industrial visual quality-control process.” Modelo: `gpt-6-luna`. Pregunta: `predicate`, nombre `visible_defect`.

## V1 exacto del archivo actual

```text
'''+prompts['PROMPT_V1']+'''
```

## V2 exacto del archivo actual

```text
'''+prompts['PROMPT_V2']+'''
```

V2 describe un módulo HC-SR04 y una configuración concreta de pines/transductores. La inspección visual de los casos documentados sí muestra módulos HC-SR04: no se sostiene afirmar un desajuste de dominio por ese nombre. V2 puede cambiar la sensibilidad a configuración, geometría o variaciones visibles, pero el descenso observado de precision no demuestra la causa de sus seis FP. Hace falta una revisión caso por caso y un experimento nuevo para atribuirla. No se retocó V1/V2 ni se volvió a consultar la API después de mirar test.

Un eventual V3 debe apoyarse en anomalías anotadas de pcb1, evitar requisitos no verificados del objeto y reservar un conjunto de evaluación nuevo. Cambiar prompt, modelo, imagen o instrucciones exige nueva identidad de experimento; no reutilizar el caché histórico bajo una versión existente.
''')
put(docs/'entorno-entrenamiento.md',f'''# Entorno RTX 5050 y entrenamiento verificado

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

SHA-256 best.pt: `{audit['model_sha256']}`.

La inferencia de validación de hoy reproduce exactamente las 40 probabilidades históricas (máxima diferencia absoluta 0). Clase 0=anomaly, clase 1=normal; el evaluador busca el nombre en lugar de asumir el índice. Las transformaciones reales son Resize(320), CenterCrop(320×320), ToTensor y Normalize(mean=0,std=1), registradas en entorno. La [documentación de clasificación Ultralytics](https://docs.ultralytics.com/tasks/classify) describe los recortes de entrenamiento/inferencia; pueden ocultar regiones importantes. No se cambió esa transformación después de ver test.

`requirements.txt` original no incluye torch/torchvision/ultralytics. Se añadió `requirements-yolo-observed.txt` como referencia de versiones observadas y se conservó el original. No equivale a un lock de todas las dependencias; la instantánea completa figura en packages_snapshot.json. Reentrenar no garantiza pesos idénticos incluso con deterministic=True en otra plataforma/versión.
''')
put(ROOT/'requirements-yolo-observed.txt','# Versiones funcionales observadas 2026-10-13; conservar wheel CUDA 12.8.\n'+ '\n'.join(f'{p}=={md.version(p)}' for p in ['torch','torchvision','ultralytics','openai','pandas','scikit-learn','numpy','Pillow','matplotlib'])+'\n')
put(docs/'reproduccion.md','''# Reproducción y conservación de evidencia

Todas las rutas relativas parten del repositorio original. Usar su `.venv`, que ya funciona. No regenerar splits, copiar de nuevo el dataset ni reentrenar para leer los resultados existentes.

```powershell
Set-Location 'D:\\Proyectos\\IA\\Lab\\DecisionAPI'
.\\.venv\\Scripts\\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
Get-Content results/audit_20261013/comparison_test.json
Get-Content results/audit_20261013/yolo_threshold_frozen.json
```

La evidencia de 2026-10-13 se reutiliza. Para una reproducción local nueva con los mismos pesos/particiones y una carpeta que no exista:

```powershell
.\\.venv\\Scripts\\python.exe scripts/offline_audit_compare.py audit --output results/reproduccion_nueva
.\\.venv\\Scripts\\python.exe scripts/offline_audit_compare.py freeze --output results/reproduccion_nueva
# Revisar audit.json y yolo_threshold_frozen.json antes del paso siguiente.
.\\.venv\\Scripts\\python.exe scripts/offline_audit_compare.py test --output results/reproduccion_nueva
```

El primer paso verifica counts, exclusiones, semilla, hashes y copias; el segundo infiere solo validation y verifica coincidencia con CSV histórico antes de congelar. El tercero exige el congelamiento, comprueba que modelo/manifest/validación/imágenes no cambiaron y compara las mismas 60 imágenes. Las salidas usan apertura exclusiva y se rechaza repetir una etapa ya terminada en su directorio. Una interrupción que deje archivos parciales requiere revisar esos archivos y usar otro directorio, sin eliminarlos automáticamente. No se importa OpenAI ni se lee `.env`; no hay tráfico de API en este flujo.

El test ya fue visto. Repetir inferencia de estos pesos solo verifica reproducibilidad; no crea una evaluación nueva ni permite ajustar decisiones al test. Los diagnósticos adicionales `scripts/document_verified_state.py` generan estadísticas de máscaras, errores y docs exclusivamente en rutas nuevas; su ejecución es única para esta continuación y rechaza salidas existentes.

## Scripts históricos que pueden sustituir salidas o generar gasto

`prepare_benchmark.py` y `prepare_dataset_splits.py` reescriben manifests. `prepare_yolo_dataset.py` copia sobre imágenes existentes. `evaluate_yolo.py` y `evaluate_final_test.py` reescriben sus CSV/JSON. `run_benchmark.py` hoy fija V2, no implementa cache seguro y reescribe `predictions_v2.csv`; ejecutarlo repite llamadas. `run_benchmark_v4.py` reutiliza éxitos, pero su cache se identifica solo por ruta y carpeta V1/V2/split, no por contenido/prompt. No ejecutar ninguno solo para actualizar documentación. `train_yolo.py` crea una nueva corrida con exist_ok=False, pero debe planificarse como experimento nuevo, nunca como parte de releer métricas.

Para un clon limpio, instalar primero dependencias en un entorno nuevo y verificar PyTorch CUDA; usar [instrucciones oficiales de PyTorch](https://pytorch.org/get-started/locally/) para la rueda que corresponda a hardware/plataforma. `requirements-yolo-observed.txt` conserva las versiones funcionales actuales. La reproducción exacta requiere las imágenes originales, manifiestos, best.pt y CSV/checkpoints guardados, cuyos hashes se encuentran en la auditoría.

Para un nuevo experimento de API, primero crear un directorio y protocolo diferentes, comprobar cobertura del cache y requerimientos del modelo, después proporcionar `OPENAI_API_KEY` mediante entorno o `.env` local ignorado por Git. Nunca imprimir la clave ni guardar archivos de credenciales en reportes. Los reportes actuales no contienen valores de claves y no se abrió `.env`.
''')
put(docs/'limitaciones-proximos-pasos.md','''# Limitaciones y próximos pasos

## Límites que afectan la conclusión

1. Solo pcb1, test de 60 imágenes, 30 anomalías y 30 normales. Los IC son amplios y no representan una línea industrial con otra prevalencia.
2. Solo 40 anomalías de entrenamiento, train 5:1; split local, no protocolo oficial. No se verificó separación por lote, pieza física o captura ni duplicados perceptuales.
3. Validation sirve tanto para elegir best.pt como para elegir umbral. Su rendimiento está condicionado por selección; no es estimación final independiente. La muestra inicial Anomaly/000.JPG ya estaba preparada antes de quedar en validation, y sus respuestas originales no fueron recuperadas.
4. API test ya fue inspeccionado en la conversación antes de esta continuación. YOLO no se ajustó con él: pesos/umbral quedaron fijados antes de su inferencia test. El conjunto debe retirarse de cualquier futura selección de prompts/arquitectura/hiperparámetros.
5. YOLO recibió entrenamiento específico; Decisions API no. La comparación corresponde a esos sistemas finales y ese presupuesto de datos; no a igualdad de entrenamiento ni a todas las variantes posibles de cada familia.
6. V2 describe HC-SR04, coherente con los casos inspeccionados visualmente. Sus seis FP exploratorios no demuestran un desajuste de dominio ni acreditan que todo prompt más detallado sea peor. El texto histórico exacto de los prompts no está vinculado al cache original por hash.
7. Probabilidad no equivale a calibración comprobada; el umbral es operativo. FPR ≤10% en validation no garantiza ese FPR en una población futura.
8. Recorte central y resolución 320 pueden perder o reducir anomalías. Las estadísticas de máscaras proyectadas son diagnósticas aproximadas con interpolación nearest; el clasificador usa imagen bilinear. No prueban la causa de cada error ni predicción localizada.
9. Los tiempos API históricos incluyen red/servicio y posibles reintentos; los tiempos YOLO incluyen llamada local, preprocesado y arranque. No son un benchmark homogéneo. No se calculó precio en dinero sin facturación verificable.
10. El chat recuperado está incompleto; se preservó su procedencia y no se inventaron anexos, métricas o decisiones ausentes. La independencia comprobada es por rutas y bytes conocidos, no una garantía de ausencia de exposición en preentrenamiento del proveedor.

## Trabajo terminado

Auditoría sin reemplazar evidencia, reconstrucción de métricas, inferencia YOLO validation, congelación 0,50 con hashes, inferencia test, comparación pareada con API cacheada, IC Wilson, análisis de máscaras/errores y documentación. No hubo nuevo entrenamiento, nuevas llamadas pagadas, cambio de prompt ni ajuste posterior al test.

## Planes pendientes — sin métricas atribuidas

1. Revisar manualmente los FP/FN y máscaras con un criterio de anotación consistente; mantener estos hallazgos como exploratorios.
2. Diseñar una nueva separación por lote/captura para categorías adicionales; excluir cualquier imagen vista, incluidas las muestras iniciales. Reservar un test nuevo antes de cambiar el sistema; pcb1 ya agotó sus anomalías para esta partición.
3. Comparar conservación de imagen completa, recortes/tiles de zonas o resolución mayor en validación nueva. Congelar arquitectura y umbral antes de abrir el test nuevo. Estos cambios requieren entrenamiento/evaluación nuevos.
4. Evaluar más semillas y learning curves sobre anomalías supervisadas; medir variabilidad del entrenamiento y comparar modelos con igual presupuesto de datos.
5. Proponer un prompt nuevo con contexto correcto de pcb1 y hashes completos del experimento. Revisar cache, cobertura y coste antes de autorizar nuevas llamadas; ninguna se hizo aquí.
6. Medir fiabilidad probabilística y curvas ROC/PR, y seleccionar costes FP/FN según operación real en un conjunto nuevo. Incorporar revisión humana con capacidad y tasas de aceptación medidas.
7. Medir latencia warm/cold, throughput, batch y costes con protocolo comparable. Validar deriva, prevalencia real y defectos no visibles antes de integración operativa.
''')
put(docs/'procedencia.md','''# Procedencia e integridad de esta continuación

Solicitud actual: trabajar sobre el proyecto local DecisionAPI-vs-YOLO y documentar arquitectura, datasets, prompts, particiones, calibración, resultados API/YOLO, máscaras, entorno, reproducción y próximos pasos. Proyecto localizado por el catálogo de proyectos en `D:\\Proyectos\\IA\\Lab\\DecisionAPI`; no se creó un repositorio alternativo ni se cambió rama.

Conversación de referencia: “Información oficial Decision API”, ID `6ac69b1a-0b90-83e9-9294-ee1e506a02c2`. La recuperación devolvió cinco intercambios recientes, sin cursor para anteriores. Las respuestas del asistente eran marcadores `chatgpt-content-reference`, no su contenido, y no se pudieron reconstruir. Por ello esta es documentación integral de los temas y evidencia accesible, no una transcripción integral de todo el chat. Se requiere exportación completa si se pretende un archivo literal de los intercambios ausentes.

Se recuperaron: solicitud de trasladar documentación; archivo adjunto de entrenamiento; reporte NVIDIA con RTX 5050, driver 610.74, CUDA UMD 13.3 y 8.151 MiB; mención anterior “NO GPU” en Python; resultados de test API con 60 imágenes, umbral 0,20, accuracy 0,8166667, TP23/TN26/FP4/FN7. El adjunto es un registro de consola aportado por el usuario; los datos de epochs/resultados se corroboraron con archivos reales y la disponibilidad de GPU se verificó hoy.

Fuentes prioritarias: CSV/manifiestos/checkpoints, JSON de umbral/test/experimento, args.yaml/results.csv, pesos y código. Documentos anteriores `sprint-1.md`, `sprint-4.md` y README se conservan como historial. Al inicio Git tenía HEAD `96d3b44` (feat: sprint 3 completado), sin cambios tracked, y varios artefactos v4/YOLO sin seguimiento; no se añadieron a staging ni se hizo commit/push. No se encontraron AGENTS.md aplicables en el proyecto o sus ancestros inspeccionados.

`results/audit_20261013/original_hashes.json` registra SHA-256 de scripts, docs, resultados, pesos y archivos principales antes de crear documentación. Se excluyeron `.env`, `.git`, `.venv`, cachés Python y datos masivos del inventario general; las 340 imágenes/copies tienen un inventario propio. El README original se conserva íntegro byte por byte; no se añadió contenido a él. Se creó `README-comparacion.md` y el índice `docs/README.md`. La comprobación final debe dejar todos los hashes originales intactos.

Solo se escribieron archivos nuevos, salvo cambios internos de caché no semántico que pudiera producir el runtime. La auditoría de hashes originales comprueba los archivos relevantes. Nunca se abrió ni volcó `.env`. Las nuevas salidas se revisaron para detectar patrones de claves antes de la entrega.

Fuentes externas consultadas el 13-10-2026: [API oficial Decisions](https://developers.openai.com/api/reference/resources/decisions/methods/create), [dataset oficial Amazon Science](https://github.com/amazon-science/spot-diff), [IDs de máscaras](https://raw.githubusercontent.com/amazon-science/spot-diff/main/utils/id2class.py), [clasificación Ultralytics](https://docs.ultralytics.com/tasks/classify). Las métricas publicadas en este proyecto proceden de datos locales, no de ejemplos en esas páginas.
''')
put(ROOT/'docs/README.md','''# Documentación del laboratorio

La [guía actual](../README-comparacion.md) y el [reporte comparativo](../reports/comparacion-20261013.md) consolidan la evidencia al 13 de octubre de 2026.

La carpeta [continuacion-20261013](continuacion-20261013/procedencia.md) conserva metodología, prompts, arquitectura, entorno, reproducción y limitaciones. `sprint-1.md` y `sprint-4.md` permanecen como historial.
''')

def row(label,m): return f'| {label} | {m["samples"]} | {m["threshold"]:.2f} | {100*m["accuracy"]:.2f}% | {100*m["precision"]:.2f}% | {100*m["recall"]:.2f}% | {m["f1"]:.4f} | {m["tp"]}/{m["tn"]}/{m["fp"]}/{m["fn"]} |'
table='| Método / conjunto | N | Umbral | Accuracy | Precision | Recall | F1 | TP/TN/FP/FN |\n|---|---:|---:|---:|---:|---:|---:|---|\n'+ '\n'.join(row(n,m) for n,m in experiments.items())
error_table='| Método | Imagen | Error | Probabilidad | Etiquetas VisA | Área máscara | Retención aproximada crop |\n|---|---|---|---:|---|---:|---:|\n'
for e in errors:
    if e['experiment'] in ['API_test_v1','YOLO_test']:
        area=f'{100*e["mask_area_fraction"]:.4f}%' if e['mask_area_fraction']!='' else '—'
        retain=f'{100*e["approx_crop_retained_fraction"]:.1f}%' if e['approx_crop_retained_fraction']!='' and e['approx_crop_retained_fraction'] is not None else '—'
        error_table+=f'| {e["experiment"]} | {Path(e["image"]).name} | {e["error"]} | {e["probability"]:.5f} | {e["annotation_labels"]} | {area} | {retain} |\n'
report=f'''# Comparación verificada Decisions API vs YOLO11n-cls

Fecha: 13 de octubre de 2026, zona America/Lima. Proyecto original: `D:\\Proyectos\\IA\\Lab\\DecisionAPI`. Evidencia numérica: `results/audit_20261013/`. Estado: evaluación terminada con umbrales fijados en validación; no se ejecutaron llamadas pagadas ni nuevo entrenamiento.

En el test de 60 imágenes de VisA pcb1, YOLO11n-cls obtuvo **93,33% accuracy**, **96,67% recall** y **F1=0,9355**; Decisions API V1 obtuvo **81,67%**, **76,67%** y **F1=0,8070**. YOLO produjo 1 FN y 3 FP; API, 7 FN y 4 FP. Es una comparación de un modelo entrenado específicamente y una API sin ajuste específico, sobre una muestra local pequeña y balanceada.

## Métricas verificadas por fase

{table}

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

En test, el área anómala mediana ocupa {100*summaries['test']['area_fraction_median']:.4f}% de la imagen; rango {100*summaries['test']['area_fraction_min']:.4f}–{100*summaries['test']['area_fraction_max']:.4f}%. Proyectando las máscaras con nearest a Resize(320)+CenterCrop, {summaries['test']['approx_crop_total_loss']} de 30 anomalías pierden todos sus píxeles positivos. Es un diagnóstico aproximado: el procesamiento real de RGB usa bilinear y la máscara no mide evidencia contextual visible.

{error_table}

Los FN de exploración V1 a 0,50 son 047 (p=0,07), 058 (0,11), 097 (0,18), 012 (0,18), coherentes con `results/false_negatives/`. El inventario completo añade FP/FN de V2, API validation/test y YOLO val/test en `error_inventory.csv`; las métricas geométricas se encuentran en `mask_statistics.csv`. Ver también [figura de FN test](../results/audit_20261013/test_false_negative_masks.png); los contornos rojos son ground truth, no localización producida por los modelos.

## Entorno y latencia

RTX 5050 Laptop GPU, driver610.74, memoria8.151MiB, Python3.11.9, PyTorch2.11.0+cu128/CUDA12.8, Ultralytics8.4.177, SDK OpenAI3.29.0. CUDA se confirmó con inferencia. Entrenamiento histórico: 240 imágenes, 21 épocas de un máximo de50, mejor época9, imgsz320/batch16/patience12/seed42, 68,54seg según experiment.json. No se reentrenó.

Latencia test observada: YOLO media {comparison['latency']['yolo_mean_ms']:.2f}ms incluyendo primera llamada, {comparison['latency']['yolo_excluding_first_mean_ms']:.2f}ms excluyéndola; API media histórica {comparison['latency']['api_historical_mean_ms']:.2f}ms. No se calculó un factor de aceleración: son mediciones diferentes, con red/servicio en API y arranque/local en YOLO. No hay coste monetario verificado.

## Integridad, límites y planes

Se verificaron semilla42, ausencia de duplicados por ruta/contenido entre splits, exclusión de las20 imágenes exploratorias y las340 copias. La anomalía000 de la prueba inicial aparece en validation; el test no contiene las muestras iniciales y no comparte datos con train. Esto limita la independencia de validación y queda explícito, sin redefinir splits retrospectivamente.

Hash best.pt: `{audit['model_sha256']}`. Hash manifest: `{audit['manifest_sha256']}`. `yolo_threshold_frozen.json` conserva hashes, política, grilla, score de validación y tiempo de congelación; `original_hashes.json` permite comprobar que se conservaron los archivos anteriores. El README original sigue intacto; [README-comparacion.md](../README-comparacion.md) y [docs](../docs/README.md) son las guías actuales.

La conversación accesible contiene solo cinco intercambios recientes y omite textos de asistentes/turnos previos. Esta documentación reúne los temas solicitados con evidencia real, pero no se presenta como exportación completa del chat. No se inventaron métricas de fases ausentes. Ver [procedencia](../docs/continuacion-20261013/procedencia.md).

Pendiente: categorías/lotes nuevos, test nuevo antes de ajustes, revisión manual de errores, evaluación de imagen completa/resolución/tiles, más semillas, prompt con contexto de dominio confirmado, calibración probabilística y costes FP/FN, latencia comparable y revisión humana. Ninguno de esos planes tiene resultados ejecutados atribuidos. Ver [limitaciones y próximos pasos](../docs/continuacion-20261013/limitaciones-proximos-pasos.md) y [reproducción](../docs/continuacion-20261013/reproduccion.md).
'''
put(ROOT/'reports/comparacion-20261013.md',report)
print(json.dumps(dict(historical_metrics=experiments,mask_summary=summaries,test_errors=[e for e in errors if e['experiment'] in ['API_test_v1','YOLO_test']],documentation_files=9),ensure_ascii=False,indent=2))
