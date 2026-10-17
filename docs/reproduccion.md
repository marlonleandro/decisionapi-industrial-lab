# Reproducción y conservación de evidencia

Todas las rutas relativas parten del repositorio original. Usar su `.venv`, que ya funciona. No regenerar splits, copiar de nuevo el dataset ni reentrenar para leer los resultados existentes.

```powershell
Set-Location 'D:\Proyectos\IA\Lab\DecisionAPI'
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
Get-Content results/audit_20261013/comparison_test.json
Get-Content results/audit_20261013/yolo_threshold_frozen.json
```

La evidencia de 2026-10-13 se reutiliza. Para una reproducción local nueva con los mismos pesos/particiones y una carpeta que no exista:

```powershell
.\.venv\Scripts\python.exe scripts/offline_audit_compare.py audit --output results/reproduccion_nueva
.\.venv\Scripts\python.exe scripts/offline_audit_compare.py freeze --output results/reproduccion_nueva
# Revisar audit.json y yolo_threshold_frozen.json antes del paso siguiente.
.\.venv\Scripts\python.exe scripts/offline_audit_compare.py test --output results/reproduccion_nueva
```

El primer paso verifica counts, exclusiones, semilla, hashes y copias; el segundo infiere solo validation y verifica coincidencia con CSV histórico antes de congelar. El tercero exige el congelamiento, comprueba que modelo/manifest/validación/imágenes no cambiaron y compara las mismas 60 imágenes. Las salidas usan apertura exclusiva y se rechaza repetir una etapa ya terminada en su directorio. Una interrupción que deje archivos parciales requiere revisar esos archivos y usar otro directorio, sin eliminarlos automáticamente. No se importa OpenAI ni se lee `.env`; no hay tráfico de API en este flujo.

El test ya fue visto. Repetir inferencia de estos pesos solo verifica reproducibilidad; no crea una evaluación nueva ni permite ajustar decisiones al test. Los diagnósticos adicionales `scripts/document_verified_state.py` generan estadísticas de máscaras, errores y docs exclusivamente en rutas nuevas; su ejecución es única para esta continuación y rechaza salidas existentes.

## Scripts históricos que pueden sustituir salidas o generar gasto

`prepare_benchmark.py` y `prepare_dataset_splits.py` reescriben manifests. `prepare_yolo_dataset.py` copia sobre imágenes existentes. `evaluate_yolo.py` y `evaluate_final_test.py` reescriben sus CSV/JSON. `run_benchmark.py` hoy fija V2, no implementa cache seguro y reescribe `predictions_v2.csv`; ejecutarlo repite llamadas. `run_benchmark_v4.py` reutiliza éxitos, pero su cache se identifica solo por ruta y carpeta V1/V2/split, no por contenido/prompt. No ejecutar ninguno solo para actualizar documentación. `train_yolo.py` crea una nueva corrida con exist_ok=False, pero debe planificarse como experimento nuevo, nunca como parte de releer métricas.

Para un clon limpio, instalar primero dependencias en un entorno nuevo y verificar PyTorch CUDA; usar [instrucciones oficiales de PyTorch](https://pytorch.org/get-started/locally/) para la rueda que corresponda a hardware/plataforma. `requirements-yolo-observed.txt` conserva las versiones funcionales actuales. La reproducción exacta requiere las imágenes originales, manifiestos, best.pt y CSV/checkpoints guardados, cuyos hashes se encuentran en la auditoría.

Para un nuevo experimento de API, primero crear un directorio y protocolo diferentes, comprobar cobertura del cache y requerimientos del modelo, después proporcionar `OPENAI_API_KEY` mediante entorno o `.env` local ignorado por Git. Nunca imprimir la clave ni guardar archivos de credenciales en reportes. Los reportes actuales no contienen valores de claves y no se abrió `.env`.
