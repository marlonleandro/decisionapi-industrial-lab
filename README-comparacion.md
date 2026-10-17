# DecisionAPI-vs-YOLO — estado verificado al 13 de octubre de 2026

Proyecto local: `D:\Proyectos\IA\Lab\DecisionAPI`. Esta continuación conserva los scripts, pesos, CSV y reportes anteriores. El README original permanece como registro del Sprint 1; esta guía reúne el estado actual.

Se auditó el repositorio, se reprodujo exactamente la validación YOLO y se congeló su umbral en **0,50** antes de inferir test. En las mismas 60 imágenes independientes de entrenamiento, YOLO alcanzó **93,33% accuracy / 96,67% recall** y Decisions API V1 **81,67% / 76,67%**. Son resultados de laboratorio; ver limitaciones.

## Documentación

- [Arquitectura y alcance](docs/arquitectura.md)
- [Dataset, particiones y metodología](docs/dataset-metodologia.md)
- [Prompts V1/V2 exactos](docs/prompts.md)
- [Entorno RTX 5050 y entrenamiento](docs/entorno-entrenamiento.md)
- [Reproducción sin nuevas llamadas a la API](docs/reproduccion.md)
- [Limitaciones y próximos pasos](docs/limitaciones-proximos-pasos.md)
- [Procedencia y alcance del chat recuperado](docs/procedencia.md)
- [Reporte completo con errores y máscaras](reports/comparacion-20261013.md)
- [Evidencia estructurada](results/audit_20261013/audit.json)

## Archivos nuevos de evaluación

`scripts/offline_audit_compare.py` separa `audit`, `freeze` y `test`, valida hashes, y rechaza sustituir salidas existentes. Las salidas están en `results/audit_20261013/`; se reutilizaron los checkpoints de Decisions API. No se ejecutaron llamadas pagadas ni un nuevo entrenamiento.

El directorio contiene hashes originales, hashes de las 340 imágenes, entorno y dependencias, grilla de umbrales, umbral congelado, predicciones YOLO, comparación pareada, intervalos Wilson, métricas históricas recalculadas, inventario de errores, máscaras y figura diagnóstica. Las máscaras se usaron después de la evaluación para diagnóstico, nunca como entrada al modelo.
