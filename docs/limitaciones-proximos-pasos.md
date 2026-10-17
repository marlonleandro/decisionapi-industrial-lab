# Limitaciones y próximos pasos

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
