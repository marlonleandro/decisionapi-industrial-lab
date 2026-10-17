# Prompts preservados V1/V2

Fuente principal: `src/decisions/client.py` actual. Los CSV de exploración no registran versión explícita y no se recuperaron las respuestas originales del chat: la atribución V1/V2 se basa en los nombres de archivo y el código conservado. No puede demostrarse retroactivamente que el texto actual de V1 coincide byte por byte con el enviado en exploración. Los checkpoints v4 sí registran `prompt_version=v1`, pero no guardan instrucciones. Se conservó hoy una instantánea con hashes en `results/audit_20261013/prompt_snapshot.json`.

Contexto común: “Inspect this printed circuit board (PCB) as part of an industrial visual quality-control process.” Modelo: `gpt-6-luna`. Pregunta: `predicate`, nombre `visible_defect`.

## V1 exacto del archivo actual

```text
Does this PCB show a visible manufacturing anomaly or defect? Look for observable abnormalities such as missing or misplaced components, surface damage, scratches, cracks, unusual spots, or other visible structural inconsistencies. Ignore ordinary lighting variation, shadows, reflections, perspective differences, and normal component variation.
```

## V2 exacto del archivo actual

```text

        Determine whether this HC-SR04 ultrasonic sensor module
        contains any visible manufacturing anomaly or defect.

        Expected configuration:
        - Four connector pins labeled Vcc, Trig, Echo and Gnd.
        - All four pins must be physically present.
        - Pins should be straight and properly aligned.
        - No visibly broken, bent or missing pins.
        - No abnormal solder joints or damaged PCB traces.
        - No missing or displaced components.
        - No unusual physical damage or contamination.

        Pay special attention to:
        1. Missing connector pins.
        2. Bent or misaligned pins.
        3. Physical damage around solder joints.
        4. Scratches or damage on the PCB.
        5. Deformation of ultrasonic transducers.

        Ignore normal lighting differences, shadows,
        reflections and harmless cosmetic variation.

        A visually evident deviation from the expected
        configuration counts as a defect.

        Do not infer hidden electrical defects from
        visual appearance alone.
        
```

V2 describe un módulo HC-SR04 y una configuración concreta de pines/transductores. La inspección visual de los casos documentados sí muestra módulos HC-SR04: no se sostiene afirmar un desajuste de dominio por ese nombre. V2 puede cambiar la sensibilidad a configuración, geometría o variaciones visibles, pero el descenso observado de precision no demuestra la causa de sus seis FP. Hace falta una revisión caso por caso y un experimento nuevo para atribuirla. No se retocó V1/V2 ni se volvió a consultar la API después de mirar test.

Un eventual V3 debe apoyarse en anomalías anotadas de pcb1, evitar requisitos no verificados del objeto y reservar un conjunto de evaluación nuevo. Cambiar prompt, modelo, imagen o instrucciones exige nueva identidad de experimento; no reutilizar el caché histórico bajo una versión existente.
