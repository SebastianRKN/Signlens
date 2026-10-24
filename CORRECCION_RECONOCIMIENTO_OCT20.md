# Corrección de regresión del reconocimiento — 20 de octubre de 2026

## Causa

La actualización activó por defecto un modo limitado a A B D E G K L O S T, elevó el umbral de 0,56
a 0,80 y exigió 0,45 segundos de estabilidad continua. Además, la app se abrió con `--reserve`;
esa opción ocultaba todas las predicciones incluso cuando no se estaba grabando una toma.
Estas restricciones cambiaron el comportamiento del reconocimiento sin haber sido validadas como mejora.

## Corrección

- Inicio en COMPLETO, usando las 23 letras y el umbral original del modelo.
- Confirmación normal mediante votos de fotogramas, sin la espera adicional de 0,45 segundos.
- El modo limitado y su espera adicional son optativos mediante D y se identifican en pantalla.
- Seleccionar destino EVALUAR o etiqueta NINGUNA no oculta las predicciones en reposo.
- Las predicciones se ocultan únicamente durante una captura reservada, para evitar que influyan en la toma.
- Se avisa si la mano visible no coincide con la mano seleccionada.
- Se conservaron las protecciones para guardar tomas sin sobrescribirlas y el entrenamiento por versiones.

## Verificación

PASS: 60 pruebas automatizadas en el entorno local de la entrega.
Incluyen una predicción de F con confianza 0,65 que vuelve a aceptarse con el umbral 0,56,
la activación y desactivación explícita del demo, y la visualización fuera de una captura reservada.

PASS: reproducción sobre las 1.712 muestras de cámara ya guardadas. Con la política original se aceptan
1.709 predicciones; el modo limitado rechazaba 858 de ellas adicionalmente y solo aceptaba 851.
El controlador corregido confirmó ejemplos de las 21 letras presentes en esas capturas.
Esto comprueba la regresión del software: **son datos de entrenamiento, no una medición independiente de precisión**.

El modelo no se modificó ni reentrenó para esta corrección. Su SHA-256 sigue siendo
`9eaeed88b66a63a6945c8ab85899fb628c8a989ecd70fc28f930f58c88ee6f13`.

Evidencia detallada: `reports/recognition_regression_oct20.json`.
La exactitud con nuevas señas en vivo permanece pendiente de una prueba del usuario.
