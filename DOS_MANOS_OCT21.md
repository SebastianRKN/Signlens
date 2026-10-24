# Reconocimiento simultáneo de dos manos — 21 de octubre de 2026

Nota posterior: la interfaz actual usa subtítulos automáticos sin botones. Ambas manos escriben;
H solo elige la mano a grabar dentro de DATOS (Tab). Ver `SUBTITULOS_OCT21.md`.
La verificación de 67 pruebas de este documento corresponde a la versión anterior de dos manos.

## Implementado

- El bucle de cámara entrega ambas manos frescas al controlador en cada fotograma.
- Dos historiales de estabilización separados: una mano puede confirmar A mientras la otra confirma L.
- Dos resultados visibles, etiquetados IZQUIERDA y DERECHA, con predicción provisional y estado individual.
- Perder o retener una detección antigua limpia solo el resultado de esa mano. La otra continúa.
- H elige la mano a grabar; no cambia ni interrumpe las predicciones.
- La captura guarda únicamente la mano seleccionada con la etiqueta elegida. Así no se mezclan señas diferentes.
- En una toma reservada solo se oculta el resultado de la mano que se graba.
- Pérdida de cámara, cambio de modo y recarga de modelo limpian ambos historiales.

El modelo y su umbral 0,56 no se modificaron. El inicio sigue en COMPLETO con las 23 letras.
SHA-256 del modelo: `9eaeed88b66a63a6945c8ab85899fb628c8a989ecd70fc28f930f58c88ee6f13`.

## Verificación

- **PASS:** 67 pruebas automatizadas (`67 passed in 18.02s`).
- **PASS:** pruebas de letras distintas, cambio unilateral, pérdida o detección antigua de cualquiera de
  las manos, cambio del selector de captura, aislamiento de muestras y limpieza al perder la cámara.
- **PASS:** reproducción de 21 pares distintos con el clasificador real y landmarks previamente guardados.
  Ejemplos A/B, F/G y W/A; la retirada de la izquierda conservó la letra de la derecha en todos los pares.
- **PASS:** apertura de la app y revisión visual de los dos resultados y los controles.
- **PENDIENTE:** evaluación independiente de precisión con dos manos reales simultáneas, cruces y oclusiones.

Los pares de reproducción se construyeron combinando muestras individuales existentes. Comprueban el
funcionamiento del software, no constituyen una nueva medición de exactitud con firmantes reales.
Detalle: `reports/two_hands_oct21.json`.

La función clasifica cada mano por separado. Las señas cuyo significado depende de combinar ambas manos
o de un movimiento requieren otro modelo y datos de entrenamiento específicos.

## Probar

1. Abrir `ABRIR_APP.cmd`; comprobar que indica COMPLETO.
2. Mostrar ambas manos separadas y dentro de la cámara, con letras distintas.
3. Observar los resultados IZQUIERDA y DERECHA. Cambiar solo una letra y comprobar que la otra se conserva.
4. Retirar una mano: su casilla debe mostrar `--` y la otra debe continuar.
