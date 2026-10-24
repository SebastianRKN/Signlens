# Plan para letras con movimiento

Estado: propuesta pendiente de implementación y validación. No se ha entrenado ni activado un reconocedor temporal.

## Datos antes del modelo

1. Acordar el alfabeto y validar las letras y movimientos con una referencia o persona competente en LSEC. No cambiar etiquetas ASL por LSEC.
2. Grabar clips cortos por letra, incluyendo el comienzo y el final, junto con movimientos de transición y ejemplos sin seña.
3. Como objetivo inicial de recolección, reunir 30–50 repeticiones por letra en al menos tres sesiones reales, con variedad de velocidad y posición. Es una propuesta de arranque, no una cantidad que garantice precisión.
4. Reservar sesiones completas para evaluación antes de entrenar e incorporar otras personas cuando sea posible.

## Representación propuesta

Por fotograma: tiempo real de captura, lado de la mano, presencia/visibilidad, coordenadas de los 21 puntos,
forma relativa de los dedos y trayectoria de muñeca/puntas en la imagen respecto al cuerpo. Mantener identidad izquierda/derecha.
Los puntos centrados en la mano conservan su forma, pero se necesita además la trayectoria para representar desplazamiento.
La API de Hand Landmarker ofrece landmarks y exige marcas de tiempo para vídeo; el módulo actual ya proporciona
landmarks de imagen y mundo. [Referencia oficial de MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/hand_landmarker.py).

El grabador de secuencias existente guarda vectores de características, pero requiere ampliar metadatos de persona/sesión,
tiempos, presencia y límites de gesto antes de usarlo como conjunto temporal evaluable. No mezclar estos clips con las tomas estáticas.

## Primer experimento

- Comenzar con letras aisladas y ventanas delimitadas, antes de segmentar un flujo continuo.
- Comparar secuencias mediante DTW como referencia inicial: permite alinear series de distinta duración.
  [Documentación oficial de tslearn](https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html).
- Calibrar rechazo con ejemplos desconocidos y transiciones. Si los datos lo justifican, comparar después un modelo temporal compacto.
- Medir la separación de letras de forma similar y las falsas aceptaciones durante movimientos cotidianos.

## Integración posterior

Cada mano tendrá su propio historial temporal. Al completar un gesto, el módulo emitirá un único evento de letra hacia
el compositor de subtítulos. La integración deberá evitar emitir letras estáticas intermedias mientras se ejecuta
un gesto dinámico y resolver los eventos simultáneos con la misma regla de orden.

## Condiciones para habilitarlo

Informe por sesión/persona reservada con acierto por gesto, rechazos, falsas letras en reposo/transiciones, latencia
y errores de caracteres en palabras completas. Medir captura y procesamiento por separado en este equipo.
No presentar el módulo como listo solo por funcionar con las repeticiones usadas para entrenarlo.
