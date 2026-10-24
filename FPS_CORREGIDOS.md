# Corrección de captura y FPS — 9 de octubre de 2026

Cambios sobre la versión nueva del ZIP:
- Sondeo exclusivo: libera la cámara antes de probar otro modo o backend; reabre el ganador y verifica que entregue frames.
- Rechaza modos y cachés con cero frames; prioriza resolución solicitada cuando las diferencias de velocidad son menores del 5%.
- FPS calculados contando llegadas en una ventana temporal, evitando promediar inversos de intervalos que inflaban el contador con ráfagas.
- Publicación y consumo del indicador de nuevo frame bajo el mismo bloqueo.
- Marcas de tiempo estrictamente crecientes para MediaPipe: evita que se detenga la pose con frames entregados en el mismo milisegundo.

Mediciones reales en este equipo (muestras de 8 segundos; las escenas no fueron idénticas):
- DirectShow sin reconocimiento: 9,59 FPS. Con seguimiento/render: captura 10,07; procesamiento 9,32 FPS (29 clasificaciones de mano fresca).
- Sondeo secuencial: DirectShow aproximadamente 10 FPS aunque reporta 60; Media Foundation aproximadamente 31 FPS.
- Versión corregida, Media Foundation: captura sola 29,96 FPS; con seguimiento/render captura 30,00 y procesamiento 20,04 FPS. En esta última muestra no había detecciones frescas de la mano Right, por lo que no mide rendimiento sostenido del clasificador con una mano presente.
- Prueba aislada del clasificador sobre un ejemplo guardado: media 10,14 ms; limitar BLAS a un hilo dio 9,61 ms. No se cambió esa configuración porque la ganancia fue pequeña.

41 pruebas pasaron. La diferencia captura/procesamiento se muestra por separado: no se prometen 60 FPS. La precisión del modelo no cambió; no se reentrenó.

Abrir ABRIR_APP.cmd. Usar el sondeo automático normal (no --no-probe, que elige DirectShow en este equipo). --reprobe vuelve a medir si cambia la cámara.
