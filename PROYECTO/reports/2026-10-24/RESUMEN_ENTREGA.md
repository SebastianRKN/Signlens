# Verificación de entrega

- Fuente exclusiva: entrega-extraida-oct23.zip; original intacto.
- Windows 11 x64; Python incluido en el ejecutable, sin intérprete externo en PATH durante las comprobaciones portátiles.
- 79 pruebas originales; 86 pruebas finales aprobadas.
- PE AMD64 real; dependencias, los tres clasificadores y los modelos MediaPipe cargados.
- Entrenamiento con el ejecutable: completado, publicación atómica y modelo base conservado. Datos sintéticos, exclusivamente para comprobar el empaquetado.
- Evaluación ejecutada con fixture sintético; sin tomas reservadas reales, la app lo rechaza con un aviso comprensible (salida 1 esperada).
- Rutas con espacios, otro directorio de trabajo, almacenamiento alternativo y ausencia de cámara comprobados.
- Cámara real: cierre limpio, sin errores; 372 fotogramas procesados en el ensayo final. Medidas finales aproximadas: captura 31,2 FPS; procesamiento 19,6 FPS. La selección inicial de backend consumió parte del ensayo de 50 segundos.
- Inspección visual: cámara, esqueleto de mano/brazo, predicción y subtítulos visibles. No se infiere exactitud de reconocimiento a partir de FPS ni de datos sintéticos.
- No se distribuye la caché personal de cámara. Los logs históricos conservan sus resultados; USER_HOME sustituye únicamente rutas personales.
- Sin prueba en otro PC ni validación independiente con nuevos participantes. Borrado gestual y reconocimiento siguen siendo experimentales según su alcance original.

Consulte delivery_verification_*.json, distribution_checks_*.json y MANIFEST.sha256. Los registros incluyen intentos anteriores y correcciones; las verificaciones finales indicadas arriba son las vigentes.
