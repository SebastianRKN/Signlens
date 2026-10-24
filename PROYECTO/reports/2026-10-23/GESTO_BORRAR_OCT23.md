# Gesto de borrar — implementación inicial

Actualización de entrega: replay del video FAIL (0 de 2 empujes). El gesto queda desactivado al iniciar; G permite probarlo expresamente. Retroceso es el borrado estable. Las 79 pruebas de software de la entrega no validan la precisión física de este gesto.

Con la palma abierta, desplaza la mano horizontalmente desde su lado inicial hacia el lado contrario, como en el video enviado. Cada empuje aceptado borra el último carácter, incluido un espacio. Ctrl+Z permite deshacerlo.

El detector usa ambas manos por separado y las coordenadas de imagen: la dirección se calcula desde el lado donde empieza la mano, por lo que también funciona geométricamente con la cámara reflejada. Exige tres dedos extendidos, un recorrido de al menos el 20 % del ancho de imagen y predominio horizontal. Dos empujes simultáneos producen un solo borrado. Hay un segundo de espera tras cada borrado. Las letras de la mano en movimiento se suspenden mientras se evalúa el gesto; una palma quieta puede seguir escribiendo después de la espera inicial.

No cambia el modelo de letras, ni los esqueletos, ni añade botones. El comando funciona en la vista de subtítulos, no durante captura o entrenamiento.

## Evidencia y límites

Se inspeccionaron fotogramas y landmarks del video de referencia (7,07 segundos, aproximadamente 10 FPS). El seguimiento pierde puntos durante ambos desplazamientos rápidos y llega a cambiar la identidad de una mano. El detector rechaza interrupciones de más de 0,25 segundos para evitar interpretar saltos como borrados. Por ello NO se afirma que el video completo sea reconocido correctamente con esta implementación.

Compilación de fuentes: correcta. Pruebas automatizadas: no ejecutadas. Validación física del gesto y falsos borrados: pendiente. Empezar con un movimiento amplio y moderado, manteniendo la mano visible. Si sigue perdiéndose, el siguiente paso es mejorar seguimiento durante el barrido, no reducir indiscriminadamente las protecciones.

Es un comando de interfaz definido por movimiento, no un modelo entrenado para letras dinámicas ni una validación de LSEC.
