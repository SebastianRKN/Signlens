# Subtítulos automáticos — 21 de octubre de 2026

## Uso actual

Abre `ABRIR_APP.cmd`. La escritura comienza automáticamente con la cámara.
Las manos izquierda y derecha se reconocen de forma independiente y ambas aportan letras.
No hay selector de mano de escritura, botón de inicio ni botones en pantalla.

- Una letra confirmada se añade después de permanecer estable 0,35 s adicionales a la estabilización del clasificador.
- Mantener la misma seña no repite continuamente la letra.
- Se respeta el orden de confirmación. Si ambas confirman en el mismo fotograma, se añade izquierda y después derecha, según tus lados anatómicos.
- Si ambas manos confirman la misma letra en ese fotograma, se escriben dos letras: ambas contribuyen.
- Para repetir una letra con la misma mano, retírala al menos 0,6 s y vuelve a mostrarla. Una pérdida breve o una predicción incierta no basta.
- Para separar palabras, retira ambas manos al menos 1,4 s. No se inventan espacios a partir de palabras de un diccionario.
- Los subtítulos muestran las últimas dos líneas del texto, hasta 1.000 caracteres en total.
- Arriba aparecen las dos letras actuales. La cámara ocupa todo el área de contenido; el texto aparece en la franja inferior.
- El esqueleto de ambas manos y brazos permanece visible también durante la escritura automática.

## Atajos opcionales

| Tecla | Acción |
|---|---|
| Retroceso | Borrar el último carácter |
| Espacio | Separar palabras manualmente |
| . | Añadir punto y espacio |
| Ctrl+Z | Deshacer edición |
| Ctrl+S | Guardar el texto en `texts/` |
| Ctrl+U | Vaciar texto; se puede deshacer |
| ? | Mostrar/ocultar ayuda |
| Tab | Abrir/cerrar herramientas de datos |
| R | Reconectar cámara |
| Q / Esc | Salir |

Dentro de DATOS, C/Espacio captura, K/L cambia la etiqueta, H elige exclusivamente la mano a grabar,
P cambia la persona, N inicia sesión, V cambia el destino, T entrena, X cancela y E evalúa.
D alterna completo/limitado. La lista de atajos se muestra como texto, sin botones.
La escritura se suspende durante el trabajo con datos y vuelve automáticamente al salir con Tab.
Ambas predicciones siguen funcionando; durante una toma reservada se oculta la de la mano grabada.

El texto permanece en memoria mientras la app está abierta. Guarda con Ctrl+S si deseas conservarlo.

## Alcance y cambios

Esta versión sustituye la propuesta anterior de una sola mano elegida para escribir y del panel lateral con botones.
Modifica `composer.py`, `runtime.py`, `panel.py` y `app.py`. Actualiza las expectativas de las pruebas existentes
que describían la interfaz anterior. No modifica los pesos del modelo, el vocabulario ni el umbral 0,56.

El compositor une las letras que confirma el clasificador. Los errores de clasificación pueden llegar al texto;
no es una evaluación independiente de exactitud ni traducción de gramática LSEC. El orden de dos señas simultáneas
es una convención visible en la ayuda, no una intención inferida de la persona.

La ejecución previa de 74 pruebas pertenece a la interfaz anterior. No se ejecutó de nuevo la suite en esta actualización.
El código actualizado se compiló con `compileall`. La revisión visual del arranque se registra por separado.
Las letras con movimiento requieren los datos y el modelo descritos en `reports/sin_fecha/PLAN_LETRAS_CON_MOVIMIENTO.md`.
