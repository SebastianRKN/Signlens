# SignLens — guía para el jurado

1. Extraiga **todo el ZIP** a una carpeta local, por ejemplo Documentos. No abra el programa dentro del ZIP.
2. Abra **EJECUTABLE_WINDOWS**.
3. Haga doble clic en **SignLens.exe**. Conserve `_internal` junto al ejecutable. El primer inicio puede tardar mientras carga las bibliotecas y copia los recursos locales.
4. Permita el uso de la cámara en Windows si el sistema lo solicita. Cierre otras aplicaciones que estén utilizando la cámara.
5. Coloque las manos frente a la cámara, con buena iluminación. Mantenga cada letra estable: ambas manos se reconocen por separado y el texto aparece como subtítulos.

**No necesita instalar Python, abrir una terminal ni conectarse a Internet.**

## Controles

| Acción | Control |
|---|---|
| Ayuda en pantalla | `?` |
| Separar palabras | Retirar ambas manos 1,4 segundos; también Espacio |
| Borrar una letra / deshacer | Retroceso / Ctrl+Z |
| Guardar texto / vaciar texto | Ctrl+S / Ctrl+U |
| Reconectar cámara | R |
| Herramientas de capturas, entrenamiento y evaluación | Tab |
| Borrado por movimiento **experimental** | G; desactivado al iniciar |
| Salir | Q o Esc |

Si ambas manos estabilizan letras distintas simultáneamente, se agrega primero la izquierda y después la derecha. Para repetir una letra, relaje o retire la mano entre emisiones.

En herramientas: K/L cambia la letra; H cambia únicamente la mano que se graba; C inicia/guarda una toma; V reserva las tomas para evaluación; P cambia participante; N inicia sesión; T entrena; X cancela entrenamiento; E evalúa. Las dos manos siguen reconociéndose.

## Requisitos y almacenamiento

Windows de 64 bits y cámara compatible con OpenCV. Esta distribución se comprueba en Windows 11 x64; Windows 10 x64 es un destino previsto, **sin prueba física en otro equipo**. Los tiempos de inicio y FPS dependen del procesador, la cámara y la iluminación. No se ha establecido un mínimo de RAM por medición. Mantenga espacio libre para extraer la distribución y guardar capturas/modelos nuevos; al primer inicio se copian los recursos de trabajo.

No se requieren permisos de administrador. `models`, `data`, `config` y `reports` se crean junto al ejecutable. Si esa ubicación no permite escribir, se usa `%LOCALAPPDATA%\SignLens`. La ubicación alternativa aparece en la interfaz cuando se utiliza; la ruta completa queda en el registro de inicio. Los reportes están en `reports/AAAA-MM-DD/` con nombres únicos; las capturas en `data` y los textos guardados en `texts`.

Si no hay cámara, la interfaz muestra un aviso y permite reintentar con R. Si falta un archivo, vuelva a extraer **la distribución completa**. El programa no descarga modelos al arrancar. Para comunicar un error, adjunte el último registro `application_*.log` de `reports`.

## Alcance de la demostración

Prototipo experimental de reconocimiento de **23 clases de letras estáticas** y composición de texto por deletreo. No es traducción continua de lengua de señas, ni reconocimiento de palabras completas o letras dinámicas. El borrado gestual conserva su carácter experimental. Los resultados históricos de entrenamiento no equivalen a exactitud garantizada con una persona nueva.

El código, las pruebas, los modelos y la evidencia de esta entrega están en **PROYECTO**. Consulte su README y el informe `delivery_verification_*.json` más reciente dentro de `PROYECTO/reports`.
