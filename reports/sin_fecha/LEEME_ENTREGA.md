# SignLens — entrega del 23 de octubre

## Instalar y abrir

1. Extraer todo el ZIP en una carpeta local.
2. Instalar Python 3.12 para Windows. La primera instalación de dependencias requiere internet.
3. Ejecutar INSTALAR_DEPENDENCIAS.cmd una vez y esperar el mensaje de instalación completada.
4. Ejecutar VERIFICAR_APP.cmd y después ABRIR_APP.cmd. Cerrar otras aplicaciones que usen la cámara.

No es un ejecutable autónomo. Se incluyen el código, modelos y datos locales; no se incluye el entorno virtual de este equipo.

## Demostración de dos minutos

- Presentar: «SignLens es un prototipo que reconoce letras estáticas de ambas manos y las une en subtítulos».
- Mostrar el esqueleto y una letra con cada mano. Ambas aportan letras; el empate se ordena izquierda y derecha.
- Deletrear una palabra corta con movimientos pausados. Mantener cada letra hasta que aparezca.
- Retirar ambas manos 1,4 s para separar palabras. Para repetir una letra, retirar esa mano 0,6 s.
- Retroceso borra el último carácter. Ctrl+Z deshace; Ctrl+S guarda el texto.
- ? muestra la ayuda y Tab abre herramientas de datos. No hay botones en la vista principal.

## Borrado por gesto

Está desactivado al iniciar. G activa o desactiva el experimento. La reproducción del video proporcionado no reconoció los dos empujes por pérdidas de seguimiento. No presentar esta función como validada. Retroceso es el control estable de la entrega.

## Qué está comprobado

79 pruebas automatizadas pasaron con esta versión, incluyendo el compositor automático de ambas manos y las correcciones de texto. El replay del gesto tiene resultado FAIL documentado. Estas pruebas no representan un porcentaje de precisión de señas con personas nuevas.

La validación física de una palabra con el usuario se registra por separado. El modelo conserva 23 clases y el umbral 0,56. No se entrenó un modelo nuevo ni se implementó reconocimiento de letras dinámicas o traducción de gramática LSEC.

Los informes anteriores conservan fechas y describen versiones anteriores. Este archivo tiene prioridad como guía de uso de la entrega actual.
