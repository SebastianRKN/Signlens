# SignLens

Prototipo local de visión por computadora para reconocer letras estáticas con una cámara y convertir el deletreo en subtítulos. Busca explorar una interfaz accesible de entrada de texto mediante señas. No constituye un intérprete ni un traductor completo de lengua de señas ecuatoriana.

## Entrega

El jurado abre `../EJECUTABLE_WINDOWS/SignLens.exe`; no necesita Python ni Internet. Extraer toda la distribución. Ver `../instrucciones.md`.

## Arquitectura y capacidades

Cámara OpenCV → landmarks de manos y pose MediaPipe → características geométricas → clasificador scikit-learn → estabilización independiente de ambas manos → composición de subtítulos. Se conservan el modelo original, umbrales, funciones de características y lógica de reconocimiento del ZIP recibido.

- Seguimiento de manos/brazos y reconocimiento simultáneo de ambas manos.
- 23 clases del modelo: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, W, X, Y.
- Texto automático, separación de palabras, edición y guardado.
- Captura de landmarks; separación entre entrenamiento y evaluación reservada.
- Entrenamiento cancelable en otro proceso y publicación atómica del modelo verificado.
- Diagnóstico local, evaluación, medición de cámara y recuperación de errores.
- Borrado gestual experimental, desactivado por defecto; G lo habilita.

Los FPS de captura y procesamiento son medidas diferentes. Las métricas históricas de validación cruzada dependen de las sesiones del dataset; no certifican precisión con nuevos participantes. No existen tomas reservadas independientes en el ZIP original. No se agrega reconocimiento temporal, síntesis de voz ni otras funciones de versiones posteriores al ZIP fuente.

## Organización

| Ruta | Contenido |
|---|---|
| `src/signlens` | Aplicación, seguimiento, composición, entrenamiento, rutas y reportes |
| `models` | Modelos .task/.joblib originales y landmarks de entrenamiento |
| `data` | Capturas originales y respaldo de metadatos |
| `tests` | Pruebas originales y regresiones de portabilidad |
| `reports/AAAA-MM-DD` | Reportes, evaluaciones, matrices, logs y evidencia fechada |
| `reports/sin_fecha` | Historial sin fecha comprobable |
| `experiments/legacy_v1` | Método anterior conservado para reproducibilidad; aislado del modelo activo |
| `THIRD_PARTY_LICENSES` | Licencias extraídas de dependencias y su inventario |
| `SignLens.spec`, `build_windows.py` | Compilación reproducible para Windows |

`reporting.py` centraliza los reportes con fecha de Ecuador (UTC-5), hora y sufijo único. Los resultados de una corrida de entrenamiento comparten una subcarpeta única. `report_manifest.json` vincula cada nuevo modelo con su evaluación sin duplicarla en models. La migración histórica y sus criterios están en `reports/2026-10-23/migration_original.json`. Las rutas personales de los informes distribuidos se anonimizan; no son rutas utilizadas por la aplicación.

## Desarrollo desde código

Entorno comprobado: Python 3.12 x64 en Windows. Desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe run_local.py
```

El primer comando requiere que `python` sea realmente Python 3.12 x64. Si no lo es, utilice la ruta completa al intérprete. Nunca se presupone `py -3.12`. No continúe si una etapa falla. `INSTALAR_DEPENDENCIAS.cmd` ofrece instalación con detección; `BUILD_WINDOWS.ps1 -Python "ruta a python.exe" -InstallOnly` permite indicarlo explícitamente.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_local.py --self-check
.\.venv\Scripts\python.exe run_local.py --tracker-check
```

El ejecutable también acepta `--self-check`, `--tracker-check`, `--train`, `--evaluate`, `--extract`, `--probe-camera`, `--benchmark SEGUNDOS`, `--camera INDICE` y `--duration SEGUNDOS`. Los tres primeros comandos de procesamiento aceptan `--help` para sus argumentos. El entrenamiento habitual se inicia con T dentro de la vista de datos; conserva el modelo anterior hasta publicar correctamente el nuevo.

## Compilar

```powershell
.\BUILD_WINDOWS.ps1 -Python "ruta completa a Python 3.12 x64\python.exe"
```

Crea un entorno aislado fuera del proyecto, instala versiones fijadas, ejecuta pruebas, compila onedir y comprueba el ejecutable. Detiene la entrega ante errores y conserva el registro fechado. El ZIP generado lleva fecha/hora para evitar sobrescribir una entrega previa. Necesita Internet para instalar dependencias de compilación; el programa terminado funciona sin descargar modelos.

La distribución contiene el runtime de Python y bibliotecas nativas. Se mantiene el modo consola internamente para recibir la salida del proceso de entrenamiento, pero la ventana de consola propia se oculta al abrir la interfaz. Los jueces usan doble clic. Referencias técnicas: https://www.pyinstaller.org/en/stable/runtime-information.html y https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html.

## Datos y límites

Los recursos incluidos en `_internal/seed` son de lectura. Al iniciar se crean copias de trabajo modificables sin reemplazar datos previos. La caché de cámara del autor no se distribuye: se genera en `config`. La selección automática intenta índices 0–4; `--camera` permite otros índices. La cámara debe estar autorizada y libre.

Fuente exclusiva: `entrega-extraida-oct23.zip`. Las pruebas de empaquetado no equivalen a validación lingüística ni evaluación en vivo con nuevos usuarios. Consulte el informe final de verificaciones para distinguir PASS y NOT RUN. Se conservan `DATASET_LICENSE.txt` y los avisos de dependencias; las imágenes originales del dataset externo no forman parte de este ZIP.
