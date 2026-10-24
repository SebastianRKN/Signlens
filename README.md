> Entrenamiento local integrado: ver RESULTADOS.md. Abrir con ABRIR_APP.cmd. Modelo experimental de letras, no traductor validado.

**Entrega actual, 23 de octubre:** consulta `LEEME_ENTREGA.md`. Borrado estable con Retroceso; el gesto es experimental y se activa expresamente con G. La suite actual tiene 79 pruebas aprobadas.

## Interfaz actual: subtítulos automáticos — 21 de octubre

Abre `ABRIR_APP.cmd` y empieza a hacer letras: **ambas manos escriben automáticamente**.
No hay botones, selector de mano de escritura ni activación manual. El texto se muestra abajo,
como subtítulos sobre la cámara. Las letras se ordenan por confirmación; en un empate del mismo
fotograma, izquierda y luego derecha. Mantener una seña no la repite continuamente.

Retira ambas manos **1,4 segundos** para separar palabras. Para repetir la misma letra con una mano,
retírala **0,6 segundos** y vuelve a mostrarla. Retroceso borra, Ctrl+Z deshace y Ctrl+S guarda el texto.
`?` abre la ayuda. `Tab` muestra las herramientas de datos, que ahora se manejan con teclado.
Consulta `SUBTITULOS_OCT21.md` para las reglas y el alcance, y `PLAN_LETRAS_CON_MOVIMIENTO.md` para el trabajo temporal pendiente.

## Dos manos simultáneas — 21 de octubre

La app reconoce ambas manos a la vez y muestra resultados separados para **IZQUIERDA** y **DERECHA**.
Cada mano puede mostrar una letra distinta y tiene su propia confirmación temporal. Al retirar una mano,
solo se limpia su resultado. No hace falta alternar con H para reconocerlas.

**H, dentro de DATOS**, selecciona únicamente qué mano se guarda durante una captura etiquetada. La otra
mano continúa reconociéndose. Una toma tiene una sola etiqueta: no se mezclan las muestras de las dos manos.
Durante una toma reservada se oculta solo el resultado de la mano grabada. Ambas usan por defecto las
23 letras y el umbral original del modelo. Ver `DOS_MANOS_OCT21.md`.

## Corrección de reconocimiento — 20 de octubre

Al iniciar se usa **COMPLETO**: las 23 letras y el umbral guardado en el modelo (actualmente 0,56).
Dentro de DATOS, `D` activa de forma opcional **DEMO LIMITADO (10)**, que aplica restricciones adicionales; vuelve a pulsar
`D` para regresar a COMPLETO. `V` elige si una captura se guarda para entrenar o para evaluar. Elegir
EVALUAR no oculta las predicciones mientras observas la cámara: solo se ocultan durante la toma reservada.
`H` cambia la mano seleccionada para grabar muestras; el reconocimiento procesa ambas manos.
Los controles actuales son atajos de teclado. Consulta `CORRECCION_RECONOCIMIENTO_OCT20.md` para las verificaciones históricas.

# signlens

Pipeline de tracking de manos y brazos en tiempo real, base para un traductor de lenguaje de señas.

## Uso
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m signlens                    # visualizar tracking
python -m signlens --label hola       # grabar dataset (ESPACIO inicia/detiene, q sale)
python -m signlens --benchmark 10     # medir FPS y latencia sin ventana
pytest
```
Los modelos de MediaPipe se descargan solos la primera vez a `~/.signlens/models`.

### Interfaz de captura y entrenamiento
Al abrir con `ABRIR_APP.cmd`, pulsa `Tab` para las herramientas de datos. Allí puedes cambiar la letra, iniciar/detener
la captura, entrenar el modelo y elegir qué mano grabar. Funcionan `L` (letra), `C` o `Espacio`
(captura), `T` (entrenar), `H` (mano a grabar) y `Q` (salir). La cámara permanece activa durante el entrenamiento.
Las capturas se guardan bajo `data/static`; al completar el entrenamiento se recarga el modelo y se deja
`training.log` junto al proyecto. El entrenamiento combina el dataset incluido con tus capturas y puede
tardar bastante cuando hay muchos ejemplos; deja abierta la app mientras corre y espera el mensaje final.
Graba cada letra en sesiones distintas y con variedad de luz y fondo. La evaluación mostrada sigue
siendo por sesiones del conjunto, no una validación independiente por persona.
Para preparar otro equipo con Windows, instala Python 3.12 y ejecuta `INSTALAR_DEPENDENCIAS.cmd` una vez.

## Clasificador de letras (formas estaticas)
```bash
pip install -e ".[ml]"
python -m signlens.dataset path/al/dataset_yolov8 -o models/landmarks.npz   # imagenes -> landmarks MediaPipe
python -m signlens.train models/landmarks.npz --out models                  # evaluacion por sesiones + modelo final
python -m signlens.train models/landmarks.npz --scope all --out models      # las 33 clases (experimental)
python run_local.py                                                         # reconocimiento en vivo
```
**Por que la evaluacion es por sesiones.** Las ~600 fotos originales vienen de solo ~17-27 sesiones de captura
(mismo lugar, misma persona, segundos de diferencia) y cada foto trae ~3 copias aumentadas. Separar por foto deja
vecinos casi identicos a ambos lados y da cifras enganosas (~87 %). `signlens.train` usa *class-session holdout*:
cada clase se prueba solo en sesiones que no vio al entrenar (`splits.py`, con test anti-fuga). Es la cifra a citar.

**Mejoras de modelo.** (1) 42 descriptores invariantes (angulos de flexion, distancias entre puntas, apertura entre
dedos) anadidos a las coordenadas orientadas; (2) aumento solo en entrenamiento: espejo, rotaciones 3D pequenas y
ruido de landmarks; (3) ensamble SVM+MLP+LogReg; (4) umbral de abstencion calibrado con predicciones fuera de muestra.
El modelo actual se reentreno con tus capturas el 18 de octubre: 70 % de exactitud y macro-F1 0,627 en class-session
holdout sobre el conjunto combinado. La evaluación no demuestra generalización a personas nuevas. Ver `RESULTADOS.md`.

**Anade tus propias capturas** (lo que mas mejora el modelo):
```bash
python run_local.py --collect A --signer tunombre      # ESPACIO inicia/detiene, H cambia de mano, Q sale
python -m signlens.train models/landmarks.npz --extra data/static --out models
```
Graba cada letra en varias sesiones (otro dia, luz y fondo); el reporte incluye la transferencia
"dataset -> tus capturas".

Dataset: ver `DATASET_LICENSE.txt` (CC BY 4.0, Roboflow Universe).

## Camara y FPS
```bash
python -m signlens --probe-camera     # tabla con los FPS REALES de cada backend/modo
python -m signlens --benchmark 10     # FPS de la camara vs. del pipeline, frames descartados, latencia p95
```
Los drivers informan lo que se les pidio, no lo que entregan: en Windows el backend por defecto (Media Foundation)
suele ignorar MJPG/60 FPS y corre a 30. `capture.py` mide cada backend/modo, elige el mas rapido y guarda el
resultado en cache. Sin mano a la vista, el detector de palma corre cada 2 frames; la pose asincrona vieja (>250 ms)
se descarta.

## Arquitectura
```
Camera (hilo, MJPG 60fps) -> Tracker [manos: hilo principal | pose: hilo async] -> FrameResult -> features.extract -> vector (147,) -> [Recorder | futuro modelo]
                    |                                                  
              OneEuroFilter                       viz.draw (solo render)
```
| Módulo | Responsabilidad |
|---|---|
| `capture` | Hilo de lectura + sondeo de FPS reales por backend/modo con caché |
| `tracker` | Único módulo que conoce MediaPipe: 2 manos (21 pts + landmarks 3D métricos) + pose (brazos) |
| `hands` | Identidad izq/der anclada al brazo, rechazo de outliers, retención en pérdidas cortas, suavizado (numpy puro, testeado) |
| `smoothing` | One Euro Filter: sin jitter en reposo, sin lag al mover rápido |
| `features` | Vector (147) con forma de la mano en 3D métrico + ubicación respecto al cuerpo |
| `splits` | Sesiones de captura y particiones sin fuga para evaluar |
| `train` | Entrenamiento y evaluación honesta de letras estáticas |
| `recorder` | Dataset `.npz` con `x (T, 147)` y `feature_version` etiquetado, listo para entrenar |
| `types`/`config` | Contratos de datos y configuración inmutable |

Como solo `tracker.py` importa MediaPipe, puedes cambiar de backend sin tocar el resto.

## Roadmap
1. **Dataset**: grabar 50+ repeticiones por seña, con varias personas e iluminaciones.
2. **Modelo**: ventana deslizante de ~30 frames -> clasificador (BiGRU o Transformer pequeño) con PyTorch.
3. **Inferencia**: exportar a ONNX, decodificación con umbral de confianza y estabilización temporal.
4. **Gramática**: añadir landmarks faciales (las expresiones son parte de la seña) y un modelo de secuencia a texto.
5. **Despliegue**: API con FastAPI/WebSocket o versión web con MediaPipe JS.
6. **Evaluación**: accuracy por seña, matriz de confusión, latencia p95 y FPS reportados en el README.

## Decisiones de diseño
- Vector de features normalizado respecto a la muñeca/hombros: el modelo aprende la forma y movimiento de la seña, no dónde estás parado.
- Lados izquierda/derecha por posición en pantalla (`skeleton.py`), no por la etiqueta del modelo: no depende del espejo.
- Rendimiento: pose ligero en hilo propio, inferencia a 640px, captura MJPG. El FPS no puede superar el de la cámara.
- Tracking de manos robusto (`hands.py`): la etiqueta Left/Right de MediaPipe es solo un prior suave; manda la posición de la muñeca del brazo. Saltos o cambios de tamaño imposibles se rechazan y las pérdidas de hasta 4 frames se retienen (se dibujan en gris).
- Forma de la mano con landmarks mundiales (metros), no con coordenadas de imagen: invariante a distancia y tamaño de mano, conserva la orientación de la palma.
- Ubicación de la mano respecto a los hombros: en lenguaje de señas *dónde* se hace la seña es parte del significado.
- Flags de presencia: distingue "mano ausente" de "mano en el origen".
- Etiquetas izquierda/derecha según el usuario (frame espejo; ver nota en `features.py`).
