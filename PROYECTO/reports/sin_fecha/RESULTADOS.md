# Signlens: entrenamiento experimental con el dataset proporcionado

## Reentrenamiento con capturas de cámara — 18 de octubre de 2026

Se añadieron 1.712 muestras de webcam de una persona para 21 letras, a las 832 muestras del dataset. El total es 2.544 muestras, 473 grupos de fotos, 17 sesiones y dos fuentes (`dataset` y `me`). Las 53 capturas se hicieron durante una sola ejecución de la app; se corrigió su metadato para que todas compartan un único identificador de sesión. El ZIP `data/capture_metadata_backup_oct18.zip` conserva los metadatos originales; los arrays de landmarks y etiquetas no se modificaron.

| Evaluación class-session, 3 semillas | Exactitud | Macro-F1 |
|---|---:|---:|
| Baseline reevaluado sobre el conjunto combinado | 0,601 ± 0,003 | 0,526 |
| Ensamble seleccionado y guardado | **0,704 ± 0,005** | **0,627** |

El umbral de abstención quedó en 0,56: responde en el 86 % de las muestras fuera de fold y acierta el 75 % de las aceptadas. Un modelo ajustado solo con el dataset obtuvo 83,1 % al probarse en estas capturas. La evaluación optimista agrupada por foto dio 90,0 % y no debe citarse como generalización; la evaluación agrupada por sesión dio 75,5 %.

Límites: las nuevas tomas son de una sola persona, cámara y sesión, y contienen fotogramas correlacionados. No prueban rendimiento con firmantes nuevos. Las clases más débiles siguen siendo Q (0,03), X (0,04), U (0,20), Y (0,38), R (0,41) e I (0,43); no conviene afirmar que esas letras ya son fiables. El modelo continúa siendo de formas estáticas, no un traductor de LSEC.

Artefactos actualizados: `models/sign_classifier.joblib`, `reports/2026-10-18/evaluation.json`, matrices de confusión y `reports/sin_fecha/training.log`. La fecha de entrenamiento registrada es 2026-10-18 17:10.

## Abrir
Ejecutar ABRIR_APP.cmd en este equipo. El lanzador usa el entorno Python instalado en work/signlens/signlens/.venv y carga models/sign_classifier.joblib.
Mano inicial: Right. H alterna Left/Right; Q o Esc cierra. ? significa que no hay una predicción aceptada. La confianza mostrada es la salida del clasificador, no una garantía de corrección.
Para otra instalación: crear un entorno Python 3.12, instalar requirements-tested.txt y ejecutar python run_local.py.

## Alcance
El modelo actualmente guardado es un ensamble (SVM + MLP + regresión logística) de 23 etiquetas de letras: A B C D E F G H I K L M N O P Q R S T U W X Y. Se reentrenó el 18 de octubre con 2.544 muestras, 473 grupos de fotos y 17 sesiones; los 832 ejemplos, 420 grupos y 16 sesiones corresponden al modelo inicial anterior a las capturas de webcam.
Clasifica la apariencia de un fotograma. No aprende movimientos, palabras, frases ni el alfabeto completo.
Se excluyeron números, MI, NOMBRE, HOLA-5 y Te amo del modelo de letras. Existe un modelo experimental con las 33 clases (`models/sign_classifier_all.joblib`, ver abajo).
El publicador identifica el conjunto como ecuatoriano; no se ha validado lingüísticamente con personas usuarias de LSEC.

## Datos
- ZIP: ecuadorian sign language.v3i.yolov8.zip, versión 3, CC BY 4.0. Fuente: https://universe.roboflow.com/ecuadorian-language-sign/ecuadorian-sign-language/dataset/3
- 33 clases; manos extraídas en 1.227 anotaciones, 452 fallaron (resultados solo sobre las extraídas).
- Hay ~3 copias aumentadas por foto original, y las fotos originales provienen de pocas sesiones de captura (se infieren de la fecha/hora o contador en el nombre del archivo: 16 sesiones para letras, 27 en total). No hay identificadores de persona.

## Evaluación inicial del 9 de octubre (referencia histórica)
Las cifras de esta sección describen el modelo entrenado solo con el dataset antes de incorporar las nuevas capturas. Los resultados actuales están al inicio de este archivo.
El split original de Roboflow para letras tenía 11 imágenes de prueba: demasiado poco. La evaluación nueva (`signlens.train`) es **class-session holdout**: cada clase se prueba solo en sesiones que no vio al entrenar; el aumento de datos se aplica solo al entrenamiento. 3 semillas.

| Modelo (letras, 23 clases) | Exactitud | macro-F1 |
|---|---|---|
| Anterior: forma orientada + SVM C=10, sin aumento | 0,403 ± 0,005 | 0,344 |
| Nuevo: orientada + 42 descriptores + aumento, SVM | 0,519 ± 0,025 | 0,429 |
| Nuevo: ensamble (seleccionado) | **0,532 ± 0,012** | **0,449** |

- Referencia optimista (agrupada por foto, vecinos de la misma sesión a ambos lados): 0,875. No citar como rendimiento real.
- Referencia estricta (agrupada por sesión): 0,531.
- Abstención: con umbral 0,84 el modelo responde en el 39 % de los fotogramas y acierta el 76 % de ellos (objetivo 75 %, calibrado con predicciones fuera de muestra). Con umbral 0,9: 28 % de cobertura, 79 % de acierto.
- Mejores letras: G (0,87), C (0,85), L (0,81), S (0,79). Más débiles: F, K, P (0,00), M, Q, N (≈0,02-0,05).
- Confusiones principales: U→R, N↔M, K→R, F→Y/I. Son letras que se distinguen por detalles finos del pulgar o la orientación; 22 fotos por letra de pocas sesiones no bastan para aprenderlas.
- 33 clases (SVM, 27 sesiones, 2 semillas): exactitud 0,550, macro-F1 0,454, umbral 0,73 (43 % de cobertura, 75 % de acierto). La clase MI aparece en una sola sesión y no se puede evaluar.
- Costo de inferencia del clasificador: ≈1,5 ms por fotograma con el ensamble (medido en el entorno de prueba, no en tu PC); no es el cuello de botella de los FPS.

## Límites
- Sin identificadores de persona: las sesiones son un sustituto, no una prueba con personas nuevas. Con tus propias capturas en otras sesiones la cifra será más realista.
- La curva de aprendizaje dentro de este dataset es plana: más fotos del mismo tipo no ayudan; falta variedad (personas, fondos, luz).
- Sin conjunto de no-señas: no se mide cuántas falsas aceptaciones hay con gestos desconocidos.
- NOT RUN: prueba en cámara real, con cámara de 50+ FPS, ni personas nuevas.

## Cambios en esta versión
- Cámara: sondeo de FPS reales por backend (DirectShow/Media Foundation en Windows) y modo, con caché; informa FPS de la cámara, del pipeline y frames descartados. `--probe-camera` imprime la tabla.
- Tracker: sin mano en pantalla, el detector de palma corre cada 2 frames; pose asíncrona de más de 250 ms se descarta.
- Manos: la identidad al cruzar las manos usa la velocidad extrapolada de cada muñeca (test de cruce con etiquetas de MediaPipe intercambiadas).
- Muestreo: `splits.py` (sesiones y particiones sin fuga), extractor con 3 reintentos de recorte (ampliado y con aumento de resolución), `--collect` para grabar tus letras con signer/sesión.
- PASS: 38 pruebas automatizadas (incluye sondeo de cámara con un driver simulado y la ausencia de fugas en los splits).

## Archivos reproducibles
reports/2026-10-18/evaluation.json: métricas, resultados por clase y matriz de confusión.
reports/sin_fecha/dataset_audit.json y extraction.json: recuentos y pérdidas de extracción (versión anterior).
reports/sin_fecha/evaluation_previous.json: evaluación del modelo anterior.
models/landmarks.npz: puntos extraídos, etiquetas y procedencia (CC BY 4.0 del dataset).
`python -m signlens.train models/landmarks.npz --out models` reproduce el entrenamiento y la evaluación (escribe sign_classifier.joblib, evaluation.json y la matriz de confusión). experiments/legacy_v1/ conserva el script del entrenamiento anterior; el modelo previo está en models/sign_classifier_previous.joblib.
requirements-tested.txt fija las versiones del entorno utilizado.

## Siguiente mejora necesaria
Grabar tus propias capturas con `--collect` en varias sesiones y volver a entrenar con `--extra data/static`. Recolectar ejemplos LSEC validados, diversos y con identificadores de persona, incluyendo movimientos, dos manos, fondos variados y no-señas. Separar personas entre entrenamiento y evaluación. El modelo entregado es un prototipo experimental y no un traductor validado.

