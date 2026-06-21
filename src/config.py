"""Configuración central del pipeline de QC del Módulo de Rostro.

Todas las rutas son locales (ver CLAUDE.md, sección 5.4): el QC de rostro NO se
ejecuta en Colab. Si se mueve la carpeta del dataset, basta con ajustar
`DATASET_ROOT` aquí y el resto del pipeline lo hereda.
"""

from pathlib import Path

# Raíz del proyecto (carpeta que contiene main.py y src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raíz del dataset AffectNet reducido (v1: 30.000 imágenes, 96x96).
DATASET_ROOT = PROJECT_ROOT / "AffNet"

# Subcarpetas donde están físicamente las imágenes. El QC de integridad revisa
# los archivos en disco directamente (no depende de labels.csv), porque su
# objetivo es justamente detectar archivos rotos antes de cualquier otra etapa.
IMAGE_DIRS = ["Train", "Test"]

# Extensiones de imagen que el pipeline considera válidas. Cualquier otra cosa
# dentro de las carpetas de imágenes se reporta como descartada.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Dimensiones esperadas de las imágenes originales (v1). Cualquier imagen con
# un tamaño distinto se marca como dimensión anómala (tarea 1).
EXPECTED_SIZE = (96, 96)  # (ancho, alto)

# Carpeta raíz donde se guardan los reportes/logs de QC. Cada etapa escribe en
# su propia subcarpeta para no mezclar salidas.
REPORTES_DIR = PROJECT_ROOT / "reportes"

# Subcarpeta y archivo de reporte de la verificación de integridad (tarea 1).
# El .txt solo se genera cuando hay imágenes corruptas o con dimensión anómala.
INTEGRIDAD_DIR = REPORTES_DIR / "integridad"
REPORTE_INTEGRIDAD_TXT = INTEGRIDAD_DIR / "reporte_integridad.txt"

# Subcarpeta y archivo de reporte de la detección de duplicados (tarea 2).
# El .txt solo se genera cuando hay grupos de duplicados.
DUPLICADOS_DIR = REPORTES_DIR / "duplicados"
REPORTE_DUPLICADOS_TXT = DUPLICADOS_DIR / "reporte_duplicados.txt"

# Subcarpeta y archivo de reporte de luminancia/contraste (tarea 3).
# El .txt siempre se genera (es una etapa exploratoria): incluye la distribución
# global y la lista de imágenes que se salen de la norma.
LUMINANCIA_DIR = REPORTES_DIR / "luminancia"
REPORTE_LUMINANCIA_TXT = LUMINANCIA_DIR / "reporte_luminancia.txt"

# Umbrales de luminancia/contraste (tarea 3). Escala 0-255 sobre la imagen en
# escala de grises. Valores deliberadamente LAXOS: solo marcan casos extremos,
# no recortan el dataset agresivamente. Ajustables tras ver la distribución real.
LUMINANCIA_MIN = 35.0    # media por debajo -> subexpuesta (muy oscura)
LUMINANCIA_MAX = 220.0   # media por encima -> sobreexpuesta (muy clara)
CONTRASTE_MIN = 18.0     # desviación estándar por debajo -> sin contraste

# Reescalado (tarea 5). Resolución objetivo que leerá Py-Feat. Las imágenes
# sobrevivientes (en DATASET_ROOT) se redimensionan y se guardan en un dataset
# NUEVO y aparte, sin tocar el original. Se conserva la estructura de carpetas.
TARGET_SIZE = (128, 128)                      # (ancho, alto)
DATASET_V4_ROOT = PROJECT_ROOT / "AffNet_128"  # dataset reescalado (v4, local)

# Subcarpeta y archivo de reporte del reescalado (tarea 5).
REESCALADO_DIR = REPORTES_DIR / "reescalado"
REPORTE_REESCALADO_TXT = REESCALADO_DIR / "reporte_reescalado.txt"

# Py-Feat (tarea 6). Se corre sobre el dataset reescalado (v4) y se guarda un
# único CSV local (v5) con detección, confianza, pose, landmarks y AUs.
FEATURES_DIR = PROJECT_ROOT / "features"
PYFEAT_CSV = FEATURES_DIR / "pyfeat_v5.csv"
PYFEAT_DIR = REPORTES_DIR / "pyfeat"
REPORTE_PYFEAT_TXT = PYFEAT_DIR / "reporte_pyfeat.txt"

# Dispositivo: "auto" detecta una GPU NVIDIA (CUDA) y, si no hay, usa CPU.
# Lo normal es no tocar esto: el modo test (-t) detecta el hardware solo.
PYFEAT_DEVICE = "auto"
# Defaults usados solo si todavía no hay pyfeat_config.json (sin calibrar).
PYFEAT_BATCH_SIZE = 16     # tamaño de lote para la inferencia
PYFEAT_WORKERS = 4         # procesos en paralelo en CPU (cada uno con su detector)
# Tope de batch_size que el calibrador puede recomendar en GPU (cota de cordura).
PYFEAT_BATCH_MAX = 256

# Calibración automática (-t) -> escribe este archivo, que -p lee. Es específico
# de cada máquina (depende de su GPU/CPU/RAM), por eso va al .gitignore y no se
# versiona.
PYFEAT_CONFIG_JSON = PROJECT_ROOT / "pyfeat_config.json"
# Fracción de VRAM a usar como tope al recomendar batch_size (0.85 = 85%, deja
# un 15% de margen para no llenar la tarjeta).
PYFEAT_VRAM_MARGEN = 0.85
# Núcleos de CPU a dejar libres para que el sistema siga usable mientras corre.
PYFEAT_CPU_RESERVA = 2

# Filtrado del CSV (tareas 7, 8, 9): v5 -> v6. v5 es la salida cruda de Py-Feat
# (local o Colab). El archivo que tenemos vino de Colab y está en la raíz del
# proyecto. v6 es el CSV final que usa el Random Forest.
PYFEAT_CSV_V5 = PROJECT_ROOT / "pyfeat_features_v5.csv"
PYFEAT_CSV_V6 = FEATURES_DIR / "pyfeat_v6.csv"
FILTRADO_DIR = REPORTES_DIR / "filtrado"
REPORTE_FILTRADO_TXT = FILTRADO_DIR / "log_qc_consolidado.txt"

# Taxonomía unificada (7 emociones, CLAUDE.md §7). AffectNet trae además
# 'contempt' (desprecio), que NO está en la taxonomía del proyecto y se descarta.
TAXONOMIA_7 = ["neutral", "happy", "sad", "anger", "fear", "disgust", "surprise"]
CLASE_EXCLUIDA = "contempt"

# Umbrales de filtrado del CSV (tareas 7-8), confirmados con el Líder Desarrollo.
FACESCORE_MIN = 0.90      # tarea 7: confianza mínima de detección de rostro
POSE_MAX_GRADOS = 45.0    # tarea 7: |yaw|,|pitch|,|roll| máximo (rostro no girado)

# ----------------------------------------------------------------------------- #
# Random Forest (entrenamiento del clasificador de emociones)
# ----------------------------------------------------------------------------- #
# Lee el v6 (features/pyfeat_v6.csv). La etiqueta es la columna 'clase' (derivada
# de la carpeta de AffectNet). Decisión del Líder Desarrollo: NO se usa
# labels_split.csv; el split se hace aquí, 70/15/15 estratificado.
LABEL_COL = "clase"
# Features de entrada: los 20 AUs (se importan de src.qc.pyfeat) + FaceScore.
RF_FEATURE_EXTRA = ["FaceScore"]

# Split 70/15/15 (train/val/test), estratificado por clase y DETERMINISTA (semilla
# fija). Se guarda en disco y se reutiliza: así el test queda reservado y NUNCA se
# toca hasta la evaluación final. Para rehacerlo hay que pedirlo explícitamente.
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train, val, test
SPLIT_SEED = 42
SPLIT_CSV = FEATURES_DIR / "split_train_val_test.csv"

# Artefactos del modelo (pesados / derivados -> van al .gitignore).
MODELO_DIR = PROJECT_ROOT / "models"
MODELO_RF = MODELO_DIR / "random_forest.joblib"          # modelo final (inferencia)
MODELO_META = MODELO_DIR / "random_forest_meta.json"     # features, clases, métricas
MODELO_CHECKPOINTS_DIR = MODELO_DIR / "checkpoints"      # checkpoints durante el fit
REPORTE_MODELO_DIR = REPORTES_DIR / "modelo"
REPORTE_ENTRENAMIENTO_TXT = REPORTE_MODELO_DIR / "reporte_entrenamiento.txt"
REPORTE_TEST_TXT = REPORTE_MODELO_DIR / "reporte_test_final.txt"

# Hiperparámetros de REFERENCIA del Random Forest. Son los valores a los que
# `--reset-config` restaura el archivo editable de entrenamiento.
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None            # None = sin límite
RF_MIN_SAMPLES_LEAF = 1
RF_MIN_SAMPLES_SPLIT = 2
RF_MAX_FEATURES = "sqrt"
RF_CRITERION = "gini"          # gini | entropy | log_loss
RF_BOOTSTRAP = True
RF_CLASS_WEIGHT = "balanced"   # compensa el desbalance de clases
RF_CHECKPOINT_CADA = 50        # guardar un checkpoint cada N árboles

# Archivo de texto editable para tunear el entrenamiento sin tocar código. Lo lee
# `-rf`; `--reset-config` lo restaura a los valores de referencia de arriba. Es
# de trabajo (cada experimento), por eso va al .gitignore.
ENTRENAMIENTO_CONFIG_TXT = PROJECT_ROOT / "config_entrenamiento.txt"
