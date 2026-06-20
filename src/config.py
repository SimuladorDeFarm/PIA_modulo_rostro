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
