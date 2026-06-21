"""Tareas 7, 8 y 9 del QC: filtrado del CSV de Py-Feat (v5 -> v6).

Lee el CSV crudo de Py-Feat (v5) y produce el CSV final de entrenamiento (v6)
aplicando, en orden:

- **Quitar la clase excluida** (`contempt`): AffectNet trae 8 categorías, pero la
  taxonomía del proyecto son 7 (CLAUDE.md §7); `contempt` no está y se descarta.
- **Dedup multi-rostro**: Py-Feat puede devolver varias filas para una misma
  imagen si detecta más de un rostro. Se conserva una fila por imagen, la de
  mayor `FaceScore`.
- **Tarea 7 (filtros de detección)**: descartar rostros con confianza bajo el
  umbral (`FaceScore`) o con pose extrema (yaw/pitch/roll fuera de rango).
- **Tarea 8 (calidad de extracción)**: descartar filas con landmarks `NaN` o con
  todos los AUs degenerados (todos en 0 o todos saturados en 1).
- **Tarea 9 (log consolidado)**: cuántas descartó cada filtro, la distribución de
  clases que queda, y se conserva `FaceScore` como feature para el Random Forest.

Todo es tabular: no toca imágenes, corre en CPU. La pose de Py-Feat viene en
RADIANES, por eso el umbral en grados se convierte a radianes al comparar.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .. import config
from .pyfeat import AU_COLUMNS, LANDMARK_COLUMNS


@dataclass
class ResultadoFiltrado:
    """Conteos de cada etapa del filtrado (para el log consolidado, tarea 9)."""

    total_inicial: int = 0
    desc_contempt: int = 0
    desc_multirostro: int = 0
    desc_facescore: int = 0
    desc_pose: int = 0
    desc_landmark_nan: int = 0
    desc_au_degenerado: int = 0
    total_final: int = 0
    distribucion: dict = field(default_factory=dict)
    csv_v6: Path = config.PYFEAT_CSV_V6


def derivar_clase(df):
    """Deriva la etiqueta de emoción del segundo componente de la ruta.

    Las rutas vienen como 'Split/Clase/archivo.jpg' y la clase aparece con
    distinta capitalización según el split (Test usa 'Anger', Train 'anger'),
    por eso se normaliza a minúsculas.
    """
    return df["imagen"].astype(str).str.split("/").str[1].str.lower()


def filtrar(
    csv_v5: Path = config.PYFEAT_CSV_V5,
    csv_v6: Path = config.PYFEAT_CSV_V6,
    facescore_min: float = config.FACESCORE_MIN,
    pose_max_grados: float = config.POSE_MAX_GRADOS,
    escribir: bool = True,
) -> ResultadoFiltrado:
    """Aplica los filtros (tareas 7-8) y escribe el CSV v6. Devuelve los conteos.

    Args:
        escribir: si es False, calcula los conteos pero no escribe el CSV v6
            (modo exploratorio).
    """
    import numpy as np
    import pandas as pd

    # .copy() defragmenta el frame (166 columnas) antes de agregar 'clase',
    # evitando el PerformanceWarning de pandas.
    df = pd.read_csv(csv_v5).copy()
    res = ResultadoFiltrado(total_inicial=len(df), csv_v6=csv_v6)
    df["clase"] = derivar_clase(df)

    # 0) Quitar la clase excluida (contempt): no está en la taxonomía de 7.
    antes = len(df)
    df = df[df["clase"] != config.CLASE_EXCLUIDA].copy()
    res.desc_contempt = antes - len(df)

    # 1) Dedup multi-rostro: una fila por imagen, la de mayor FaceScore.
    antes = len(df)
    df = (df.sort_values("FaceScore", ascending=False)
            .drop_duplicates("imagen", keep="first"))
    res.desc_multirostro = antes - len(df)

    # 2) TAREA 7: confianza de detección bajo el umbral (incluye FaceScore=0,
    #    que son detecciones fallidas).
    antes = len(df)
    df = df[df["FaceScore"] >= facescore_min].copy()
    res.desc_facescore = antes - len(df)

    # 3) TAREA 7: pose extrema. Py-Feat entrega los ángulos en radianes.
    pose_max_rad = np.deg2rad(pose_max_grados)
    antes = len(df)
    pose_ok = (
        (df["Yaw"].abs() <= pose_max_rad)
        & (df["Pitch"].abs() <= pose_max_rad)
        & (df["Roll"].abs() <= pose_max_rad)
    )
    df = df[pose_ok].copy()
    res.desc_pose = antes - len(df)

    # 4) TAREA 8: landmarks con coordenadas NaN (extracción fallida).
    lm = [c for c in LANDMARK_COLUMNS if c in df.columns]
    antes = len(df)
    df = df[~df[lm].isna().any(axis=1)].copy()
    res.desc_landmark_nan = antes - len(df)

    # 5) TAREA 8: AUs degenerados (todos en 0 o todos saturados en 1).
    au = [c for c in AU_COLUMNS if c in df.columns]
    antes = len(df)
    degenerado = (df[au] == 0).all(axis=1) | (df[au] == 1).all(axis=1)
    df = df[~degenerado].copy()
    res.desc_au_degenerado = antes - len(df)

    res.total_final = len(df)
    res.distribucion = df["clase"].value_counts().to_dict()

    if escribir:
        csv_v6.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_v6, index=False)

    return res


def guardar_log(
    res: ResultadoFiltrado,
    facescore_min: float = config.FACESCORE_MIN,
    pose_max_grados: float = config.POSE_MAX_GRADOS,
    salida: Path = config.REPORTE_FILTRADO_TXT,
) -> Path:
    """Escribe el log consolidado de QC (tarea 9)."""
    salida.parent.mkdir(parents=True, exist_ok=True)
    total_desc = res.total_inicial - res.total_final
    with salida.open("w", encoding="utf-8") as f:
        f.write("Log consolidado de QC - Modulo de Rostro (tareas 7, 8 y 9)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Umbrales aplicados:\n")
        f.write(f"  FaceScore minimo:   {facescore_min}\n")
        f.write(f"  Pose maxima:        {pose_max_grados} grados (|yaw|,|pitch|,|roll|)\n")
        f.write(f"  Clase excluida:     {config.CLASE_EXCLUIDA}\n\n")
        f.write("Descartes por filtro:\n")
        f.write(f"  Total inicial (v5):              {res.total_inicial}\n")
        f.write(f"  - Clase excluida (contempt):     {res.desc_contempt}\n")
        f.write(f"  - Multi-rostro (1 fila/imagen):  {res.desc_multirostro}\n")
        f.write(f"  - T7 FaceScore bajo umbral:      {res.desc_facescore}\n")
        f.write(f"  - T7 Pose extrema:               {res.desc_pose}\n")
        f.write(f"  - T8 Landmarks NaN:              {res.desc_landmark_nan}\n")
        f.write(f"  - T8 AUs degenerados:            {res.desc_au_degenerado}\n")
        f.write(f"  Total descartado:                {total_desc}\n")
        f.write(f"  Total final (v6):                {res.total_final}\n\n")
        f.write("Distribucion de clases final (v6):\n")
        for clase in config.TAXONOMIA_7:
            n = res.distribucion.get(clase, 0)
            pct = 100 * n / res.total_final if res.total_final else 0.0
            f.write(f"  {clase:10s} {n:6d}  ({pct:4.1f}%)\n")
        f.write("\nFeatures para el Random Forest: los 20 AUs + FaceScore "
                "(CLAUDE.md tarea 9).\n")
    return salida
