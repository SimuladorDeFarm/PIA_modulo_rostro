"""Carga / descubrimiento del dataset AffectNet.

El Líder Desarrollo pidió "cargar AffNet" pero sin hacer los splits: el split
(train/val/test) es asunto de entrenamiento, no de control de calidad. Por eso
este módulo solo enumera las imágenes presentes en disco, que es lo que necesita
el QC de integridad (tarea 1) para revisar archivo por archivo.
"""

from pathlib import Path
from typing import Iterator

from . import config


def descubrir_imagenes(
    dataset_root: Path = config.DATASET_ROOT,
    image_dirs: list[str] = config.IMAGE_DIRS,
) -> Iterator[Path]:
    """Recorre el dataset y entrega la ruta de cada archivo de imagen.

    Recorre recursivamente las subcarpetas indicadas en `image_dirs`
    (p. ej. Train/ y Test/) y entrega cualquier archivo cuya extensión esté en
    `config.IMAGE_EXTENSIONS`. No abre ni valida las imágenes: de eso se encarga
    el módulo de integridad. Tampoco lee labels.csv ni arma splits.

    Yields:
        Path absoluto de cada archivo de imagen encontrado, en orden estable.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset AffNet en: {dataset_root}. "
            "Revisa DATASET_ROOT en src/config.py."
        )

    for nombre_dir in image_dirs:
        carpeta = dataset_root / nombre_dir
        if not carpeta.exists():
            # Una subcarpeta faltante no es fatal: puede que solo exista Train.
            continue
        for ruta in sorted(carpeta.rglob("*")):
            if ruta.is_file() and ruta.suffix.lower() in config.IMAGE_EXTENSIONS:
                yield ruta


def contar_imagenes(
    dataset_root: Path = config.DATASET_ROOT,
    image_dirs: list[str] = config.IMAGE_DIRS,
) -> int:
    """Cuenta cuántas imágenes hay en el dataset (sin abrirlas)."""
    return sum(1 for _ in descubrir_imagenes(dataset_root, image_dirs))
