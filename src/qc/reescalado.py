"""Tarea 5 del QC: reescalado de las imágenes sobrevivientes a 128x128.

Toma las imágenes que pasaron los filtros anteriores (las que quedan en
`DATASET_ROOT`) y las redimensiona a la resolución objetivo (`TARGET_SIZE`,
128x128), guardándolas en un dataset NUEVO y aparte (`DATASET_V4_ROOT`), sin
tocar el original. Conserva la estructura de carpetas (Train/clase/..., etc.)
para que Py-Feat (tarea 6) lea siempre desde el mismo lugar.

A diferencia del módulo de Voz (que guardaba en Drive), esto es 100% local
(CLAUDE.md, sección 5.4).

El reescalado se paraleliza con un pool de hilos. Se usan automáticamente todos
los hilos disponibles del sistema. Funciona bien con hilos porque Pillow libera
el GIL durante decodificar/redimensionar/codificar, que es donde está el costo.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from .. import config

# Filtro de remuestreo: LANCZOS da el mejor resultado al ampliar 96 -> 128.
_RESAMPLING = Image.Resampling.LANCZOS


@dataclass
class ResultadoReescalado:
    """Resumen de una corrida de reescalado."""

    ok: int                              # imágenes reescaladas con éxito
    errores: list[tuple[Path, str]]      # (ruta_origen, motivo) de las que fallaron
    destino: Path                        # raíz del dataset generado
    hilos: int                           # hilos usados


def hilos_disponibles() -> int:
    """Cantidad de hilos a usar: todos los que reporte el sistema (mínimo 1)."""
    return os.cpu_count() or 1


def _reescalar_una(
    ruta_origen: Path,
    raiz_origen: Path,
    raiz_destino: Path,
    size: tuple[int, int],
) -> tuple[Path, str] | None:
    """Reescala una imagen y la guarda en el destino. None si OK, (ruta, error) si falla.

    Recrea la ruta relativa del origen bajo `raiz_destino`, de modo que la
    estructura de carpetas (Train/clase/archivo.jpg) se mantiene idéntica.
    """
    try:
        rel = ruta_origen.relative_to(raiz_origen)
        destino = raiz_destino / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(ruta_origen) as im:
            im = im.convert("RGB")  # normaliza modo para guardar como JPEG
            im_red = im.resize(size, _RESAMPLING)
            im_red.save(destino, format="JPEG", quality=95)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        return (ruta_origen, f"{type(e).__name__}: {e}")
    return None


def reescalar_dataset(
    imagenes: Iterable[Path],
    raiz_origen: Path = config.DATASET_ROOT,
    raiz_destino: Path = config.DATASET_V4_ROOT,
    size: tuple[int, int] = config.TARGET_SIZE,
    hilos: int | None = None,
) -> ResultadoReescalado:
    """Reescala todas las imágenes en paralelo y devuelve el resultado.

    Args:
        imagenes: rutas de las imágenes a reescalar (sobrevivientes del QC).
        raiz_origen: raíz desde la que se calcula la ruta relativa de cada imagen.
        raiz_destino: raíz del dataset nuevo donde se guardan las reescaladas.
        size: resolución objetivo (ancho, alto).
        hilos: nº de hilos; por defecto, todos los disponibles del sistema.
    """
    imagenes = list(imagenes)
    n_hilos = hilos if hilos is not None else hilos_disponibles()
    raiz_destino.mkdir(parents=True, exist_ok=True)

    errores: list[tuple[Path, str]] = []
    ok = 0
    with ThreadPoolExecutor(max_workers=n_hilos) as pool:
        for resultado in pool.map(
            lambda r: _reescalar_una(r, raiz_origen, raiz_destino, size),
            imagenes,
        ):
            if resultado is None:
                ok += 1
            else:
                errores.append(resultado)

    return ResultadoReescalado(ok=ok, errores=errores, destino=raiz_destino, hilos=n_hilos)


def guardar_reporte(
    resultado: ResultadoReescalado,
    total_origen: int,
    size: tuple[int, int] = config.TARGET_SIZE,
    salida: Path = config.REPORTE_REESCALADO_TXT,
) -> Path:
    """Escribe el reporte .txt del reescalado (siempre, deje o no errores)."""
    salida.parent.mkdir(parents=True, exist_ok=True)
    try:
        destino_rel = resultado.destino.relative_to(config.PROJECT_ROOT)
    except ValueError:
        destino_rel = resultado.destino

    with salida.open("w", encoding="utf-8") as f:
        f.write("Reporte de reescalado - Modulo de Rostro (QC tarea 5)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Resolucion objetivo:   {size[0]}x{size[1]}\n")
        f.write(f"Dataset destino:       {destino_rel}\n")
        f.write(f"Hilos usados:          {resultado.hilos}\n\n")
        f.write(f"Imagenes de origen:    {total_origen}\n")
        f.write(f"Reescaladas OK:        {resultado.ok}\n")
        f.write(f"Con error:             {len(resultado.errores)}\n\n")
        if resultado.errores:
            f.write("Errores (ruta_origen | motivo):\n")
            f.write("-" * 70 + "\n")
            for ruta, motivo in resultado.errores:
                try:
                    ruta_rel = ruta.relative_to(config.PROJECT_ROOT)
                except ValueError:
                    ruta_rel = ruta
                f.write(f"{ruta_rel} | {motivo}\n")
        else:
            f.write("Sin errores: todas las imagenes se reescalaron correctamente.\n")
    return salida
