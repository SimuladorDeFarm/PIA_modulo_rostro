"""Tarea 3 del QC: luminancia y contraste sobre las imágenes originales (96x96).

Por cada imagen calcula, sobre su versión en escala de grises:

- **Luminancia media**: promedio de los píxeles (0 = negro, 255 = blanco).
- **Contraste**: desviación estándar de los píxeles (qué tan repartidos están).

Y marca las que se salen de la norma según los umbrales de `config`:

- `subexpuesta`   → media por debajo de `LUMINANCIA_MIN` (imagen muy oscura).
- `sobreexpuesta` → media por encima de `LUMINANCIA_MAX` (imagen muy clara).
- `bajo_contraste` → desviación por debajo de `CONTRASTE_MIN` (imagen plana).

Por qué importa (CLAUDE.md, sección 5.2): una imagen sobre/subexpuesta o sin
contraste hace que Py-Feat no detecte el rostro o lo detecte con landmarks
imprecisos, degradando los Action Units.

Por defecto es EXPLORATORIA: solo mide y reporta (incluida la distribución
global) para poder calibrar los umbrales. El borrado de las imágenes fuera de
norma es explícito: se hace solo cuando main.py recibe la opción -r/--remove
(que llama a `eliminar`).
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from PIL import Image, ImageStat, UnidentifiedImageError

from .. import config

# Motivos de marcado (texto estable para filtrar el reporte después).
MOTIVO_SUBEXPUESTA = "subexpuesta"
MOTIVO_SOBREEXPUESTA = "sobreexpuesta"
MOTIVO_BAJO_CONTRASTE = "bajo_contraste"


@dataclass
class Medicion:
    """Luminancia y contraste medidos para una imagen."""

    ruta: Path
    luminancia: float
    contraste: float
    motivos: list[str] = field(default_factory=list)  # vacío si está dentro de norma


def _medir_imagen(ruta: Path) -> Medicion | None:
    """Mide luminancia y contraste de una imagen. None si no se pudo abrir.

    La tarea 1 (integridad) ya debería haber sacado las corruptas; aquí
    simplemente saltamos cualquier archivo ilegible sin frenar el proceso.
    """
    try:
        with Image.open(ruta) as im:
            gris = im.convert("L")  # escala de grises
            stat = ImageStat.Stat(gris)
            luminancia = stat.mean[0]
            contraste = stat.stddev[0]
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return None

    motivos: list[str] = []
    if luminancia < config.LUMINANCIA_MIN:
        motivos.append(MOTIVO_SUBEXPUESTA)
    elif luminancia > config.LUMINANCIA_MAX:
        motivos.append(MOTIVO_SOBREEXPUESTA)
    if contraste < config.CONTRASTE_MIN:
        motivos.append(MOTIVO_BAJO_CONTRASTE)

    return Medicion(
        ruta=ruta,
        luminancia=luminancia,
        contraste=contraste,
        motivos=motivos,
    )


def analizar(imagenes: Iterable[Path]) -> tuple[list[Medicion], list[Medicion]]:
    """Mide todas las imágenes.

    Returns:
        (todas, marcadas): la lista completa de mediciones y, aparte, solo las
        que se salen de la norma (las que tienen al menos un motivo).
    """
    todas: list[Medicion] = []
    for ruta in imagenes:
        m = _medir_imagen(ruta)
        if m is not None:
            todas.append(m)
    marcadas = [m for m in todas if m.motivos]
    return todas, marcadas


def resumen(marcadas: list[Medicion]) -> dict[str, int]:
    """Cuenta cuántas imágenes cayó en cada motivo."""
    conteo = {
        MOTIVO_SUBEXPUESTA: 0,
        MOTIVO_SOBREEXPUESTA: 0,
        MOTIVO_BAJO_CONTRASTE: 0,
    }
    for m in marcadas:
        for motivo in m.motivos:
            conteo[motivo] = conteo.get(motivo, 0) + 1
    return conteo


def _distribucion(valores: list[float]) -> str:
    """Resume una lista de valores: min / mediana / media / max."""
    if not valores:
        return "sin datos"
    return (
        f"min={min(valores):.1f}  "
        f"mediana={median(valores):.1f}  "
        f"media={mean(valores):.1f}  "
        f"max={max(valores):.1f}"
    )


def eliminar(marcadas: list[Medicion]) -> int:
    """Elimina del disco las imágenes fuera de norma. Devuelve cuántas borró.

    Idempotente: si un archivo ya no existe, lo ignora.
    """
    total = 0
    for m in marcadas:
        m.ruta.unlink(missing_ok=True)
        total += 1
    return total


def guardar_reporte(
    todas: list[Medicion],
    marcadas: list[Medicion],
    eliminado: bool = False,
    salida: Path = config.REPORTE_LUMINANCIA_TXT,
) -> Path | None:
    """Escribe el reporte .txt de la etapa (siempre, por ser exploratoria).

    Incluye los umbrales usados, la distribución global de luminancia y
    contraste, y la lista de imágenes marcadas con sus valores.

    Args:
        eliminado: si las imágenes fuera de norma ya se borraron del disco (-r).
    """
    if not todas:
        return None

    conteo = resumen(marcadas)
    lums = [m.luminancia for m in todas]
    contrs = [m.contraste for m in todas]

    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", encoding="utf-8") as f:
        f.write("Reporte de luminancia y contraste - Modulo de Rostro (QC tarea 3)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Umbrales usados (laxos, solo casos extremos):\n")
        f.write(f"  subexpuesta    : luminancia < {config.LUMINANCIA_MIN}\n")
        f.write(f"  sobreexpuesta  : luminancia > {config.LUMINANCIA_MAX}\n")
        f.write(f"  bajo_contraste : contraste  < {config.CONTRASTE_MIN}\n\n")

        f.write("Distribucion global (escala 0-255):\n")
        f.write(f"  luminancia: {_distribucion(lums)}\n")
        f.write(f"  contraste : {_distribucion(contrs)}\n\n")

        f.write(f"Imagenes medidas:      {len(todas)}\n")
        f.write(f"Subexpuestas:          {conteo[MOTIVO_SUBEXPUESTA]}\n")
        f.write(f"Sobreexpuestas:        {conteo[MOTIVO_SOBREEXPUESTA]}\n")
        f.write(f"Bajo contraste:        {conteo[MOTIVO_BAJO_CONTRASTE]}\n")
        f.write(f"Fuera de norma (total): {len(marcadas)}\n\n")

        if eliminado:
            f.write("NOTA: las imagenes fuera de norma fueron ELIMINADAS del disco (-r).\n\n")
        else:
            f.write("NOTA: esta etapa NO elimino nada del dataset; solo midio y marco.\n\n")

        f.write("Imagenes fuera de norma (ruta | motivos | luminancia | contraste):\n")
        f.write("-" * 70 + "\n")
        for m in marcadas:
            try:
                ruta_rel = m.ruta.relative_to(config.PROJECT_ROOT)
            except ValueError:
                ruta_rel = m.ruta
            motivos = ",".join(m.motivos)
            f.write(f"{ruta_rel} | {motivos} | {m.luminancia:.1f} | {m.contraste:.1f}\n")
    return salida
