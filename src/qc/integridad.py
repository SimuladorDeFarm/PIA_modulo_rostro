"""Tarea 1 del QC: verificación de integridad de archivos.

Detecta dos problemas sobre las imágenes originales de AffectNet (v1, 96x96):

1. **Imágenes corruptas**: archivos que no se pueden abrir o decodificar
   completamente (truncados, cabecera rota, contenido que no es imagen, etc.).
2. **Dimensiones anómalas**: imágenes cuyo tamaño no es exactamente 96x96.

Por qué importa (CLAUDE.md, sección 5.2): un archivo corrupto puede romper el
pipeline de carga, y una imagen con dimensiones distintas introduce distorsión
si después se fuerza a 96x96. Ambas deben descartarse antes de seguir.

Por defecto solo reporta (ruta + motivo). El borrado de los archivos descartados
es opcional y explícito: se hace solo cuando main.py recibe la opción -r/--remove
(que llama a `eliminar`).
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from .. import config

# Motivos de descarte (texto estable para poder filtrar el CSV después).
MOTIVO_CORRUPTA = "corrupta"
MOTIVO_DIMENSION = "dimension_anomala"


@dataclass
class Descarte:
    """Una imagen descartada por la verificación de integridad."""

    ruta: Path
    motivo: str
    detalle: str  # info extra: el error de decodificación o el tamaño hallado


def _verificar_imagen(ruta: Path) -> Descarte | None:
    """Verifica una sola imagen. Devuelve un Descarte si falla, o None si está OK.

    Abre la imagen, fuerza la decodificación completa con `load()` (así detecta
    archivos truncados, no solo cabeceras rotas) y compara el tamaño contra
    `config.EXPECTED_SIZE`.
    """
    try:
        with Image.open(ruta) as im:
            size = im.size  # (ancho, alto); se lee de la cabecera
            im.load()       # fuerza el decode real -> revienta si está corrupta
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        # OSError cubre archivos truncados; UnidentifiedImageError, contenido
        # que no es imagen; el resto, formatos malformados.
        return Descarte(ruta=ruta, motivo=MOTIVO_CORRUPTA, detalle=f"{type(e).__name__}: {e}")

    if size != config.EXPECTED_SIZE:
        esperado = f"{config.EXPECTED_SIZE[0]}x{config.EXPECTED_SIZE[1]}"
        hallado = f"{size[0]}x{size[1]}"
        return Descarte(
            ruta=ruta,
            motivo=MOTIVO_DIMENSION,
            detalle=f"esperado {esperado}, hallado {hallado}",
        )

    return None


def verificar_integridad(imagenes: Iterable[Path]) -> list[Descarte]:
    """Verifica un conjunto de imágenes y devuelve la lista de descartados."""
    descartes: list[Descarte] = []
    for ruta in imagenes:
        resultado = _verificar_imagen(ruta)
        if resultado is not None:
            descartes.append(resultado)
    return descartes


def eliminar(descartes: list[Descarte]) -> int:
    """Elimina del disco los archivos descartados. Devuelve cuántos borró.

    Idempotente: si un archivo ya no existe, lo ignora.
    """
    total = 0
    for d in descartes:
        d.ruta.unlink(missing_ok=True)
        total += 1
    return total


def guardar_reporte(
    descartes: list[Descarte],
    total_revisadas: int,
    eliminado: bool = False,
    salida: Path = config.REPORTE_INTEGRIDAD_TXT,
) -> Path | None:
    """Escribe un .txt con el reporte de integridad SOLO si hay descartes.

    El reporte incluye un resumen y la lista de descartados (ruta + motivo +
    detalle). Las rutas se guardan relativas a la raíz del proyecto para que el
    reporte sea portable entre máquinas.

    Args:
        eliminado: si los archivos descartados ya se borraron del disco (-r).

    Returns:
        La ruta del .txt generado, o None si no hubo nada que reportar (dataset
        limpio en integridad).
    """
    if not descartes:
        return None

    salida.parent.mkdir(parents=True, exist_ok=True)
    conteo = resumen(descartes)
    accion = "ELIMINADAS del disco" if eliminado else "marcadas (no se borraron)"
    with salida.open("w", encoding="utf-8") as f:
        f.write("Reporte de integridad de archivos - Modulo de Rostro (QC tarea 1)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Imagenes revisadas:   {total_revisadas}\n")
        f.write(f"Corruptas:            {conteo[MOTIVO_CORRUPTA]}\n")
        f.write(f"Dimension anomala:    {conteo[MOTIVO_DIMENSION]}\n")
        f.write(f"Descartadas en total: {len(descartes)} ({accion})\n\n")
        f.write("Lista de descartados (ruta | motivo | detalle):\n")
        f.write("-" * 70 + "\n")
        for d in descartes:
            try:
                ruta_rel = d.ruta.relative_to(config.PROJECT_ROOT)
            except ValueError:
                ruta_rel = d.ruta
            f.write(f"{ruta_rel} | {d.motivo} | {d.detalle}\n")
    return salida


def resumen(descartes: list[Descarte]) -> dict[str, int]:
    """Cuenta los descartes por motivo (para imprimir el log de la etapa)."""
    conteo: dict[str, int] = {MOTIVO_CORRUPTA: 0, MOTIVO_DIMENSION: 0}
    for d in descartes:
        conteo[d.motivo] = conteo.get(d.motivo, 0) + 1
    return conteo
