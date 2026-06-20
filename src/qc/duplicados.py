"""Tarea 2 del QC: detección y eliminación de duplicados exactos.

Calcula el hash SHA256 del contenido de cada imagen y agrupa los archivos
idénticos byte a byte. Por cada grupo de duplicados conserva una sola copia
(la primera en orden alfabético, para que el resultado sea determinista) y
elimina el resto, registrando en un reporte cuáles se eliminaron y cuál quedó.

Por qué importa (CLAUDE.md, sección 5.2): los duplicados exactos sesgan la
distribución de clases sin aportar variedad real e inflan artificialmente el
tamaño del dataset.

Por defecto solo detecta y reporta. El borrado de las copias duplicadas es
explícito: se hace solo cuando main.py recibe la opción -r/--remove (que llama
a `eliminar_duplicados`).
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .. import config

# Tamaño de bloque para leer archivos sin cargarlos enteros en memoria.
_BLOQUE = 1 << 20  # 1 MiB


@dataclass
class GrupoDuplicados:
    """Un grupo de archivos con contenido idéntico (mismo hash)."""

    hash: str
    conservado: Path                       # la copia que se conserva
    eliminados: list[Path] = field(default_factory=list)  # las que se eliminan


def calcular_hash(ruta: Path) -> str:
    """Calcula el SHA256 del contenido binario de un archivo.

    Lee por bloques para no cargar la imagen entera en memoria.
    """
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(_BLOQUE), b""):
            h.update(bloque)
    return h.hexdigest()


def detectar_duplicados(imagenes: Iterable[Path]) -> list[GrupoDuplicados]:
    """Agrupa las imágenes por hash y devuelve solo los grupos con duplicados.

    Por cada grupo con más de un archivo, conserva el primero en orden
    alfabético y marca el resto como eliminables. Los archivos únicos (sin
    duplicado) no aparecen en el resultado.
    """
    por_hash: dict[str, list[Path]] = defaultdict(list)
    for ruta in imagenes:
        por_hash[calcular_hash(ruta)].append(ruta)

    grupos: list[GrupoDuplicados] = []
    for h, rutas in por_hash.items():
        if len(rutas) < 2:
            continue  # archivo único, no es duplicado
        rutas_ordenadas = sorted(rutas)
        grupos.append(
            GrupoDuplicados(
                hash=h,
                conservado=rutas_ordenadas[0],
                eliminados=rutas_ordenadas[1:],
            )
        )
    return grupos


def eliminar_duplicados(grupos: list[GrupoDuplicados]) -> int:
    """Elimina del disco las copias duplicadas (deja una por grupo).

    Idempotente: si un archivo ya no existe, lo ignora.

    Returns:
        Cantidad de archivos eliminados.
    """
    total = 0
    for grupo in grupos:
        for ruta in grupo.eliminados:
            ruta.unlink(missing_ok=True)
            total += 1
    return total


def guardar_reporte(
    grupos: list[GrupoDuplicados],
    total_revisadas: int,
    eliminado: bool = False,
    salida: Path = config.REPORTE_DUPLICADOS_TXT,
) -> Path | None:
    """Escribe un .txt con el reporte de duplicados SOLO si hay grupos.

    Registra cada grupo: el hash, la copia conservada y las copias eliminadas.
    Las rutas se guardan relativas a la raíz del proyecto para que el reporte
    sea portable entre máquinas.

    Args:
        eliminado: si las copias duplicadas ya se borraron del disco (-r).

    Returns:
        La ruta del .txt generado, o None si no hubo duplicados.
    """
    if not grupos:
        return None

    total_eliminados = sum(len(g.eliminados) for g in grupos)
    salida.parent.mkdir(parents=True, exist_ok=True)
    accion = "eliminados" if eliminado else "duplicados detectados (no se borraron)"
    with salida.open("w", encoding="utf-8") as f:
        f.write("Reporte de duplicados exactos - Modulo de Rostro (QC tarea 2)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Imagenes revisadas:        {total_revisadas}\n")
        f.write(f"Grupos de duplicados:      {len(grupos)}\n")
        f.write(f"Archivos {accion}: {total_eliminados}\n")
        f.write(f"Imagenes unicas restantes: {total_revisadas - total_eliminados}\n\n")
        f.write("Detalle por grupo (hash | conservado | eliminados):\n")
        f.write("-" * 70 + "\n")
        for grupo in grupos:
            conservado = _rel(grupo.conservado)
            f.write(f"\n[{grupo.hash[:16]}...]\n")
            f.write(f"  CONSERVADO: {conservado}\n")
            for ruta in grupo.eliminados:
                f.write(f"  eliminado:  {_rel(ruta)}\n")
    return salida


def _rel(ruta: Path) -> Path:
    """Devuelve la ruta relativa a la raíz del proyecto si es posible."""
    try:
        return ruta.relative_to(config.PROJECT_ROOT)
    except ValueError:
        return ruta
