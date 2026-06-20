"""Punto de entrada del pipeline de QC del Módulo de Rostro.

Orquesta las etapas del control de calidad descritas en CLAUDE.md (sección 5.3).
Cada etapa se activa con su propia opción de línea de comandos, de modo que se
puede ejecutar solo lo que se necesita sin re-correr todo el pipeline ni tocar
este archivo.

Cada etapa por defecto solo ESCANEA y reporta. Para que además elimine del disco
lo que detectó, se agrega la opción -r/--remove.

Uso:
    python main.py -i               # tarea 1: escanear integridad (no borra)
    python main.py -ir              # tarea 1: escanear y eliminar
    python main.py -d               # tarea 2: detectar duplicados (no borra)
    python main.py -dr              # tarea 2: detectar y eliminar duplicados
    python main.py -l               # tarea 3: luminancia/contraste (no borra)
    python main.py -lr              # tarea 3: escanear y eliminar fuera de norma
    python main.py -i -d            # varias etapas en una corrida
    python main.py                  # sin opciones: muestra la ayuda
"""

import argparse

from src import config, dataset
from src.qc import duplicados, integridad, luminancia


def ejecutar_integridad(remove: bool = False) -> None:
    """Tarea 1: verifica integridad de archivos y genera el reporte de descartes.

    Con remove=True (-r) elimina del disco las imágenes descartadas.
    """
    print("== QC tarea 1: verificación de integridad de archivos ==")
    if remove:
        print("Modo -r: se ELIMINARÁN del disco las imágenes descartadas.")
    print(f"Dataset: {config.DATASET_ROOT}")

    imagenes = list(dataset.descubrir_imagenes())
    total = len(imagenes)
    print(f"Imágenes encontradas: {total}")

    if total == 0:
        print("No se encontraron imágenes. Revisa DATASET_ROOT en src/config.py.")
        return

    descartes = integridad.verificar_integridad(imagenes)
    if remove:
        integridad.eliminar(descartes)
    reporte = integridad.guardar_reporte(descartes, total, eliminado=remove)
    conteo = integridad.resumen(descartes)

    accion = "Eliminadas" if remove else "Descartadas en total"
    print("\n-- Resumen --")
    print(f"Total revisadas:        {total}")
    print(f"Corruptas:              {conteo[integridad.MOTIVO_CORRUPTA]}")
    print(f"Dimensión anómala:      {conteo[integridad.MOTIVO_DIMENSION]}")
    print(f"{accion}:   {len(descartes)}")
    print(f"Sobreviven:             {total - len(descartes)}")

    if reporte is not None:
        print(f"\nReporte de descartes -> {reporte}")
    else:
        print("\nSin imágenes corruptas ni con dimensión anómala: no se generó reporte.")


def ejecutar_duplicados(remove: bool = False) -> None:
    """Tarea 2: detecta duplicados exactos por hash.

    Con remove=True (-r) elimina las copias extra (deja una por grupo).
    """
    print("== QC tarea 2: detección de duplicados exactos (SHA256) ==")
    if remove:
        print("Modo -r: se ELIMINARÁN las copias duplicadas (se deja una por grupo).")
    print(f"Dataset: {config.DATASET_ROOT}")

    imagenes = list(dataset.descubrir_imagenes())
    total = len(imagenes)
    print(f"Imágenes encontradas: {total}")

    if total == 0:
        print("No se encontraron imágenes. Revisa DATASET_ROOT en src/config.py.")
        return

    grupos = duplicados.detectar_duplicados(imagenes)
    a_eliminar = sum(len(g.eliminados) for g in grupos)
    if remove:
        duplicados.eliminar_duplicados(grupos)
    reporte = duplicados.guardar_reporte(grupos, total, eliminado=remove)

    accion = "Eliminados" if remove else "Duplicados detectados"
    print("\n-- Resumen --")
    print(f"Total revisadas:        {total}")
    print(f"Grupos de duplicados:   {len(grupos)}")
    print(f"{accion}:  {a_eliminar}")
    print(f"Únicas restantes:       {total - a_eliminar}")

    if reporte is not None:
        print(f"\nReporte de duplicados -> {reporte}")
    else:
        print("\nSin duplicados exactos: no se generó reporte.")


def ejecutar_luminancia(remove: bool = False) -> None:
    """Tarea 3: mide luminancia/contraste y reporta las imágenes fuera de norma.

    Por defecto solo mide y reporta. Con remove=True (-r) elimina del disco las
    imágenes fuera de norma.
    """
    print("== QC tarea 3: luminancia y contraste ==")
    if remove:
        print("Modo -r: se ELIMINARÁN del disco las imágenes fuera de norma.")
    print(f"Dataset: {config.DATASET_ROOT}")
    print(
        f"Umbrales: lum<{config.LUMINANCIA_MIN} (sub), "
        f"lum>{config.LUMINANCIA_MAX} (sobre), "
        f"contraste<{config.CONTRASTE_MIN} (plano)"
    )

    imagenes = list(dataset.descubrir_imagenes())
    total = len(imagenes)
    print(f"Imágenes encontradas: {total}")

    if total == 0:
        print("No se encontraron imágenes. Revisa DATASET_ROOT en src/config.py.")
        return

    todas, marcadas = luminancia.analizar(imagenes)
    if remove:
        luminancia.eliminar(marcadas)
    reporte = luminancia.guardar_reporte(todas, marcadas, eliminado=remove)
    conteo = luminancia.resumen(marcadas)

    etiqueta = "Eliminadas (fuera de norma)" if remove else "Fuera de norma (total)"
    print("\n-- Resumen --")
    print(f"Total medidas:          {len(todas)}")
    print(f"Subexpuestas:           {conteo[luminancia.MOTIVO_SUBEXPUESTA]}")
    print(f"Sobreexpuestas:         {conteo[luminancia.MOTIVO_SOBREEXPUESTA]}")
    print(f"Bajo contraste:         {conteo[luminancia.MOTIVO_BAJO_CONTRASTE]}")
    print(f"{etiqueta}: {len(marcadas)}")

    if reporte is not None:
        print(f"\nReporte de luminancia -> {reporte}")


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de opciones. Cada etapa de QC es una opción aparte."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Pipeline de control de calidad (QC) del Módulo de Rostro.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--integridad",
        action="store_true",
        help="Tarea 1: verificar integridad de archivos (corruptas y dimensión anómala).",
    )
    parser.add_argument(
        "-d",
        "--duplicados",
        action="store_true",
        help="Tarea 2: detectar y eliminar duplicados exactos por hash (SHA256).",
    )
    parser.add_argument(
        "-l",
        "--luminancia",
        action="store_true",
        help="Tarea 3: medir luminancia/contraste y reportar fuera de norma.",
    )
    parser.add_argument(
        "-r",
        "--remove",
        action="store_true",
        help="Elimina del disco lo detectado por las etapas activas (-i/-d/-l).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Sin ninguna etapa seleccionada no hay nada que ejecutar: mostramos la ayuda.
    if not (args.integridad or args.duplicados or args.luminancia):
        # -r por sí solo no hace nada: hay que indicar qué etapa correr.
        parser.print_help()
        return

    if args.integridad:
        ejecutar_integridad(remove=args.remove)

    if args.duplicados:
        ejecutar_duplicados(remove=args.remove)

    if args.luminancia:
        ejecutar_luminancia(remove=args.remove)


if __name__ == "__main__":
    main()
