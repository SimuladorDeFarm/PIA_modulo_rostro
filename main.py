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
    python main.py -e               # tarea 5: reescalar a 128x128 (dataset nuevo)
    python main.py -p               # tarea 6: Py-Feat -> CSV (detección/pose/landmarks/AUs)
    python main.py -i -d            # varias etapas en una corrida
    python main.py                  # sin opciones: muestra la ayuda
"""

import argparse
import os
import sys
import time

from src import config, dataset
from src.qc import duplicados, integridad, luminancia, pyfeat, reescalado


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


def ejecutar_reescalado() -> None:
    """Tarea 5: reescala las imágenes sobrevivientes a 128x128 en un dataset nuevo."""
    hilos = reescalado.hilos_disponibles()
    print("== QC tarea 5: reescalado a 128x128 ==")
    print(f"Origen:  {config.DATASET_ROOT}")
    print(f"Destino: {config.DATASET_V4_ROOT}")
    print(f"Hilos:   {hilos} (máximo automático)")

    imagenes = list(dataset.descubrir_imagenes())
    total = len(imagenes)
    print(f"Imágenes a reescalar: {total}")

    if total == 0:
        print("No se encontraron imágenes. Revisa DATASET_ROOT en src/config.py.")
        return

    resultado = reescalado.reescalar_dataset(imagenes)
    reporte = reescalado.guardar_reporte(resultado, total)

    print("\n-- Resumen --")
    print(f"Imágenes de origen:     {total}")
    print(f"Reescaladas OK:         {resultado.ok}")
    print(f"Con error:              {len(resultado.errores)}")
    print(f"\nReporte de reescalado -> {reporte}")


def ejecutar_pyfeat(workers: int | None = None) -> None:
    """Tarea 6: corre Py-Feat sobre el dataset reescalado y guarda el CSV (v5).

    Lee la configuración calibrada (pyfeat_config.json) que deja el modo test
    (-pt N). Si no existe, usa los defaults de config.py y avisa que conviene
    calibrar primero.
    """
    cfg = pyfeat.cargar_config()
    if cfg:
        device = cfg.get("device") or pyfeat.resolver_dispositivo()
        batch_size = int(cfg.get("batch_size", config.PYFEAT_BATCH_SIZE))
        n_workers = workers if workers is not None else int(cfg.get("workers", config.PYFEAT_WORKERS))
        origen_cfg = f"config calibrado ({config.PYFEAT_CONFIG_JSON.name}, {cfg.get('generado', '?')})"
    else:
        device = pyfeat.resolver_dispositivo()
        batch_size = config.PYFEAT_BATCH_SIZE
        n_workers = workers if workers is not None else config.PYFEAT_WORKERS
        origen_cfg = "defaults (sin calibrar)"

    print("== QC tarea 6: extracción de features con Py-Feat ==")
    print(f"Origen:  {config.DATASET_V4_ROOT}")
    print(f"Salida:  {config.PYFEAT_CSV}")
    print(f"Parámetros: {origen_cfg}")
    if device == "cuda":
        print(f"Dispositivo: GPU (CUDA)  |  batch_size: {batch_size}")
    else:
        nucleos = os.cpu_count() or 1
        hilos = max(1, max(1, nucleos - config.PYFEAT_CPU_RESERVA) // max(1, n_workers))
        print(f"Dispositivo: CPU  |  Paralelismo: {n_workers} procesos × {hilos} hilos torch")
    if not cfg:
        print("AVISO: no hay pyfeat_config.json. Recomendado calibrar antes: "
              "python main.py -pt 10")

    imagenes = list(dataset.descubrir_imagenes(config.DATASET_V4_ROOT))
    total = len(imagenes)
    print(f"Imágenes a procesar: {total}")

    if total == 0:
        print("No hay imágenes reescaladas. Corre primero la tarea 5 (python main.py -e).")
        return

    # Impresor de progreso en vivo: una sola línea que se reescribe.
    ultimo = [0.0]

    def progreso(hechas: int, tot: int, seg: float) -> None:
        ahora = time.time()
        # Throttle: refresca como máximo ~3 veces por segundo (y siempre al final).
        if hechas < tot and ahora - ultimo[0] < 0.33:
            return
        ultimo[0] = ahora
        vel = hechas / seg if seg > 0 else 0.0
        eta = (tot - hechas) / vel if vel > 0 else 0.0
        pct = 100.0 * hechas / tot
        sys.stdout.write(
            f"\r[Py-Feat] {hechas}/{tot} ({pct:4.1f}%) | "
            f"{vel:4.1f} img/s | transcurrido {pyfeat.formato_tiempo(seg)} | "
            f"ETA {pyfeat.formato_tiempo(eta)}   "
        )
        sys.stdout.flush()

    print("Procesando (la primera vez puede tardar en cargar los modelos)...")
    resultado = pyfeat.extraer(
        imagenes, device=device, workers=n_workers,
        batch_size=batch_size, callback_progreso=progreso,
    )
    print()  # salto de línea tras la barra de progreso

    reporte = pyfeat.guardar_reporte(resultado)
    vel = resultado.total / resultado.segundos if resultado.segundos else 0.0

    print("\n-- Resumen --")
    print(f"Imágenes procesadas:    {resultado.total}")
    print(f"Con rostro detectado:   {resultado.detectadas}")
    print(f"Sin rostro detectado:   {resultado.sin_deteccion}")
    print(f"Tiempo total:           {pyfeat.formato_tiempo(resultado.segundos)} ({vel:.1f} img/s)")
    print(f"\nCSV de features -> {resultado.csv}")
    print(f"Reporte         -> {reporte}")


def ejecutar_pyfeat_test(n: int) -> None:
    """Modo test (-pt N): calibra Py-Feat sobre N imágenes y escribe el config.

    Detecta el dispositivo (GPU/CPU), mide el costo real (VRAM o RAM y tiempo),
    recomienda batch_size/workers dejando margen para no saturar la máquina, y
    guarda pyfeat_config.json para que -p lo use.
    """
    device = pyfeat.resolver_dispositivo()
    print(f"== TEST/calibración Py-Feat ({n} imágenes) ==")
    print(f"Dispositivo detectado: {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")
    print(f"Origen: {config.DATASET_V4_ROOT}")

    imagenes = list(dataset.descubrir_imagenes(config.DATASET_V4_ROOT))[:n]
    if not imagenes:
        print("No hay imágenes reescaladas. Corre primero la tarea 5 (python main.py -e).")
        return
    todas = dataset.contar_imagenes(config.DATASET_V4_ROOT)

    print(f"Midiendo {len(imagenes)} imágenes (carga de modelos + inferencia)...\n")
    m = pyfeat.test_rendimiento(imagenes, device=device)

    print("-- Métricas --")
    print(f"Imágenes medidas:        {m.n}")
    print(f"Con rostro detectado:    {m.detectadas}/{m.n}")
    print(f"Carga de modelos:        {pyfeat.formato_tiempo(m.t_carga)} (una sola vez)")
    print(f"Inferencia total:        {pyfeat.formato_tiempo(m.t_total_infer)} "
          f"(batch del test: {m.batch_test})")
    print(f"Tiempo PROMEDIO/imagen:  {m.t_promedio:.3f} s/img  ({1/m.t_promedio:.2f} img/s)"
          if m.t_promedio else "Tiempo PROMEDIO/imagen:  n/a")

    if m.device == "cuda":
        print(f"VRAM tarjeta:            {m.vram_total_mb / 1024:.1f} GB")
        print(f"VRAM modelos (base):     {m.vram_base_mb / 1024:.2f} GB")
        print(f"VRAM pico (inferencia):  {m.vram_pico_mb / 1024:.2f} GB")
        per_img = (m.vram_pico_mb - m.vram_base_mb) / m.batch_test if m.batch_test else 0.0
        print(f"VRAM por imagen (aprox): {per_img:.1f} MB")
    else:
        ram_total, ram_disp = pyfeat.memoria_sistema_gb()
        print(f"RAM PICO (1 detector):   {m.ram_pico_mb:.0f} MB ({m.ram_pico_mb / 1024:.2f} GB)")
        if ram_total:
            print(f"RAM sistema:             {ram_total:.1f} GB total, "
                  f"{ram_disp:.1f} GB disponible")

    # Recomendación + escritura del config.
    rec = pyfeat.recomendar(m)
    cfg_path = pyfeat.guardar_config(rec, m)

    print("\n-- Estimación para el dataset completo --")
    print(f"Imágenes totales:        {todas}")
    est_1 = m.t_promedio * todas
    print(f"A este ritmo (1 hilo):   ~{pyfeat.formato_tiempo(est_1)}")

    print("\n-- Recomendación (escrita en el config) --")
    if rec["device"] == "cuda":
        print(f"Dispositivo:             GPU (CUDA)")
        print(f"batch_size:              {rec['batch_size']}")
        print("Procesos:                1 (en GPU no se reparte en procesos)")
    else:
        print(f"Dispositivo:             CPU")
        print(f"Procesos (--workers):    {rec['workers']} "
              f"(deja {config.PYFEAT_CPU_RESERVA} núcleos libres)")
        est_par = est_1 / rec["workers"] if rec["workers"] else est_1
        print(f"Tiempo estimado:         ~{pyfeat.formato_tiempo(est_par)}")
    print(f"\nConfig guardado en -> {cfg_path}")
    print("Ahora puedes correr la tarea 6 con esos parámetros: python main.py -p")


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
    parser.add_argument(
        "-e",
        "--reescalar",
        action="store_true",
        help="Tarea 5: reescalar las sobrevivientes a 128x128 en un dataset nuevo (paralelo).",
    )
    parser.add_argument(
        "-p",
        "--pyfeat",
        action="store_true",
        help="Tarea 6: correr Py-Feat sobre el dataset 128x128 y guardar el CSV de features.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=f"Con -p: nº de procesos en paralelo (por defecto {config.PYFEAT_WORKERS}).",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=int,
        default=None,
        metavar="N",
        help="Con -p: modo test de rendimiento sobre N imágenes (mide tiempo y RAM). "
             "Usar como: python main.py -pt N  (o -p -t N).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Sin ninguna etapa seleccionada no hay nada que ejecutar: mostramos la ayuda.
    if not (args.integridad or args.duplicados or args.luminancia
            or args.reescalar or args.pyfeat):
        # -r por sí solo no hace nada: hay que indicar qué etapa correr.
        parser.print_help()
        return

    if args.integridad:
        ejecutar_integridad(remove=args.remove)

    if args.duplicados:
        ejecutar_duplicados(remove=args.remove)

    if args.luminancia:
        ejecutar_luminancia(remove=args.remove)

    if args.reescalar:
        ejecutar_reescalado()

    if args.pyfeat:
        if args.test is not None:
            ejecutar_pyfeat_test(args.test)
        else:
            ejecutar_pyfeat(workers=args.workers)


if __name__ == "__main__":
    main()
