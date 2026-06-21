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
from src.modelo import config_entrenamiento
from src.modelo import datos as modelo_datos
from src.modelo import random_forest
from src.qc import duplicados, filtrado, integridad, luminancia, pyfeat, reescalado


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


def ejecutar_filtrado() -> None:
    """Tareas 7-9: filtra el CSV de Py-Feat (v5 -> v6) y escribe el log consolidado."""
    print("== QC tareas 7-9: filtrado del CSV (v5 -> v6) ==")
    if not config.PYFEAT_CSV_V5.exists():
        print(f"No existe el CSV de Py-Feat (v5): {config.PYFEAT_CSV_V5}")
        print("Genera primero v5 (tarea 6, Py-Feat) o ajusta PYFEAT_CSV_V5 en src/config.py.")
        return

    print(f"Entrada (v5): {config.PYFEAT_CSV_V5}")
    print(f"Salida (v6):  {config.PYFEAT_CSV_V6}")
    print(f"Umbrales: FaceScore >= {config.FACESCORE_MIN}, "
          f"pose <= {config.POSE_MAX_GRADOS}°, excluye '{config.CLASE_EXCLUIDA}'")

    res = filtrado.filtrar()
    log = filtrado.guardar_log(res)

    print("\n-- Descartes por filtro --")
    print(f"Total inicial (v5):       {res.total_inicial}")
    print(f"  contempt:               {res.desc_contempt}")
    print(f"  multi-rostro:           {res.desc_multirostro}")
    print(f"  T7 FaceScore:           {res.desc_facescore}")
    print(f"  T7 pose:                {res.desc_pose}")
    print(f"  T8 landmarks NaN:       {res.desc_landmark_nan}")
    print(f"  T8 AUs degenerados:     {res.desc_au_degenerado}")
    print(f"Total final (v6):         {res.total_final}")

    print("\n-- Distribución de clases (v6) --")
    for clase in config.TAXONOMIA_7:
        n = res.distribucion.get(clase, 0)
        pct = 100 * n / res.total_final if res.total_final else 0.0
        print(f"  {clase:10s} {n:6d}  ({pct:4.1f}%)")

    print(f"\nCSV v6 -> {res.csv_v6}")
    print(f"Log     -> {log}")


def ejecutar_random_forest(args) -> None:
    """Entrena el Random Forest sobre train, evalúa en val y guarda el modelo.

    El test queda reservado: NO se evalúa aquí (usar --test-final cuando el modelo
    esté elegido).
    """
    print("== Random Forest: entrenamiento (train + val) ==")
    if not config.PYFEAT_CSV_V6.exists():
        print(f"No existe el dataset v6: {config.PYFEAT_CSV_V6}")
        print("Genera primero el v6 (tareas 7-9): python main.py -f")
        return

    # Split 70/15/15: se crea una vez y se reutiliza (test estable). --rehacer-split
    # lo regenera (mueve el test, usar con cuidado).
    rehacer = args.rehacer_split
    if rehacer and config.SPLIT_CSV.exists():
        print("--rehacer-split: se REGENERA el split (cambia el conjunto de test).")
    sp = modelo_datos.crear_split(rehacer=rehacer)
    estado = "regenerado" if rehacer or not config.SPLIT_CSV.exists() else "reutilizado"
    conteo = sp["split"].value_counts()
    print(f"Split 70/15/15 ({estado}): "
          f"train {conteo.get('train', 0)} | val {conteo.get('val', 0)} | "
          f"test {conteo.get('test', 0)} (reservado)")

    # Base: el archivo editable config_entrenamiento.txt (se crea si no existe).
    if not config_entrenamiento.existe():
        config_entrenamiento.escribir_referencia()
        print(f"Creado {config.ENTRENAMIENTO_CONFIG_TXT.name} con los valores de referencia "
              "(edítalo para tunear).")
    hp = random_forest.hiperparametros_desde_config()

    # Los flags de CLI sobreescriben el archivo (para pruebas puntuales / loop).
    if args.n_estimators is not None:
        hp.n_estimators = args.n_estimators
    if args.max_depth is not None:
        hp.max_depth = args.max_depth
    if args.min_samples_leaf is not None:
        hp.min_samples_leaf = args.min_samples_leaf
    if args.seed is not None:
        hp.seed = args.seed
    if args.max_features is not None:
        mf = args.max_features
        if mf not in ("sqrt", "log2"):
            try:
                mf = int(mf) if mf.isdigit() else float(mf)
            except ValueError:
                pass
        hp.max_features = mf

    print(f"Parámetros (de {config.ENTRENAMIENTO_CONFIG_TXT.name} + CLI):")
    print(f"  n_estimators={hp.n_estimators}, max_depth={hp.max_depth}, "
          f"min_samples_leaf={hp.min_samples_leaf}, min_samples_split={hp.min_samples_split},")
    print(f"  max_features={hp.max_features}, criterion={hp.criterion}, "
          f"bootstrap={hp.bootstrap}, class_weight={hp.class_weight}, seed={hp.seed}")
    print(f"Features: {len(modelo_datos.columnas_features())} (20 AUs + FaceScore)")
    print("Entrenando (checkpoints cada "
          f"{hp.checkpoint_cada} árboles)...")

    ultimo = [0.0]

    def progreso(n, objetivo, acc_val, seg):
        ahora = time.time()
        if n < objetivo and ahora - ultimo[0] < 0.2:
            return
        ultimo[0] = ahora
        sys.stdout.write(
            f"\r[RF] {n}/{objetivo} árboles | val acc {acc_val:.4f} | {seg:5.1f}s   "
        )
        sys.stdout.flush()

    _, metricas, carpeta_graficos = random_forest.entrenar(hp, callback=progreso)
    print()

    print("\n-- Métricas en VALIDACIÓN --")
    print(f"Accuracy: {metricas['accuracy']:.4f}")
    print(f"F1 macro: {metricas['f1_macro']:.4f}")
    print("\n" + metricas["reporte"])
    print(f"Modelo   -> {config.MODELO_RF}")
    print(f"Meta     -> {config.MODELO_META}")
    print(f"Reporte  -> {config.REPORTE_ENTRENAMIENTO_TXT}")
    print(f"Gráficos -> {carpeta_graficos}/  (matriz de confusión, métricas por clase, "
          "importancia de features, curva de entrenamiento)")
    print(f"Checkpoints -> {config.MODELO_CHECKPOINTS_DIR}/")
    print("\nEl test sigue reservado. Para la evaluación final: python main.py --test-final")


def ejecutar_reset_config() -> None:
    """Restaura config_entrenamiento.txt a los valores de referencia (config.py)."""
    path = config_entrenamiento.escribir_referencia()
    print(f"Config de entrenamiento restaurado a los valores de referencia -> {path}")
    valores = config_entrenamiento.cargar(path)
    for nombre in config_entrenamiento.NOMBRES:
        print(f"  {nombre} = {valores[nombre]}")


def ejecutar_test_final() -> None:
    """Evalúa el modelo final guardado sobre el TEST reservado (una sola vez)."""
    print("== Random Forest: EVALUACIÓN FINAL sobre el test reservado ==")
    try:
        metricas = random_forest.evaluar_test()
    except FileNotFoundError as e:
        print(str(e))
        return
    print("\n-- Métricas en TEST --")
    print(f"Accuracy: {metricas['accuracy']:.4f}")
    print(f"F1 macro: {metricas['f1_macro']:.4f}")
    print("\n" + metricas["reporte"])
    print(f"Reporte  -> {config.REPORTE_TEST_TXT}")
    print(f"Gráficos -> {metricas.get('carpeta_graficos')}/")


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
        "-f",
        "--filtrar",
        action="store_true",
        help="Tareas 7-9: filtrar el CSV de Py-Feat (v5 -> v6) y escribir el log consolidado.",
    )
    parser.add_argument(
        "-rf",
        "--random-forest",
        action="store_true",
        help="Entrenar el Random Forest (train+val) y guardar el modelo. El test queda reservado.",
    )
    parser.add_argument(
        "--test-final",
        action="store_true",
        help="Evaluar el modelo final guardado sobre el TEST reservado (una sola vez).",
    )
    parser.add_argument(
        "--rehacer-split",
        action="store_true",
        help="Con -rf: regenerar el split 70/15/15 (cambia el conjunto de test).",
    )
    parser.add_argument(
        "--reset-config",
        action="store_true",
        help="Restaurar config_entrenamiento.txt a los valores de referencia.",
    )
    # Hiperparámetros del Random Forest (para -rf y el loop de entrenamiento).
    parser.add_argument("--n-estimators", type=int, default=None, metavar="N",
                        help=f"RF: nº de árboles (def. {config.RF_N_ESTIMATORS}).")
    parser.add_argument("--max-depth", type=int, default=None, metavar="N",
                        help="RF: profundidad máxima (def. sin límite).")
    parser.add_argument("--min-samples-leaf", type=int, default=None, metavar="N",
                        help=f"RF: mínimo de muestras por hoja (def. {config.RF_MIN_SAMPLES_LEAF}).")
    parser.add_argument("--max-features", default=None, metavar="V",
                        help=f"RF: features por split (def. {config.RF_MAX_FEATURES}).")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help=f"Semilla del RF y del split (def. {config.SPLIT_SEED}).")
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
            or args.reescalar or args.pyfeat or args.filtrar
            or args.random_forest or args.test_final or args.reset_config):
        # -r por sí solo no hace nada: hay que indicar qué etapa correr.
        parser.print_help()
        return

    if args.reset_config:
        ejecutar_reset_config()

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

    if args.filtrar:
        ejecutar_filtrado()

    if args.random_forest:
        ejecutar_random_forest(args)

    if args.test_final:
        ejecutar_test_final()


if __name__ == "__main__":
    main()
