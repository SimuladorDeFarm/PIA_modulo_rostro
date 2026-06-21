"""Tarea 6 del QC: extracción de features con Py-Feat.

Corre Py-Feat (Detectorv1: RetinaFace + MobileFaceNet + XGB-AU + pose) sobre el
dataset reescalado a 128x128 (v4) y guarda un único CSV local (v5) con, por cada
imagen:

- detección del rostro (bounding box + flag `face_detected`),
- score de confianza de la detección (`FaceScore`),
- ángulos de pose (`Pitch`, `Roll`, `Yaw`),
- los 68 landmarks faciales (`x_0..x_67`, `y_0..y_67`),
- los 20 Action Units FACS (`AU01..AU43`).

Todo es LOCAL (CLAUDE.md, sección 5.4): el CSV se guarda en disco, no en Drive.

Estrategia según el hardware:

- **GPU (CUDA, ej. RTX 4060):** un solo proceso con un solo detector en la GPU.
  El paralelismo lo da el `batch_size` (cuántas imágenes procesa la GPU de un
  golpe). El recurso limitante es la VRAM, que es un límite duro.
- **CPU:** varios PROCESOS en paralelo (cada uno con su detector y una fracción
  de los hilos de torch), porque en CPU la única forma de acelerar es repartir
  el trabajo. El recurso limitante es la RAM.

En ambos casos el modo test (`-t`) mide el costo real (VRAM/RAM y tiempo) sobre
unas pocas imágenes, recomienda `batch_size`/`workers` dejando un margen para no
saturar la máquina, y escribe `pyfeat_config.json`. Luego `-p` lee ese archivo.

Los modelos de identidad (embedding ArcFace, 512 cols) y gaze se desactivan
porque el proyecto no los usa: así la inferencia es más rápida y el CSV no se
infla.
"""

import json
import multiprocessing as mp
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from .. import config

# Los 20 Action Units que entrega Py-Feat (sistema FACS).
AU_COLUMNS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26",
    "AU28", "AU43",
]
# 68 landmarks: coordenadas x e y.
LANDMARK_COLUMNS = [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]
# Detección: bounding box + score de confianza.
DETECCION_COLUMNS = [
    "FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight", "FaceScore",
]
# Pose de la cabeza.
POSE_COLUMNS = ["Pitch", "Roll", "Yaw"]

# Columnas de features que se guardan (en orden), aparte de imagen/face_detected.
_FEATURE_COLUMNS = DETECCION_COLUMNS + POSE_COLUMNS + LANDMARK_COLUMNS + AU_COLUMNS


@dataclass
class ResultadoPyfeat:
    """Resumen de una corrida de Py-Feat."""

    total: int          # imágenes procesadas
    detectadas: int     # imágenes con rostro detectado
    sin_deteccion: int  # imágenes sin rostro
    csv: Path           # ruta del CSV generado
    segundos: float     # tiempo total de extracción
    device: str         # "cuda" o "cpu"
    workers: int        # procesos usados (1 en GPU)
    batch_size: int     # tamaño de lote usado


# --------------------------------------------------------------------------- #
# Detección del dispositivo
# --------------------------------------------------------------------------- #
def detectar_dispositivo() -> str:
    """Devuelve 'cuda' si hay una GPU NVIDIA disponible, si no 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolver_dispositivo(device: str | None = None) -> str:
    """Resuelve el dispositivo a usar. None/'auto' -> autodetección."""
    if device in (None, "auto"):
        return detectar_dispositivo()
    return device


def construir_detector(device: str | None = None):
    """Crea el Detectorv1 de Py-Feat sin los modelos de identidad ni gaze.

    Se desactivan (identity_model=None, gaze_model=None) porque este proyecto no
    usa ni el embedding de identidad ni la mirada, y así la inferencia es más
    rápida. Importa feat aquí dentro para no pagar el costo de carga si no se usa.
    """
    from feat import Detectorv1

    return Detectorv1(device=resolver_dispositivo(device),
                      identity_model=None, gaze_model=None)


def _hilos_cpu_seguros() -> int:
    """Núcleos de CPU a usar dejando algunos libres para no saturar el sistema."""
    return max(1, (os.cpu_count() or 1) - config.PYFEAT_CPU_RESERVA)


# --------------------------------------------------------------------------- #
# Helpers de transformación del Fex de Py-Feat
# --------------------------------------------------------------------------- #
def _ruta_relativa(valor: str, raiz: Path) -> str:
    """Convierte la ruta absoluta que entrega Py-Feat en una relativa al dataset."""
    p = Path(valor)
    try:
        return str(p.relative_to(raiz))
    except ValueError:
        return str(p)


def _fex_a_dataframe(fex, raiz: Path):
    """Reduce el Fex de Py-Feat a las columnas pedidas + imagen + face_detected."""
    df = fex.loc[:, [c for c in _FEATURE_COLUMNS if c in fex.columns]].copy()
    df.insert(0, "imagen", [_ruta_relativa(v, raiz) for v in fex["input"]])
    df.insert(1, "face_detected", df["FaceScore"].notna())
    return df


# --------------------------------------------------------------------------- #
# Extracción en GPU (un proceso, batch grande)
# --------------------------------------------------------------------------- #
def _extraer_gpu(
    rutas: list[str],
    raiz: Path,
    salida_csv: Path,
    batch_size: int,
    callback_progreso: Callable[[int, int, float], None] | None,
) -> ResultadoPyfeat:
    """Corre Py-Feat en GPU con un solo detector, por lotes de `batch_size`."""
    import warnings

    os.environ["TQDM_DISABLE"] = "1"
    warnings.filterwarnings("ignore")
    import pandas as pd
    import torch

    # Limita los hilos de CPU (carga/decodificación de imágenes) para no ahogar
    # el sistema mientras la GPU trabaja.
    torch.set_num_threads(_hilos_cpu_seguros())

    detector = construir_detector("cuda")

    total = len(rutas)
    partes = []
    detectadas = 0
    hechas = 0
    t0 = time.time()
    paso = max(1, batch_size)
    for inicio in range(0, total, paso):
        lote = rutas[inicio:inicio + paso]
        fex = detector.detect(
            lote, data_type="image", batch_size=paso, progress_bar=False
        )
        df = _fex_a_dataframe(fex, raiz)
        partes.append(df)
        detectadas += int(df["face_detected"].sum())
        hechas += len(lote)
        if callback_progreso is not None:
            callback_progreso(hechas, total, time.time() - t0)

    df = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
    salida_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida_csv, index=False)

    return ResultadoPyfeat(
        total=len(df),
        detectadas=detectadas,
        sin_deteccion=len(df) - detectadas,
        csv=salida_csv,
        segundos=time.time() - t0,
        device="cuda",
        workers=1,
        batch_size=paso,
    )


# --------------------------------------------------------------------------- #
# Extracción en CPU (varios procesos)
# --------------------------------------------------------------------------- #
def _procesar_shard(
    shard_id: int,
    paths: list[str],
    raiz: str,
    batch_size: int,
    hilos_torch: int,
    tmpdir: str,
    cola,
) -> tuple[str, int, int]:
    """Worker: procesa una porción de imágenes en su propio proceso.

    Limita los hilos de torch a `hilos_torch` (para no saturar la CPU entre
    procesos), corre Py-Feat por mini-lotes e informa el avance por `cola`.
    Escribe un CSV parcial y devuelve (ruta_parcial, total, detectadas).
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(hilos_torch))
    os.environ["TQDM_DISABLE"] = "1"  # silencia las barras internas de Py-Feat
    import warnings

    warnings.filterwarnings("ignore")
    import torch

    torch.set_num_threads(hilos_torch)
    import pandas as pd

    raiz_path = Path(raiz)
    detector = construir_detector("cpu")

    partes = []
    paso = max(batch_size, 8)  # cada cuántas imágenes reportar avance
    for inicio in range(0, len(paths), paso):
        lote = paths[inicio:inicio + paso]
        fex = detector.detect(
            lote, data_type="image", batch_size=batch_size, progress_bar=False
        )
        partes.append(_fex_a_dataframe(fex, raiz_path))
        cola.put(len(lote))  # avisa al proceso principal cuántas terminó

    df = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
    parcial = str(Path(tmpdir) / f"parte_{shard_id:03d}.csv")
    df.to_csv(parcial, index=False)
    detectadas = int(df["face_detected"].sum()) if len(df) else 0
    return parcial, len(df), detectadas


def _repartir(imagenes: list[str], n: int) -> list[list[str]]:
    """Divide la lista en n bloques contiguos lo más parejos posible."""
    if n <= 1:
        return [imagenes]
    tam = (len(imagenes) + n - 1) // n
    return [imagenes[i:i + tam] for i in range(0, len(imagenes), tam)]


def _extraer_cpu(
    rutas: list[str],
    raiz_origen: Path,
    salida_csv: Path,
    workers: int,
    batch_size: int,
    callback_progreso: Callable[[int, int, float], None] | None,
) -> ResultadoPyfeat:
    """Corre Py-Feat en CPU repartiendo el trabajo en varios procesos."""
    import pandas as pd

    total = len(rutas)
    workers = max(1, min(workers, total)) if total else 1

    # Reparte los hilos de torch entre procesos sin pasar de los núcleos seguros,
    # para no sobre-suscribir la CPU ni inutilizar el sistema.
    hilos_torch = max(1, _hilos_cpu_seguros() // workers)
    shards = _repartir(rutas, workers)

    salida_csv.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")  # 'spawn' es seguro con torch

    with tempfile.TemporaryDirectory() as tmpdir:
        with ctx.Manager() as mgr:
            cola = mgr.Queue()
            with ctx.Pool(processes=workers) as pool:
                tareas = [
                    pool.apply_async(
                        _procesar_shard,
                        (i, shard, str(raiz_origen), batch_size, hilos_torch, tmpdir, cola),
                    )
                    for i, shard in enumerate(shards)
                ]

                # Drena la cola de avance hasta procesar todas las imágenes.
                hechas = 0
                while hechas < total:
                    hechas += cola.get()
                    if callback_progreso is not None:
                        callback_progreso(hechas, total, time.time() - t0)

                resultados = [t.get() for t in tareas]  # espera a que terminen

        # Une los CSV parciales en el CSV final (en orden de shard).
        parciales = [pd.read_csv(p) for p, _, _ in resultados if Path(p).exists()]
        df = pd.concat(parciales, ignore_index=True) if parciales else pd.DataFrame()
        df.to_csv(salida_csv, index=False)

    detectadas = sum(d for _, _, d in resultados)
    procesadas = sum(n for _, n, _ in resultados)
    return ResultadoPyfeat(
        total=procesadas,
        detectadas=detectadas,
        sin_deteccion=procesadas - detectadas,
        csv=salida_csv,
        segundos=time.time() - t0,
        device="cpu",
        workers=workers,
        batch_size=batch_size,
    )


def extraer(
    imagenes: Iterable[Path],
    raiz_origen: Path = config.DATASET_V4_ROOT,
    salida_csv: Path = config.PYFEAT_CSV,
    device: str | None = None,
    workers: int = config.PYFEAT_WORKERS,
    batch_size: int = config.PYFEAT_BATCH_SIZE,
    callback_progreso: Callable[[int, int, float], None] | None = None,
) -> ResultadoPyfeat:
    """Corre Py-Feat y guarda el CSV con las columnas pedidas.

    Despacha a la estrategia GPU (un proceso, batch grande) o CPU (varios
    procesos) según el dispositivo.

    Args:
        imagenes: rutas de las imágenes (dataset reescalado v4).
        device: 'cuda', 'cpu' o None/'auto' para autodetectar.
        workers: nº de procesos en paralelo (solo CPU; en GPU se ignora).
        batch_size: tamaño de lote de inferencia.
        callback_progreso: función(hechas, total, segundos) para informar avance.
    """
    dev = resolver_dispositivo(device)
    rutas = [str(p) for p in imagenes]

    if dev == "cuda":
        return _extraer_gpu(rutas, raiz_origen, salida_csv, batch_size, callback_progreso)
    return _extraer_cpu(rutas, raiz_origen, salida_csv, workers, batch_size, callback_progreso)


# --------------------------------------------------------------------------- #
# Test de rendimiento / calibración
# --------------------------------------------------------------------------- #
@dataclass
class MetricasTest:
    """Métricas de un test de rendimiento de Py-Feat (1 solo proceso)."""

    n: int                 # imágenes del test
    device: str            # "cuda" o "cpu"
    t_carga: float         # seg en cargar los modelos
    t_total_infer: float   # seg de inferencia (sin la carga)
    t_promedio: float      # seg por imagen
    detectadas: int        # cuántas con rostro detectado
    batch_test: int        # batch usado en el test
    # CPU:
    ram_pico_mb: float = 0.0     # RAM pico del proceso (1 detector)
    # GPU:
    vram_base_mb: float = 0.0    # VRAM tras cargar los modelos (sin inferir)
    vram_pico_mb: float = 0.0    # VRAM pico durante la inferencia
    vram_total_mb: float = 0.0   # VRAM total de la tarjeta


def memoria_sistema_gb() -> tuple[float, float]:
    """Devuelve (RAM total, RAM disponible) del sistema en GB. (0,0) si falla."""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for linea in f:
                clave, _, resto = linea.partition(":")
                info[clave.strip()] = int(resto.strip().split()[0])  # kB
        return info.get("MemTotal", 0) / 1024**2, info.get("MemAvailable", 0) / 1024**2
    except OSError:
        return 0.0, 0.0


def _test_gpu(rutas: list[str], raiz: Path, batch_size: int) -> MetricasTest:
    """Mide tiempo y VRAM de Py-Feat en GPU sobre un lote de prueba.

    Lee la VRAM en dos momentos: tras cargar los modelos (base) y el pico durante
    la inferencia. La diferencia, dividida por el tamaño del lote, da el costo de
    VRAM por imagen, que sirve para recomendar el batch_size más grande que cabe.
    """
    import warnings

    os.environ["TQDM_DISABLE"] = "1"
    warnings.filterwarnings("ignore")
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    detector = construir_detector("cuda")
    torch.cuda.synchronize()
    vram_base_mb = torch.cuda.memory_reserved() / 1024**2
    t_carga = time.time() - t0

    eff_batch = max(1, min(batch_size, len(rutas)))
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    fex = detector.detect(rutas, data_type="image", batch_size=eff_batch, progress_bar=False)
    df = _fex_a_dataframe(fex, raiz)
    detectadas = int(df["face_detected"].sum())
    torch.cuda.synchronize()
    t_total = time.time() - t1

    vram_pico_mb = max(vram_base_mb, torch.cuda.max_memory_reserved() / 1024**2)
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

    n = len(rutas)
    return MetricasTest(
        n=n,
        device="cuda",
        t_carga=t_carga,
        t_total_infer=t_total,
        t_promedio=t_total / n if n else 0.0,
        detectadas=detectadas,
        batch_test=eff_batch,
        vram_base_mb=vram_base_mb,
        vram_pico_mb=vram_pico_mb,
        vram_total_mb=vram_total_mb,
    )


def _test_cpu(rutas: list[str], raiz: Path) -> MetricasTest:
    """Mide tiempo y RAM de Py-Feat en UN solo proceso en CPU."""
    import resource
    import warnings

    os.environ["TQDM_DISABLE"] = "1"
    warnings.filterwarnings("ignore")
    import pandas as pd  # noqa: F401  (asegura disponibilidad para _fex_a_dataframe)

    t0 = time.time()
    detector = construir_detector("cpu")
    t_carga = time.time() - t0

    detectadas = 0
    t1 = time.time()
    for r in rutas:
        fex = detector.detect([r], data_type="image", batch_size=1, progress_bar=False)
        df = _fex_a_dataframe(fex, raiz)
        detectadas += int(df["face_detected"].sum())
    t_total = time.time() - t1

    # ru_maxrss en Linux viene en kilobytes -> MB.
    ram_pico_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    n = len(rutas)
    return MetricasTest(
        n=n,
        device="cpu",
        t_carga=t_carga,
        t_total_infer=t_total,
        t_promedio=t_total / n if n else 0.0,
        detectadas=detectadas,
        batch_test=1,
        ram_pico_mb=ram_pico_mb,
    )


def test_rendimiento(
    imagenes: Iterable[Path],
    raiz_origen: Path = config.DATASET_V4_ROOT,
    device: str | None = None,
    batch_size: int = config.PYFEAT_BATCH_SIZE,
) -> MetricasTest:
    """Mide el costo real de Py-Feat sobre unas pocas imágenes (1 proceso).

    Despacha al test de GPU (mide VRAM) o de CPU (mide RAM) según el dispositivo.
    Con ese dato se recomienda cuánto batch/cuántos procesos caben sin saturar.
    """
    dev = resolver_dispositivo(device)
    rutas = [str(p) for p in imagenes]
    raiz = Path(raiz_origen)
    if dev == "cuda":
        return _test_gpu(rutas, raiz, batch_size)
    return _test_cpu(rutas, raiz)


def recomendar(m: MetricasTest) -> dict:
    """Recomienda device/batch_size/workers a partir de las métricas del test.

    - GPU: el batch_size más grande que quepa en la VRAM dejando un margen
      (PYFEAT_VRAM_MARGEN). Un solo proceso (workers=1).
    - CPU: cuántos procesos caben en RAM (con holgura) sin pasar de los núcleos
      seguros (dejando PYFEAT_CPU_RESERVA libres).
    """
    if m.device == "cuda":
        disponible = m.vram_total_mb * config.PYFEAT_VRAM_MARGEN - m.vram_base_mb
        per_img = (m.vram_pico_mb - m.vram_base_mb) / m.batch_test if m.batch_test else 0.0
        if per_img > 0 and disponible > 0:
            batch_rec = int(disponible / per_img)
        else:
            batch_rec = config.PYFEAT_BATCH_SIZE
        batch_rec = max(1, min(batch_rec, config.PYFEAT_BATCH_MAX))
        return {"device": "cuda", "batch_size": batch_rec, "workers": 1}

    # CPU
    _, ram_disp = memoria_sistema_gb()
    ram_pico_gb = m.ram_pico_mb / 1024
    if ram_disp and ram_pico_gb > 0:
        cabe_ram = max(1, int((ram_disp * 0.8) / ram_pico_gb))
    else:
        cabe_ram = 1
    cpu_max = _hilos_cpu_seguros()
    workers = max(1, min(cabe_ram, cpu_max))
    return {"device": "cpu", "batch_size": config.PYFEAT_BATCH_SIZE, "workers": workers}


# --------------------------------------------------------------------------- #
# Archivo de configuración (lo escribe -t, lo lee -p)
# --------------------------------------------------------------------------- #
def guardar_config(
    recomendacion: dict,
    metricas: MetricasTest,
    salida: Path = config.PYFEAT_CONFIG_JSON,
) -> Path:
    """Escribe pyfeat_config.json con la recomendación + datos del test.

    Es específico de cada máquina (depende de su GPU/CPU/RAM), por eso va en el
    .gitignore y no se versiona.
    """
    data = dict(recomendacion)
    data["generado"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["device_detectado"] = metricas.device
    data["t_promedio_img_s"] = round(metricas.t_promedio, 4)
    if metricas.device == "cuda":
        data["vram_total_gb"] = round(metricas.vram_total_mb / 1024, 2)
        data["vram_base_gb"] = round(metricas.vram_base_mb / 1024, 2)
        data["vram_pico_test_gb"] = round(metricas.vram_pico_mb / 1024, 2)
    else:
        data["ram_pico_gb"] = round(metricas.ram_pico_mb / 1024, 2)
    salida.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return salida


def cargar_config(ruta: Path = config.PYFEAT_CONFIG_JSON) -> dict | None:
    """Lee pyfeat_config.json si existe y es válido; si no, None."""
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #
def guardar_reporte(
    resultado: ResultadoPyfeat,
    salida: Path = config.REPORTE_PYFEAT_TXT,
) -> Path:
    """Escribe el reporte .txt de la corrida de Py-Feat."""
    salida.parent.mkdir(parents=True, exist_ok=True)
    try:
        csv_rel = resultado.csv.relative_to(config.PROJECT_ROOT)
    except ValueError:
        csv_rel = resultado.csv

    n_cols = len(_FEATURE_COLUMNS) + 2
    vel = resultado.total / resultado.segundos if resultado.segundos else 0.0
    if resultado.device == "cuda":
        paralelismo = f"GPU (CUDA), batch_size={resultado.batch_size}"
    else:
        paralelismo = f"CPU, {resultado.workers} procesos, batch_size={resultado.batch_size}"
    with salida.open("w", encoding="utf-8") as f:
        f.write("Reporte de Py-Feat - Modulo de Rostro (QC tarea 6)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dispositivo:            {resultado.device}\n")
        f.write(f"Paralelismo:            {paralelismo}\n")
        f.write(f"CSV generado:           {csv_rel}\n")
        f.write(f"Columnas por imagen:    {n_cols} "
                "(imagen, face_detected, 5 deteccion, 3 pose, 136 landmarks, 20 AUs)\n\n")
        f.write(f"Imagenes procesadas:    {resultado.total}\n")
        f.write(f"Con rostro detectado:   {resultado.detectadas}\n")
        f.write(f"Sin rostro detectado:   {resultado.sin_deteccion}\n")
        f.write(f"Tiempo total:           {formato_tiempo(resultado.segundos)} "
                f"({vel:.1f} img/s)\n")
    return salida


def formato_tiempo(segundos: float) -> str:
    """Formatea segundos como hh:mm:ss (o mm:ss si es menos de una hora)."""
    s = int(segundos)
    h, resto = divmod(s, 3600)
    m, s = divmod(resto, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
