"""Entrenamiento, evaluación y guardado del Random Forest (módulo de rostro).

Entrena un RandomForestClassifier sobre el v6 (20 AUs + FaceScore) usando el split
70/15/15:

- **train** para ajustar el modelo,
- **val** para medir durante el entrenamiento y para la búsqueda de hiperparámetros
  (el loop posterior),
- **test** reservado: solo se evalúa con `evaluar_test()`, nunca antes.

El entrenamiento crece los árboles por tandas (`warm_start`) y guarda un checkpoint
cada N árboles, así una corrida larga es reanudable y se ve el progreso. El modelo
final se guarda con joblib para inferencia posterior, junto con un JSON de metadatos
(features, orden de clases, hiperparámetros, métricas).
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .. import config
from . import datos


def _n_jobs() -> int:
    """Núcleos a usar dejando algunos libres (no saturar la máquina)."""
    import os

    return max(1, (os.cpu_count() or 1) - config.PYFEAT_CPU_RESERVA)


@dataclass
class Hiperparametros:
    """Hiperparámetros del Random Forest (con los defaults de config.py)."""

    n_estimators: int = config.RF_N_ESTIMATORS
    max_depth: int | None = config.RF_MAX_DEPTH
    min_samples_leaf: int = config.RF_MIN_SAMPLES_LEAF
    min_samples_split: int = config.RF_MIN_SAMPLES_SPLIT
    max_features: str | int | float = config.RF_MAX_FEATURES
    criterion: str = config.RF_CRITERION
    bootstrap: bool = config.RF_BOOTSTRAP
    class_weight: str | None = config.RF_CLASS_WEIGHT
    seed: int = config.SPLIT_SEED
    checkpoint_cada: int = config.RF_CHECKPOINT_CADA

    def como_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "criterion": self.criterion,
            "bootstrap": self.bootstrap,
            "class_weight": self.class_weight,
            "seed": self.seed,
        }


def hiperparametros_desde_config() -> "Hiperparametros":
    """Construye los Hiperparametros leyendo config_entrenamiento.txt."""
    from . import config_entrenamiento

    v = config_entrenamiento.cargar()
    return Hiperparametros(
        n_estimators=v["n_estimators"],
        max_depth=v["max_depth"],
        min_samples_leaf=v["min_samples_leaf"],
        min_samples_split=v["min_samples_split"],
        max_features=v["max_features"],
        criterion=v["criterion"],
        bootstrap=v["bootstrap"],
        class_weight=v["class_weight"],
        seed=v["seed"],
        checkpoint_cada=v["checkpoint_cada"],
    )


def evaluar(modelo, X, y) -> dict:
    """Calcula accuracy, F1 macro, reporte por clase y matriz de confusión."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    pred = modelo.predict(X)
    clases = list(modelo.classes_)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "reporte": classification_report(y, pred, labels=clases, zero_division=0),
        "reporte_dict": classification_report(
            y, pred, labels=clases, zero_division=0, output_dict=True
        ),
        "matriz": confusion_matrix(y, pred, labels=clases).tolist(),
        "clases": clases,
    }


def entrenar(hp: Hiperparametros, callback=None):
    """Entrena el RF con checkpoints cada N árboles. Evalúa en val y guarda todo.

    Args:
        hp: hiperparámetros.
        callback: función(n_arboles, objetivo, acc_val, segundos) para progreso.

    Returns:
        (modelo, metricas_val) donde metricas_val es el dict de `evaluar`.
    """
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_class_weight

    X_tr, y_tr = datos.cargar_xy("train")
    X_val, y_val = datos.cargar_xy("val")

    # Con warm_start, el preset "balanced" se recalcularía en cada tanda; sklearn
    # lo desaconseja. Precalculamos los pesos una sola vez y los pasamos como dict.
    if hp.class_weight == "balanced":
        clases = np.unique(y_tr)
        pesos = compute_class_weight("balanced", classes=clases, y=y_tr)
        class_weight = dict(zip(clases, pesos))
    else:
        class_weight = hp.class_weight

    paso = max(1, min(hp.checkpoint_cada, hp.n_estimators))
    rf = RandomForestClassifier(
        n_estimators=paso,
        max_depth=hp.max_depth,
        min_samples_leaf=hp.min_samples_leaf,
        min_samples_split=hp.min_samples_split,
        max_features=hp.max_features,
        criterion=hp.criterion,
        bootstrap=hp.bootstrap,
        class_weight=class_weight,
        random_state=hp.seed,
        n_jobs=_n_jobs(),
        warm_start=True,  # permite agregar árboles por tandas (checkpoints)
    )

    config.MODELO_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    historia = []  # [(n_arboles, acc_val)] para la curva de entrenamiento
    while n < hp.n_estimators:
        n = min(n + paso, hp.n_estimators)
        rf.n_estimators = n
        rf.fit(X_tr, y_tr)
        acc_val = rf.score(X_val, y_val)
        historia.append((n, float(acc_val)))
        ckpt = config.MODELO_CHECKPOINTS_DIR / f"rf_{n:04d}_arboles.joblib"
        joblib.dump(rf, ckpt)
        if callback is not None:
            callback(n, hp.n_estimators, acc_val, time.time() - t0)

    metricas_val = evaluar(rf, X_val, y_val)

    # Modelo final + metadatos para inferencia.
    config.MODELO_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, config.MODELO_RF)
    _guardar_meta(rf, hp, metricas_val, len(X_tr))
    _guardar_reporte(hp, metricas_val, len(X_tr), len(X_val), time.time() - t0)

    # Gráficos en carpeta con fecha y hora (matriz de confusión + otras métricas).
    from . import graficos

    carpeta = graficos.generar(
        metricas_val, rf, datos.columnas_features(),
        conjunto="validacion", historia=historia,
    )

    return rf, metricas_val, carpeta


def _guardar_meta(modelo, hp: Hiperparametros, metricas_val: dict, n_train: int):
    """Guarda el JSON de metadatos (lo necesario para inferencia y trazabilidad)."""
    meta = {
        "generado": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": datos.columnas_features(),
        "clases": list(modelo.classes_),
        "hiperparametros": hp.como_dict(),
        "split": {"ratios": list(config.SPLIT_RATIOS), "seed": config.SPLIT_SEED},
        "n_train": n_train,
        "val_accuracy": metricas_val["accuracy"],
        "val_f1_macro": metricas_val["f1_macro"],
    }
    config.MODELO_META.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _matriz_texto(metricas: dict) -> str:
    """Formatea la matriz de confusión con etiquetas de clase (filas=real)."""
    clases = metricas["clases"]
    anchos = max(len(c) for c in clases)
    cab = " " * (anchos + 2) + "".join(f"{c[:6]:>8}" for c in clases)
    lineas = [cab]
    for c, fila in zip(clases, metricas["matriz"]):
        lineas.append(f"{c:<{anchos}}  " + "".join(f"{v:>8}" for v in fila))
    return "\n".join(lineas)


def _guardar_reporte(hp, metricas_val, n_train, n_val, segundos, salida: Path = config.REPORTE_ENTRENAMIENTO_TXT):
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", encoding="utf-8") as f:
        f.write("Reporte de entrenamiento - Random Forest (modulo de rostro)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Hiperparametros:\n")
        for k, v in hp.como_dict().items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTrain: {n_train} | Val: {n_val} | Features: {len(datos.columnas_features())}\n")
        f.write(f"Tiempo de entrenamiento: {segundos:.1f}s\n\n")
        f.write("-- Metricas en VALIDACION --\n")
        f.write(f"Accuracy: {metricas_val['accuracy']:.4f}\n")
        f.write(f"F1 macro: {metricas_val['f1_macro']:.4f}\n\n")
        f.write(metricas_val["reporte"] + "\n")
        f.write("Matriz de confusion (filas=real, columnas=predicho):\n")
        f.write(_matriz_texto(metricas_val) + "\n")
    return salida


def evaluar_test():
    """Evalúa el modelo final guardado sobre el TEST reservado.

    Solo debe llamarse cuando el modelo ya está elegido: el test no se toca antes.
    Devuelve el dict de métricas y escribe el reporte de test.
    """
    import joblib

    if not config.MODELO_RF.exists():
        raise FileNotFoundError(
            f"No existe el modelo {config.MODELO_RF}. Entrena primero (-rf)."
        )
    modelo = joblib.load(config.MODELO_RF)
    X_test, y_test = datos.cargar_xy("test")
    metricas = evaluar(modelo, X_test, y_test)

    config.REPORTE_TEST_TXT.parent.mkdir(parents=True, exist_ok=True)
    with config.REPORTE_TEST_TXT.open("w", encoding="utf-8") as f:
        f.write("Reporte de TEST FINAL - Random Forest (modulo de rostro)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Conjunto reservado (test), evaluado una sola vez con el modelo final.\n\n")
        f.write(f"Accuracy: {metricas['accuracy']:.4f}\n")
        f.write(f"F1 macro: {metricas['f1_macro']:.4f}\n\n")
        f.write(metricas["reporte"] + "\n")
        f.write("Matriz de confusion (filas=real, columnas=predicho):\n")
        f.write(_matriz_texto(metricas) + "\n")

    # Gráficos del test en carpeta con fecha y hora.
    from . import graficos

    carpeta = graficos.generar(
        metricas, modelo, datos.columnas_features(), conjunto="test",
    )
    metricas["carpeta_graficos"] = str(carpeta)
    return metricas
