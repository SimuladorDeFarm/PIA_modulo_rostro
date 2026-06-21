"""Gráficos del Random Forest: matriz de confusión y otras métricas.

Genera imágenes PNG en una carpeta con fecha y hora (`reportes/modelo/<fecha_hora>/`)
para cada corrida: la matriz de confusión (conteos y normalizada), las métricas
por clase, la importancia de los features y, en entrenamiento, la curva de accuracy
de validación a medida que crecen los árboles.

Usa solo matplotlib (backend 'Agg', sin ventana) para que funcione headless.
"""

from datetime import datetime
from pathlib import Path

from .. import config


def carpeta_fecha_hora(base: Path = config.REPORTE_MODELO_DIR) -> Path:
    """Crea y devuelve una carpeta `base/AAAA-MM-DD_HH-MM-SS/`."""
    carpeta = base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def _heatmap(ax, matriz, clases, titulo, fmt: str, cmap: str):
    """Dibuja una matriz (confusión) como heatmap anotado en el eje dado."""
    im = ax.imshow(matriz, cmap=cmap)
    ax.set_xticks(range(len(clases)))
    ax.set_yticks(range(len(clases)))
    ax.set_xticklabels(clases, rotation=45, ha="right")
    ax.set_yticklabels(clases)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    umbral = matriz.max() / 2.0 if matriz.max() else 0
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, format(matriz[i, j], fmt), ha="center", va="center",
                    color="white" if matriz[i, j] > umbral else "black", fontsize=8)
    return im


def _fig_matriz(metricas: dict, conjunto: str, salida: Path):
    """Matriz de confusión: conteos y normalizada por clase real (recall)."""
    import matplotlib.pyplot as plt
    import numpy as np

    clases = metricas["clases"]
    cm = np.array(metricas["matriz"], dtype=float)
    filas = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, filas, out=np.zeros_like(cm), where=filas != 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = _heatmap(axes[0], cm.astype(int), clases,
                   f"Matriz de confusión ({conjunto}) — conteos", "d", "Blues")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = _heatmap(axes[1], cmn, clases,
                   f"Normalizada por clase real ({conjunto})", ".2f", "Greens")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Random Forest — {conjunto} | acc {metricas['accuracy']:.3f} | "
        f"F1 macro {metricas['f1_macro']:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(salida, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fig_metricas_clase(metricas: dict, salida: Path):
    """Barras de precision / recall / f1 por clase."""
    import matplotlib.pyplot as plt
    import numpy as np

    rep = metricas["reporte_dict"]
    clases = metricas["clases"]
    prec = [rep[c]["precision"] for c in clases]
    rec = [rep[c]["recall"] for c in clases]
    f1 = [rep[c]["f1-score"] for c in clases]

    x = np.arange(len(clases))
    ancho = 0.27
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - ancho, prec, ancho, label="precision")
    ax.bar(x, rec, ancho, label="recall")
    ax.bar(x + ancho, f1, ancho, label="f1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(clases, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Métricas por clase")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(salida, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fig_importancias(modelo, features: list[str], salida: Path):
    """Importancia de cada feature (Gini) ordenada."""
    import matplotlib.pyplot as plt
    import numpy as np

    imp = getattr(modelo, "feature_importances_", None)
    if imp is None:
        return
    orden = np.argsort(imp)
    feats = [features[i] for i in orden]
    vals = imp[orden]
    fig, ax = plt.subplots(figsize=(8, max(5, len(features) * 0.3)))
    ax.barh(range(len(feats)), vals, color="#4C72B0")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.set_xlabel("importancia (Gini)")
    ax.set_title("Importancia de features")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(salida, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fig_curva_entrenamiento(historia, salida: Path):
    """Curva de accuracy de validación vs número de árboles (warm_start)."""
    import matplotlib.pyplot as plt

    if not historia:
        return
    ns = [n for n, _ in historia]
    accs = [a for _, a in historia]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, accs, marker="o")
    ax.set_xlabel("nº de árboles")
    ax.set_ylabel("accuracy (val)")
    ax.set_title("Curva de validación durante el entrenamiento")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(salida, dpi=130, bbox_inches="tight")
    plt.close(fig)


def generar(
    metricas: dict,
    modelo,
    features: list[str],
    conjunto: str,
    carpeta: Path | None = None,
    historia=None,
) -> Path:
    """Genera todas las imágenes de una corrida en una carpeta con fecha y hora.

    Args:
        metricas: salida de random_forest.evaluar (incluye matriz y reporte_dict).
        modelo: el RandomForestClassifier (para feature_importances_).
        features: nombres de columnas de entrada (orden del modelo).
        conjunto: 'validacion' o 'test' (para títulos/nombres).
        carpeta: destino; si es None se crea `reportes/modelo/<fecha_hora>/`.
        historia: lista [(n_arboles, acc_val)] para la curva (solo entrenamiento).

    Returns:
        La carpeta donde quedaron las imágenes.
    """
    import matplotlib

    matplotlib.use("Agg")  # backend sin ventana (headless)

    if carpeta is None:
        carpeta = carpeta_fecha_hora()

    _fig_matriz(metricas, conjunto, carpeta / f"matriz_confusion_{conjunto}.png")
    _fig_metricas_clase(metricas, carpeta / f"metricas_por_clase_{conjunto}.png")
    _fig_importancias(modelo, features, carpeta / "importancia_features.png")
    if historia:
        _fig_curva_entrenamiento(historia, carpeta / "curva_entrenamiento.png")
    return carpeta
