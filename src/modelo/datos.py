"""Carga del dataset v6 y split 70/15/15 para el Random Forest.

El split es estratificado por clase y DETERMINISTA (semilla fija). Se persiste en
`features/split_train_val_test.csv` y se reutiliza en cada corrida: así el conjunto
de **test queda reservado y nunca se toca** hasta la evaluación final. Solo se
rehace si se pide explícitamente (`rehacer=True`).

Features de entrada del modelo: los 20 AUs + FaceScore (CLAUDE.md §5.1).
Etiqueta: la columna `clase` del v6 (derivada de la carpeta de AffectNet).
"""

from pathlib import Path

from .. import config
from ..qc.pyfeat import AU_COLUMNS


def columnas_features() -> list[str]:
    """Columnas de entrada del RF: los 20 AUs + los extra (FaceScore)."""
    return list(AU_COLUMNS) + list(config.RF_FEATURE_EXTRA)


def crear_split(
    csv_v6: Path = config.PYFEAT_CSV_V6,
    salida: Path = config.SPLIT_CSV,
    ratios: tuple[float, float, float] = config.SPLIT_RATIOS,
    seed: int = config.SPLIT_SEED,
    rehacer: bool = False,
):
    """Crea (o reutiliza) el split 70/15/15 estratificado y lo guarda en disco.

    Devuelve un DataFrame con columnas [imagen, clase, split]. Si el archivo ya
    existe y rehacer=False, lo reutiliza tal cual (para no mover el test).
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if salida.exists() and not rehacer:
        return pd.read_csv(salida)

    df = pd.read_csv(csv_v6)
    y = df[config.LABEL_COL]
    idx = df.index.to_numpy()
    r_train, r_val, r_test = ratios

    # 1) Aparta el test (15%), estratificado.
    idx_resto, idx_test = train_test_split(
        idx, test_size=r_test, stratify=y, random_state=seed
    )
    # 2) Del resto, saca validación (15% del total = val/(train+val) del resto).
    val_rel = r_val / (r_train + r_val)
    idx_train, idx_val = train_test_split(
        idx_resto, test_size=val_rel, stratify=y.loc[idx_resto], random_state=seed
    )

    split = pd.Series("train", index=df.index)
    split.loc[idx_val] = "val"
    split.loc[idx_test] = "test"

    out = pd.DataFrame({
        "imagen": df["imagen"],
        "clase": y,
        "split": split.values,
    })
    salida.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(salida, index=False)
    return out


def cargar_xy(
    split_name: str,
    csv_v6: Path = config.PYFEAT_CSV_V6,
    split_csv: Path = config.SPLIT_CSV,
):
    """Devuelve (X, y) del split pedido ('train' | 'val' | 'test').

    Une el v6 (features) con el archivo de split por la columna `imagen`. Pedir
    'test' aquí carga el conjunto reservado: hacerlo solo en la evaluación final.
    """
    import pandas as pd

    df = pd.read_csv(csv_v6)
    sp = pd.read_csv(split_csv)
    df = df.merge(sp[["imagen", "split"]], on="imagen", how="inner")
    sub = df[df["split"] == split_name]
    X = sub[columnas_features()]
    y = sub[config.LABEL_COL]
    return X, y


def resumen_split(split_csv: Path = config.SPLIT_CSV):
    """Devuelve un DataFrame con el conteo de clases por split (para informar)."""
    import pandas as pd

    sp = pd.read_csv(split_csv)
    return pd.crosstab(sp["clase"], sp["split"])
