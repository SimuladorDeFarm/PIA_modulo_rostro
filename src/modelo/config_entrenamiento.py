"""Archivo de texto editable con los hiperparámetros del entrenamiento.

`config_entrenamiento.txt` es un `.txt` simple (`clave = valor`, con comentarios
`#`) para tunear el Random Forest sin tocar código:

- `-rf` lo lee al entrenar (los flags de CLI lo sobreescriben para pruebas puntuales),
- `--reset-config` lo restaura a los valores de REFERENCIA (los `RF_*` de config.py).

Si una clave falta o no parsea, se usa su valor de referencia.
"""

from pathlib import Path

from .. import config

# Especificación de cada parámetro: (nombre, valor_referencia, kind, comentario).
# `kind` define cómo se lee/escribe el valor en el .txt.
_SPEC = [
    ("n_estimators",     config.RF_N_ESTIMATORS,     "int",          "nº de árboles del bosque"),
    ("max_depth",        config.RF_MAX_DEPTH,        "int_or_none",  "profundidad máxima (None = sin límite)"),
    ("min_samples_leaf", config.RF_MIN_SAMPLES_LEAF, "int",          "mínimo de muestras por hoja"),
    ("min_samples_split", config.RF_MIN_SAMPLES_SPLIT, "int",        "mínimo de muestras para dividir un nodo"),
    ("max_features",     config.RF_MAX_FEATURES,     "max_features", "features por split: sqrt | log2 | entero | fracción"),
    ("criterion",        config.RF_CRITERION,        "str",          "criterio de división: gini | entropy | log_loss"),
    ("bootstrap",        config.RF_BOOTSTRAP,        "bool",         "muestreo con reemplazo: true | false"),
    ("class_weight",     config.RF_CLASS_WEIGHT,     "class_weight", "desbalance: balanced | none"),
    ("seed",             config.SPLIT_SEED,          "int",          "semilla del modelo (el split ya está fijo)"),
    ("checkpoint_cada",  config.RF_CHECKPOINT_CADA,  "int",          "guardar un checkpoint cada N árboles"),
]

NOMBRES = [n for n, _, _, _ in _SPEC]


def _coerce(kind: str, s: str):
    """Convierte el texto de un valor a su tipo Python según `kind`."""
    s = s.strip()
    if kind == "int":
        return int(s)
    if kind == "int_or_none":
        return None if s.lower() in ("none", "") else int(s)
    if kind == "bool":
        return s.lower() in ("true", "1", "yes", "si", "sí")
    if kind == "max_features":
        if s in ("sqrt", "log2"):
            return s
        if s.lower() in ("none", ""):
            return None
        return int(s) if s.isdigit() else float(s)
    if kind == "class_weight":
        return None if s.lower() in ("none", "") else s
    return s  # str


def _fmt(kind: str, v) -> str:
    """Representación en texto de un valor para escribir en el .txt."""
    if v is None:
        return "None"
    if kind == "bool":
        return "true" if v else "false"
    return str(v)


def escribir_referencia(path: Path = config.ENTRENAMIENTO_CONFIG_TXT) -> Path:
    """Escribe (o sobreescribe) el .txt con los valores de referencia."""
    lineas = [
        "# Configuración de entrenamiento del Random Forest",
        "# Edita los valores y corre:           python main.py -rf",
        "# Restaurar estos valores de referencia: python main.py --reset-config",
        "# (Los flags de CLI como --n-estimators sobreescriben lo de aquí.)",
        "",
    ]
    for nombre, ref, kind, comentario in _SPEC:
        lineas.append(f"# {comentario}")
        lineas.append(f"{nombre} = {_fmt(kind, ref)}")
        lineas.append("")
    path.write_text("\n".join(lineas), encoding="utf-8")
    return path


def cargar(path: Path = config.ENTRENAMIENTO_CONFIG_TXT) -> dict:
    """Lee el .txt y devuelve un dict {nombre: valor}.

    Parte de los valores de referencia y los reemplaza con los del archivo. Las
    claves ausentes o que no parsean conservan la referencia.
    """
    valores = {n: ref for n, ref, _, _ in _SPEC}
    kinds = {n: k for n, _, k, _ in _SPEC}
    if not path.exists():
        return valores
    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        clave = clave.strip()
        if clave in kinds:
            try:
                valores[clave] = _coerce(kinds[clave], valor)
            except ValueError:
                pass  # valor inválido -> se mantiene la referencia
    return valores


def existe(path: Path = config.ENTRENAMIENTO_CONFIG_TXT) -> bool:
    return path.exists()
