# Módulo de Rostro — Pipeline de QC y entrenamiento

Pipeline completo del módulo de rostro de NeuroEmoInnovat: desde el control de
calidad del dataset AffectNet hasta el entrenamiento del Random Forest que
clasifica emociones a partir de los Action Units (AUs) extraídos con Py-Feat.

## Pipeline completo (orden de ejecución)

Cada etapa parte de la salida de la anterior. El versionado del dataset
(`v1`→`v6`) está detallado en `CLAUDE.md` §5.5.

| Orden | Etapa | Opción | Entrada → Salida |
|------:|-------|--------|------------------|
| 1 | Integridad de archivos | `-i` | `AffNet/` (v1) |
| 2 | Duplicados exactos (SHA256) | `-d` | `AffNet/` (→ v2) |
| 3 | Luminancia / contraste | `-l` | `AffNet/` (→ v3) |
| — | *Nitidez (tarea 4): **omitida** por decisión del Líder Desarrollo* | — | — |
| 5 | Reescalado a 128×128 | `-e` | `AffNet/` → `AffNet_128/` (v4) |
| 6 | Py-Feat → features | `-p` (calibrar con `-pt`) | `AffNet_128/` → `features/pyfeat_v5.csv` (v5) |
| 7-9 | Filtrado + log consolidado | `-f` | `pyfeat_features_v5.csv` → `features/pyfeat_v6.csv` (v6) |
| 10 | Entrenamiento Random Forest | `-rf` | v6 → `models/random_forest.joblib` |
| 11 | Evaluación final | `--test-final` | v6 (test reservado) → reporte |

> **Nota:** la tarea 6 (Py-Feat) se corrió en **Google Colab con GPU**
> (`pyfeat_colab.ipynb`); el `v5` vigente vino de ahí. Ver `CLAUDE.md` §5.4.

## Instalación

Requiere **Python 3.10+**. Desde la raíz del proyecto:

```bash
# 1) Crear el entorno virtual
python -m venv .venv

# 2) Activarlo
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows (PowerShell)

# 3) Instalar dependencias
pip install -r requirements.txt
```

> La **tarea 6 (Py-Feat: `-p` / `-pt`)** necesita además el paquete `py-feat` (arrastra
> PyTorch, es pesado) y, de preferencia, una GPU NVIDIA. En este proyecto esa tarea se
> corrió en **Google Colab** (`pyfeat_colab.ipynb`), así que **no hace falta instalar
> `py-feat` en local** salvo que quieras correrla aquí. En ese caso, descomenta la línea
> correspondiente en `requirements.txt`.

## Ejecución

Siempre desde la raíz del proyecto y con el entorno activo:

```bash
source .venv/bin/activate
python main.py [opciones]      # ver tabla de opciones abajo
```

---

## Probar el modelo ya entrenado (sin reentrenar ni extraer landmarks)

El repo **ya incluye el modelo entrenado** (`models/random_forest.joblib`) y los
datos necesarios (`features/pyfeat_v6.csv` + el split). Para correrlo y ver los
resultados **no hace falta GPU, ni Py-Feat, ni volver a extraer landmarks**:

```bash
git clone <repo> && cd modulo_rostro
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # NO instala py-feat (no se necesita)

python main.py --test-final          # evalúa el modelo sobre el test reservado
```

Esto imprime accuracy, F1 y el reporte por clase, y genera las imágenes (matriz de
confusión, métricas por clase, importancia de features) en
`reportes/modelo/<fecha_hora>/`.

### Revisar / mejorar el entrenamiento

El código del entrenamiento está en [`src/modelo/`](src/modelo/):

- `random_forest.py` — entrenamiento, evaluación, guardado del modelo.
- `datos.py` — carga del dataset y split 70/15/15.
- `graficos.py` — matriz de confusión y gráficos.
- `config_entrenamiento.txt` (raíz) — hiperparámetros editables.

Para experimentar con otra configuración (sigue **sin** necesitar landmarks ni GPU):

```bash
# editar config_entrenamiento.txt (o pasar flags) y reentrenar:
python main.py -rf --n-estimators 500 --max-depth 25
python main.py --test-final
```

---

## Opciones disponibles

Cada etapa por defecto **solo escanea y reporta**. Para que además elimine del
disco lo que detecta, se agrega `-r` / `--remove`.

| Opción | Descripción |
|--------|-------------|
| `-i` / `--integridad` | **Tarea 1** — detecta imágenes corruptas y con dimensiones anómalas (distinto a 96×96). |
| `-d` / `--duplicados` | **Tarea 2** — detecta duplicados exactos por hash (SHA256); conserva una copia por grupo. |
| `-l` / `--luminancia` | **Tarea 3** — mide luminancia y contraste (96×96) y marca las imágenes fuera de norma. |
| `-e` / `--reescalar` | **Tarea 5** — reescala las sobrevivientes a 128×128 en un dataset nuevo (`AffNet_128/`), en paralelo con todos los hilos. No toca el original. |
| `-p` / `--pyfeat` | **Tarea 6** — corre Py-Feat sobre `AffNet_128/` y guarda el CSV de features (detección, confianza, pose, 68 landmarks, 20 AUs). Usa lo que calibró `-pt` (ver abajo). |
| `--workers N` | Con `-p` (solo CPU): fuerza el nº de procesos en paralelo, ignorando el del config. En GPU se ignora (siempre 1 proceso). |
| `-t N` / `--test N` | Con `-p`: **calibra** sobre N imágenes. Detecta GPU/CPU, mide VRAM/RAM y tiempo, recomienda `batch_size`/`workers` y **escribe `pyfeat_config.json`**. Usar como `-pt N`. |
| `-f` / `--filtrar` | **Tareas 7-9** — filtra el CSV de Py-Feat (v5 → v6): quita `contempt`, deduplica multi-rostro, descarta baja confianza / pose extrema / landmarks NaN / AUs degenerados, y escribe el log consolidado con la distribución de clases. |
| `-rf` / `--random-forest` | **Entrenamiento** — entrena el Random Forest sobre `train` (20 AUs + FaceScore, `class_weight='balanced'`), evalúa en `val` y guarda el modelo. El `test` queda **reservado** (no se toca). |
| `--test-final` | **Evaluación final** — evalúa el modelo guardado sobre el `test` reservado. Correr solo una vez, cuando el modelo ya está elegido. |
| `--rehacer-split` | Con `-rf`: regenera el split 70/15/15 (⚠️ cambia el conjunto de test). |
| `--reset-config` | Restaura `config_entrenamiento.txt` a los valores de referencia (los de `src/config.py`). |
| `--n-estimators` `--max-depth` `--min-samples-leaf` `--max-features` `--seed` | Sobreescriben puntualmente lo de `config_entrenamiento.txt` (para `-rf` y el loop). |
| `-r` / `--remove` | Modificador: **elimina del disco** lo detectado por las etapas activas (`-i`/`-d`/`-l`). Sin `-r` no se borra nada. |

> Las flags cortas se pueden agrupar: `-lr` ≡ `-l -r`, `-ir` ≡ `-i -r`, `-dr` ≡ `-d -r`.

### Ejemplos

```bash
python main.py -i        # tarea 1: escanear integridad (no borra)
python main.py -ir       # tarea 1: escanear y eliminar
python main.py -d        # tarea 2: detectar duplicados (no borra)
python main.py -dr       # tarea 2: detectar y eliminar duplicados
python main.py -l        # tarea 3: luminancia/contraste (no borra)
python main.py -lr       # tarea 3: escanear y eliminar fuera de norma
python main.py -e        # tarea 5: reescalar a 128x128 (dataset nuevo)
python main.py -pt 10    # tarea 6: CALIBRAR con 10 imágenes -> escribe pyfeat_config.json
python main.py -p        # tarea 6: correr Py-Feat usando el config calibrado
python main.py -p --workers 3   # tarea 6: forzar 3 procesos (solo CPU; ignora el config)
python main.py -f        # tareas 7-9: filtrar el CSV (v5 -> v6) + log consolidado
python main.py -rf       # entrenar el Random Forest (lee config_entrenamiento.txt)
python main.py -rf --n-estimators 500 --max-depth 20   # sobreescribir hiperparámetros por CLI
python main.py --reset-config   # restaurar config_entrenamiento.txt a los valores de referencia
python main.py --test-final   # evaluar el modelo final sobre el test reservado
python main.py -i -d     # varias etapas en una corrida (sin borrar)
python main.py           # sin opciones: muestra esta ayuda
python main.py -h        # igual
```

### Flujo recomendado para la tarea 6 (Py-Feat)

1. **Calibrar una vez** en la máquina donde se va a correr:

   ```bash
   python main.py -pt 10
   ```

   Esto detecta solo si hay GPU NVIDIA (CUDA) o solo CPU, mide cuánta
   VRAM/RAM y tiempo consume Py-Feat, y escribe `pyfeat_config.json` con el
   `batch_size` (GPU) o el nº de procesos (CPU) que **usan harto el hardware
   sin saturar la máquina** (deja margen de VRAM y núcleos de CPU libres).

2. **Correr la extracción**, que lee ese config automáticamente:

   ```bash
   python main.py -p
   ```

> `pyfeat_config.json` es específico de cada máquina (depende de su GPU/CPU/RAM),
> por eso está en `.gitignore` y **no se versiona**: cada quien calibra el suyo
> con `-pt` la primera vez.
>
> **En GPU** (ej. RTX 4060) corre en un solo proceso y el paralelismo lo da el
> `batch_size`; el límite es la VRAM. **En CPU** reparte el trabajo en varios
> procesos; el límite es la RAM. El calibrador elige la estrategia según el
> hardware que detecte.

> ⚠️ Agregar `-r` **borra archivos del disco** de forma irreversible. Conviene
> correr primero la etapa sin `-r` y revisar el reporte.

---

## Extraer landmarks (y AUs/pose) de cualquier dataset — `pyfeat_colab.ipynb`

El notebook `pyfeat_colab.ipynb` no está atado a AffectNet: sirve para sacar los
**68 landmarks faciales** (`x_0..x_67`, `y_0..y_67`) —además de detección,
confianza, pose y los 20 AUs— de **cualquier dataset de imágenes**. Recorre la
carpeta de forma **recursiva** (`rglob`), así que funciona sin importar cómo estén
organizadas las subcarpetas; la columna `imagen` del CSV guarda la ruta relativa
de cada archivo (preservando, por ejemplo, la carpeta de clase).

### Qué hay que modificar

Solo dos variables en la **celda 3 (`# 3) Configuración e imports`)**:

| Variable | Cambiar a | Para qué |
|----------|-----------|----------|
| `BASE_PATH` | la carpeta raíz de **tu** dataset en Drive | de dónde se leen las imágenes (búsqueda recursiva) |
| `OUTPUT_CSV` | dónde quieres guardar el resultado | el CSV de salida (se guarda en Drive) |

Opcionales:

- `TARGET_SIZE` (celda 3): resolución a la que se reescala antes de Py-Feat
  (por defecto `128×128`). Cámbialo si tu pipeline usa otra.
- `EXTS` (celda 5, `# 5) Descubrir todas las imágenes`): agrega extensiones si tu
  dataset trae formatos distintos a `.jpg/.jpeg/.png/.bmp` (p. ej. `.webp`).

### Pasos

1. Subir `pyfeat_colab.ipynb` a Colab y activar GPU:
   `Entorno de ejecución → Cambiar tipo de entorno → GPU` (T4).
2. Editar `BASE_PATH` y `OUTPUT_CSV` en la celda 3.
3. Ejecutar las celdas en orden (instala Py-Feat, monta Drive, procesa por lotes
   en RAM y guarda el CSV).

El CSV resultante trae una fila por imagen con: `imagen`, `face_detected`,
bounding box + `FaceScore`, `Pitch/Roll/Yaw`, los **136 valores de landmarks** y
los 20 AUs. Si solo te interesan los landmarks, basta con quedarte con las
columnas `x_*` / `y_*`.

---

## Reportes generados

| Tarea | Archivo |
|-------|---------|
| `-i` | `reportes/integridad/reporte_integridad.txt` — solo se crea si hay imágenes descartadas. |
| `-d` | `reportes/duplicados/reporte_duplicados.txt` — solo se crea si hay grupos de duplicados. |
| `-l` | `reportes/luminancia/reporte_luminancia.txt` — siempre se genera (incluye la distribución global y la lista fuera de norma). |
| `-e` | `reportes/reescalado/reporte_reescalado.txt` — siempre se genera. Dataset de salida: `AffNet_128/`. |
| `-p` | `reportes/pyfeat/reporte_pyfeat.txt` + CSV de features `features/pyfeat_v5.csv`. |
| `-f` | `reportes/filtrado/log_qc_consolidado.txt` + CSV final de entrenamiento `features/pyfeat_v6.csv` (v6, el que usa el Random Forest). |
| `-rf` | `models/random_forest.joblib` (modelo final) + `models/random_forest_meta.json` (features/clases/métricas) + `models/checkpoints/` + `reportes/modelo/reporte_entrenamiento.txt` + **gráficos en `reportes/modelo/<fecha_hora>/`** + split fijo en `features/split_train_val_test.csv`. |
| `--test-final` | `reportes/modelo/reporte_test_final.txt` + **gráficos en `reportes/modelo/<fecha_hora>/`**. |

---

## Entrenamiento del Random Forest

- **Hiperparámetros editables:** se ajustan en el archivo de texto
  `config_entrenamiento.txt` (en la raíz). `-rf` lo lee al entrenar; si no existe,
  se crea solo con los valores de referencia. Para experimentar, edita el archivo y
  vuelve a correr `python main.py -rf`. Los flags de CLI (`--n-estimators`, etc.)
  **sobreescriben** lo del archivo para una corrida puntual (útil en el loop).
  Para volver al punto de partida: `python main.py --reset-config` (restaura los
  valores de referencia, definidos en `src/config.py`). El archivo **se versiona**
  (se sube a GitHub) porque es parte de la configuración del entrenamiento; la
  referencia para `--reset-config` vive en `config.py`.
- **Features:** los 20 AUs + `FaceScore` (CLAUDE.md §5.1).
- **Split 70/15/15** estratificado por clase y **determinista** (semilla fija): se
  guarda en `features/split_train_val_test.csv` y se **reutiliza** en cada corrida,
  de modo que el `test` siempre es el mismo y **no se toca** hasta `--test-final`.
- **Desbalance:** `class_weight='balanced'`.
- **Checkpoints:** durante el `fit` se guarda un modelo cada N árboles en
  `models/checkpoints/` (configurable con `RF_CHECKPOINT_CADA`).
- **Modelo final:** `models/random_forest.joblib` + metadatos en
  `random_forest_meta.json` (listo para inferencia).
- **Gráficos:** cada corrida (`-rf` y `--test-final`) crea una carpeta
  `reportes/modelo/<fecha_hora>/` con imágenes PNG:
  - `matriz_confusion_<conjunto>.png` — matriz de confusión (conteos + normalizada),
  - `metricas_por_clase_<conjunto>.png` — precision/recall/f1 por clase,
  - `importancia_features.png` — peso de cada AU/FaceScore,
  - `curva_entrenamiento.png` — accuracy de validación vs nº de árboles (solo `-rf`).

Flujo típico:

```bash
python main.py -rf                       # baseline con los defaults
python main.py -rf --n-estimators 500    # probar otra config (el loop variará esto)
python main.py --test-final              # SOLO cuando el modelo esté elegido
```
