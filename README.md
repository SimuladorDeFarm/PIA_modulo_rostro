# Módulo de Rostro — Comandos del pipeline QC

Ejecutar desde la raíz del proyecto con el entorno virtual activo.

```bash
source .venv/bin/activate
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

## Reportes generados

| Tarea | Archivo |
|-------|---------|
| `-i` | `reportes/integridad/reporte_integridad.txt` — solo se crea si hay imágenes descartadas. |
| `-d` | `reportes/duplicados/reporte_duplicados.txt` — solo se crea si hay grupos de duplicados. |
| `-l` | `reportes/luminancia/reporte_luminancia.txt` — siempre se genera (incluye la distribución global y la lista fuera de norma). |
| `-e` | `reportes/reescalado/reporte_reescalado.txt` — siempre se genera. Dataset de salida: `AffNet_128/`. |
| `-p` | `reportes/pyfeat/reporte_pyfeat.txt` + CSV de features `features/pyfeat_v5.csv`. |
