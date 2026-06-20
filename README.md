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
| `-r` / `--remove` | Modificador: **elimina del disco** lo detectado por las etapas activas. Sin `-r` no se borra nada. |

> Las flags cortas se pueden agrupar: `-lr` ≡ `-l -r`, `-ir` ≡ `-i -r`, `-dr` ≡ `-d -r`.

### Ejemplos

```bash
python main.py -i        # tarea 1: escanear integridad (no borra)
python main.py -ir       # tarea 1: escanear y eliminar
python main.py -d        # tarea 2: detectar duplicados (no borra)
python main.py -dr       # tarea 2: detectar y eliminar duplicados
python main.py -l        # tarea 3: luminancia/contraste (no borra)
python main.py -lr       # tarea 3: escanear y eliminar fuera de norma
python main.py -i -d     # varias etapas en una corrida (sin borrar)
python main.py           # sin opciones: muestra esta ayuda
python main.py -h        # igual
```

> ⚠️ Agregar `-r` **borra archivos del disco** de forma irreversible. Conviene
> correr primero la etapa sin `-r` y revisar el reporte.

---

## Reportes generados

| Tarea | Archivo |
|-------|---------|
| `-i` | `reportes/integridad/reporte_integridad.txt` — solo se crea si hay imágenes descartadas. |
| `-d` | `reportes/duplicados/reporte_duplicados.txt` — solo se crea si hay grupos de duplicados. |
| `-l` | `reportes/luminancia/reporte_luminancia.txt` — siempre se genera (incluye la distribución global y la lista fuera de norma). |
