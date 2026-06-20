"""Módulo de Rostro — pipeline de control de calidad (QC) de AffectNet.

Cada etapa del QC (ver CLAUDE.md, sección 5.3) vive en `src/qc/` como un paso
reproducible e independiente. La carga del dataset se centraliza en `dataset.py`
y la configuración (rutas, constantes) en `config.py`.
"""
