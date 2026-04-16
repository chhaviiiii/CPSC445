"""Shared helpers for the CPSC 445 scRNA-seq pipeline.

Used by ``01_preprocess.py`` and ``02_scvi_extension.py`` so notebook CWD does not matter.
The important bit is ``ensure_coarse_label``: CellxGene h5ads often have one value in
``cell_type_super`` per file, so coarse agreement with paper labels must be derived from
``cluster_label`` string prefixes instead.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from anndata import AnnData

PROJECT_ROOT = Path(__file__).resolve().parent

# Prefixes in SPECTRUM / paper `cluster_label` strings (e.g. CD8.T.cytotoxic → CD8)
COARSE_LABEL_PATTERN = r"^(CD4|CD8|NK|Cycling|ILC)"


def resolve_preprocessed_h5ad() -> Path:
    return PROJECT_ROOT / "data" / "adata_preprocessed.h5ad"


def ensure_coarse_label(adata: AnnData, *, cluster_col: str = "cluster_label", out_col: str = "coarse_label") -> bool:
    """Derive coarse lineage labels from fine-grained ``cluster_col``.

    ``cell_type_super`` is often a single value per split h5ad (e.g. T.super); coarse ARI
    should use this column instead. Non-matching labels are set to ``\"Other\"``.

    Returns True if ``out_col`` already exists or was created; False if ``cluster_col`` is missing.
    """
    if cluster_col not in adata.obs.columns:
        return False
    if out_col in adata.obs.columns:
        return True
    extracted = adata.obs[cluster_col].astype(str).str.extract(COARSE_LABEL_PATTERN, expand=False)
    if isinstance(extracted, pd.DataFrame):
        extracted = extracted.iloc[:, 0]
    adata.obs[out_col] = extracted.fillna("Other").astype("category")
    return True
