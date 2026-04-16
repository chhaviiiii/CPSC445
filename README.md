# CPSC 445 — Ovarian cancer scRNA-seq (Vázquez-García et al.)

Reproduction and extension pipeline for a **CellxGene** single-cell dataset: QC → normalization → HVGs → PCA / **Harmony** → UMAP → Leiden, plus a **scVI** latent space and comparison metrics.

## Repository layout

| Path | Description |
|------|-------------|
| `01_preprocess.py` | Download (first dataset in collection), QC, PCA UMAP, Leiden, markers, batch UMAP panel |
| `02_scvi_extension.py` | scVI training, PCA vs scVI UMAP, **PCA + Harmony** baseline, ARI / silhouette table |
| `project_utils.py` | Shared helpers (e.g. `coarse_label` from `cluster_label`) |
| `environment.yml` | Conda environment `ov_scrna` |
| `figures/` | Generated figures (UMAPs, dot plots, metrics bars, etc.) |
| `data/` | Small CSV outputs; large `.h5ad` files are **not** tracked (see below) |

## Setup

```bash
conda env create -f environment.yml
conda activate ov_scrna
```

## Run

From the project root:

```bash
python 01_preprocess.py   # Week 1: download → QC → normalize → UMAP → clustering
python 02_scvi_extension.py  # Week 2: scVI + Harmony baseline + metrics
```

**Environment:** use the `ov_scrna` conda environment so imports (`pandas`, `scanpy`, `scvi`, etc.) resolve. Do not rely on the system/base Python.

## Outputs

- **Preprocessed object:** `data/adata_preprocessed.h5ad` (created by `01_preprocess.py`; gitignored)
- **scVI object / model:** `data/adata_scvi.h5ad`, `data/scvi_model/` (gitignored)
- **Tables:** `data/pca_vs_scvi_metrics.csv`, `data/cluster_markers.csv` (tracked if committed)
- **Figures:** under `figures/` (e.g. `pca_vs_scvi_umap.png`, `umap_batch_effect_panel.png`)

## Data and Git

Large files are excluded via `.gitignore` (`*.h5ad`, `data/scvi_model/`). Clone the repo and run `01_preprocess.py` to download and regenerate local AnnData files.

## Scope note

The CellxGene **collection** can contain multiple `.h5ad` files (e.g. by compartment); this pipeline uses the **first** listed dataset unless you change the download index in `01_preprocess.py`. Coarse annotation **`cell_type_super`** may be a single class in a compartment-specific file; coarse ARI uses **`coarse_label`** derived from **`cluster_label`** (CD4 / CD8 / NK / Cycling / ILC prefixes).

## License / citation

Use the dataset and publication terms from [CELLxGENE](https://cellxgene.cziscience.com/) and the original paper. This repository contains course/analysis code only.
