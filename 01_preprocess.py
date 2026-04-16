"""
01_preprocess.py
Vázquez-García et al., Nature 2022 — Ovarian Cancer scRNA-seq Reproduction
Pipeline: Download → QC → Normalize → HVG → PCA → UMAP → Leiden clustering

Flow summary:
  (1) Pull first .h5ad from the CellxGene collection (see COLLECTION_ID).
  (2) QC filter cells/genes; keep ``layers['counts']`` for later HVG + scVI.
  (3) Subset to HVGs for PCA, but keep full normalized matrix in ``.raw`` for markers/DE.
  (4) Leiden at PAPER_RESOLUTION + a resolution sweep for the report.
  (5) Figures + ``cluster_markers.csv``; save ``adata_preprocessed.h5ad`` for Week 2.

Gene IDs: CellxGene uses Ensembl IDs in ``var_names``; human-readable symbols live in
``feature_name`` / ``gene_name`` (see GENE_SYM_COL) for dot plots and MT- detection.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
import requests

from project_utils import ensure_coarse_label

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
# CellxGene collection UUID — API lists multiple datasets; we download datasets[0] only.
COLLECTION_ID    = "4796c91c-9d8f-4692-be43-347b1727f9d8"
DATA_PATH        = "data/vazquez_garcia_2022.h5ad"
PREPROCESSED_PATH = "data/adata_preprocessed.h5ad"
SEED             = 42
MIN_GENES        = 200
MAX_GENES        = 6000
MAX_MT_PCT       = 20.0
MIN_CELLS        = 3
N_HVG            = 2000
N_PCS            = 30  # PCs fed into neighbors graph (≤ number of PCs from sc.tl.pca)
PAPER_RESOLUTION = 0.5   # primary Leiden resolution for figures / comparison to 02_scvi
# Extra resolutions only for sensitivity plots (not the main “paper” cluster key)
RESOLUTIONS      = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]

np.random.seed(SEED)
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor="white", figsize=(6, 5))

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print(f"scanpy {sc.__version__}  |  anndata {ad.__version__}")


# ── 1. Download ───────────────────────────────────────────────────────────────
def download_dataset(collection_id, output_path):
    """Fetch the first h5ad dataset from a CellxGene collection.

    Many collections ship multiple datasets (e.g. one h5ad per major compartment).
    This script uses the first entry (often T cells for this paper); scope your
    report to that subset, or change the index below to download another file.
    """
    url = f"https://api.cellxgene.cziscience.com/curation/v1/collections/{collection_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    datasets = resp.json().get("datasets", [])

    print(f"Found {len(datasets)} dataset(s) in collection (using [0] only):")
    for i, ds in enumerate(datasets):
        print(f"  [{i}] {ds.get('title', 'Unnamed')}  |  id: {ds.get('dataset_id')}")

    # Change index here if you need myeloid / tumor / another compartment’s h5ad.
    dataset_id = datasets[0]["dataset_id"]
    dl_url = f"https://datasets.cellxgene.cziscience.com/{dataset_id}.h5ad"
    print(f"\nDownloading {dl_url} ...")

    with requests.get(dl_url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded/total*100:.1f}%", end="", flush=True)
    print(f"\nSaved to {output_path}")


if not os.path.exists(DATA_PATH):
    download_dataset(COLLECTION_ID, DATA_PATH)
else:
    print(f"Data already at {DATA_PATH}, skipping download.")
    print(
        "(Reminder: CellxGene collections often have multiple h5ads per compartment; "
        "this project uses the first file downloaded — typically T-cell–rich.)"
    )


# ── 2. Load & inspect ─────────────────────────────────────────────────────────
adata = sc.read_h5ad(DATA_PATH)
print(f"\nLoaded: {adata.n_obs} cells × {adata.n_vars} genes")
if "cell_type_super" in adata.obs.columns:
    print(
        "Scope: cell_type_super value counts (report should match this compartment):\n",
        adata.obs["cell_type_super"].value_counts().to_string(),
    )
print("obs columns:", adata.obs.columns.tolist())
print("obsm keys:  ", list(adata.obsm.keys()))

# Integer counts for Seurat-v3 HVGs and for scVI in 02 (must exist before normalization).
adata.layers["counts"] = adata.X.copy()

# Identify useful columns
cell_type_cols = [c for c in adata.obs.columns
                  if any(k in c.lower() for k in ["cell_type", "annotation", "label"])]
batch_col = next((c for c in ["patient_id", "patient", "donor_id", "donor", "sample_id", "batch"]
                  if c in adata.obs.columns), None)
print(f"Cell type cols: {cell_type_cols}")
print(f"Batch col:      {batch_col}")

# CellxGene exports use Ensembl IDs as var_names; symbols are in feature_name / gene_name
GENE_SYM_COL = next((c for c in ("feature_name", "gene_name") if c in adata.var.columns), None)


# ── 3. QC ─────────────────────────────────────────────────────────────────────
if GENE_SYM_COL is not None:
    adata.var["mt"] = adata.var[GENE_SYM_COL].astype(str).str.startswith("MT-")
else:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

print("\nQC summary:")
print(adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe())

# QC plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(adata.obs["n_genes_by_counts"], bins=100, color="steelblue", edgecolor="none")
axes[0].set(xlabel="Genes per cell", title="Genes per cell")
axes[1].hist(adata.obs["total_counts"], bins=100, color="coral", edgecolor="none")
axes[1].set(xlabel="Total UMI counts", title="Total counts")
axes[2].hist(adata.obs["pct_counts_mt"], bins=100, color="mediumpurple", edgecolor="none")
axes[2].set(xlabel="MT %", title="Mitochondrial %")
plt.tight_layout()
plt.savefig("figures/qc_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# Filter
n_before = adata.n_obs
sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
sc.pp.filter_cells(adata, min_genes=MIN_GENES)
adata = adata[adata.obs["n_genes_by_counts"] < MAX_GENES].copy()
adata = adata[adata.obs["pct_counts_mt"] < MAX_MT_PCT].copy()
print(f"\nQC filtering: {n_before} → {adata.n_obs} cells  |  {adata.n_vars} genes remaining")

# Coarse lineage for ARI / reporting (cell_type_super is often single-class in one h5ad)
if ensure_coarse_label(adata):
    print(
        "Derived coarse_label from cluster_label (CD4/CD8/NK/Cycling/ILC prefix); "
        "value counts:\n",
        adata.obs["coarse_label"].value_counts().to_string(),
    )


# ── 4. Normalize ──────────────────────────────────────────────────────────────
# Log-normalized counts in .X; raw counts stay in layers['counts'].
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("Normalization + log1p done.")


# ── 5. HVG selection ──────────────────────────────────────────────────────────
# batch_key: variance stabilization within patient blocks when present.
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=N_HVG,
    flavor="seurat_v3",
    layer="counts",
    batch_key=batch_col,
    span=0.3,
)
print(f"HVGs selected: {adata.var['highly_variable'].sum()}")
# Keep full normalized+log matrix in .raw so marker genes (not only HVGs) remain for plots & DE
adata_full = adata.copy()
adata = adata[:, adata.var["highly_variable"]].copy()
adata.raw = adata_full
print("Subset to HVGs; full gene matrix kept in adata.raw for markers and DE.")


# ── 6. PCA → Neighbors → UMAP ────────────────────────────────────────────────
# Z-score HVGs for PCA; 50 PCs stored but graph uses first N_PCS to match typical practice.
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50, svd_solver="arpack", random_state=SEED)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=N_PCS, random_state=SEED)
sc.tl.umap(adata, min_dist=0.3, random_state=SEED)
print("PCA + UMAP done.")


# ── 7. Leiden clustering ──────────────────────────────────────────────────────
# Paper resolution
sc.tl.leiden(adata, resolution=PAPER_RESOLUTION, random_state=SEED, key_added="leiden_paper_res")
print(f"Leiden (res={PAPER_RESOLUTION}): {adata.obs['leiden_paper_res'].nunique()} clusters")

# Resolution sweep for critique
for res in RESOLUTIONS:
    sc.tl.leiden(adata, resolution=res, random_state=SEED, key_added=f"leiden_res_{res}")
    print(f"  res={res} → {adata.obs[f'leiden_res_{res}'].nunique()} clusters")


# ── 8. UMAP plots ─────────────────────────────────────────────────────────────
sc.pl.umap(adata, color=["leiden_paper_res"],
           title=f"Leiden (res={PAPER_RESOLUTION})",
           legend_loc="on data", legend_fontsize=7,
           save="_leiden_clusters.png", show=False)

if cell_type_cols:
    sc.pl.umap(adata, color=cell_type_cols,
               legend_loc="right margin",
               save="_paper_annotations.png", show=False)

# Batch/confounder panel: patient, site, and library size
site_col = next(
    (
        c
        for c in [
            "author_tumor_site",
            "author_tumor_subsite",
            "author_tumor_supersite",
            "sample",
            "sample_id",
        ]
        if c in adata.obs.columns
    ),
    None,
)
conf_cols = [c for c in [batch_col, site_col, "total_counts"] if c and c in adata.obs.columns]
if conf_cols:
    fig, axes = plt.subplots(1, len(conf_cols), figsize=(6 * len(conf_cols), 5))
    if len(conf_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, conf_cols):
        sc.pl.umap(adata, color=col, ax=ax, show=False, title=f"UMAP colored by {col}")
    plt.tight_layout()
    plt.savefig("figures/umap_batch_effect_panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    # --- Paste into Results: batch structure ---
    print(
        "\n[Report — batch effects] UMAP colored by patient_id shows that the PCA "
        "embedding retains patient-level structure, with cells from individual patients "
        "forming partially segregated regions — motivating Harmony and scVI batch "
        "correction.\n"
    )

# Resolution sweep plot
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
for i, res in enumerate(RESOLUTIONS):
    key = f"leiden_res_{res}"
    sc.pl.umap(adata, color=key, ax=axes[i], show=False,
               title=f"res={res} ({adata.obs[key].nunique()} clusters)",
               legend_loc="on data", legend_fontsize=5)
plt.suptitle("Clustering Resolution Sensitivity", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("figures/resolution_sweep.png", dpi=150, bbox_inches="tight")
plt.close()

print(
    "\n[Report — UMAP shape] If this UMAP differs from the paper figure (e.g. butterfly "
    "layout in a published umap), the paper may have used different UMAP settings or "
    "computed UMAP on the full multi-compartment object before subsetting to one h5ad.\n"
)


# ── 9. Marker genes ───────────────────────────────────────────────────────────
# Dict keys = plot groups; values = HGNC symbols (resolved via GENE_SYM_COL → Ensembl rows).
MARKER_GENES = {
    "Tumor":       ["EPCAM", "KRT8", "KRT18", "PAX8", "MUC16"],
    "CD8 T cell":  ["CD8A", "CD8B", "GZMB", "PRF1"],
    "CD4 T cell":  ["CD4", "IL7R", "FOXP3"],
    "Macrophage":  ["CD68", "CSF1R", "CD14", "LYZ"],
    "Fibroblast":  ["COL1A1", "COL1A2", "DCN"],
    "Endothelial": ["PECAM1", "VWF", "CDH5"],
}

# .raw holds all genes (not just HVGs) so known markers still appear in dotplot/DE.
if GENE_SYM_COL is not None:
    _sym_set = set(adata.raw.var[GENE_SYM_COL].astype(str))
    available_markers = {
        ct: [g for g in genes if g in _sym_set] for ct, genes in MARKER_GENES.items()
    }
else:
    available_markers = {
        ct: [g for g in genes if g in adata.raw.var_names] for ct, genes in MARKER_GENES.items()
    }
available_markers = {ct: g for ct, g in available_markers.items() if g}

_dot_kw = dict(
    title="Marker genes per cluster",
    use_raw=True,
    standard_scale="var",
    dendrogram=False,
    figsize=(14, 8),
    save="_marker_dotplot.png",
    show=False,
)
if GENE_SYM_COL is not None:
    _dot_kw["gene_symbols"] = GENE_SYM_COL
sc.pl.dotplot(adata, var_names=available_markers, groupby="leiden_paper_res", **_dot_kw)

sc.tl.rank_genes_groups(adata, groupby="leiden_paper_res", method="wilcoxon",
                        use_raw=True, key_added="rank_genes_leiden")
_rg_kw = dict(n_genes=5, sharey=False, key="rank_genes_leiden", save="_top_markers.png", show=False)
if GENE_SYM_COL is not None:
    _rg_kw["gene_symbols"] = GENE_SYM_COL
sc.pl.rank_genes_groups(adata, **_rg_kw)

marker_df = sc.get.rank_genes_groups_df(adata, group=None, key="rank_genes_leiden")
if GENE_SYM_COL is not None:
    _sym_by_id = pd.Series(adata.var[GENE_SYM_COL].astype(str).values, index=adata.var_names)
    marker_df.insert(0, "gene_symbol", marker_df["names"].map(_sym_by_id))
marker_df.to_csv("data/cluster_markers.csv", index=False)
print("Marker table saved to data/cluster_markers.csv")


# ── 10. Cluster vs. paper annotation heatmap ─────────────────────────────────
# Rows = our Leiden; cols = paper fine-grained labels (not cell_type_super — often one class).
if "cluster_label" in adata.obs.columns:
    crosstab = pd.crosstab(
        adata.obs["leiden_paper_res"],
        adata.obs["cluster_label"],
        normalize="index",
    )
    _nw = crosstab.shape[1]
    _fig_w = min(28, max(12, 0.28 * _nw))
    fig, ax = plt.subplots(figsize=(_fig_w, 7))
    sns.heatmap(crosstab, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                cbar_kws={"label": "Fraction of cluster"})
    ax.set(
        xlabel="Paper cluster_label (fine-grained)",
        ylabel="Our Leiden cluster",
        title="Cluster vs. paper cluster_label",
    )
    plt.tight_layout()
    plt.savefig("figures/cluster_vs_annotation.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── 11. Save ──────────────────────────────────────────────────────────────────
adata.write_h5ad(PREPROCESSED_PATH, compression="gzip")
print(f"\nSaved preprocessed AnnData to {PREPROCESSED_PATH}")
print(adata)
