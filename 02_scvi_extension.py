"""
02_scvi_extension.py
Vázquez-García et al., Nature 2022 — scVI Extension (Week 2)
Swaps PCA with a VAE latent space and compares cluster quality.

Prerequisite: run ``01_preprocess.py`` first to create ``data/adata_preprocessed.h5ad``.

This script:
  (1) Trains scVI on raw counts (``layers['counts']``) with patient as batch when available.
  (2) Builds UMAP + Leiden from the scVI latent space.
  (3) Reloads the PCA object for a fair PCA vs scVI UMAP figure (fresh read = clean obsm).
  (4) Runs Harmony on ``X_pca`` (see ``harmony_corrected_pca``) → neighbors → UMAP → Leiden.
  (5) Writes ``data/pca_vs_scvi_metrics.csv``: coarse ARI (``coarse_label``), granular ARI
      (``cluster_label``), silhouette for PCA / PCA+Harmony / scVI.

ARI: coarse lineage is derived in ``project_utils.ensure_coarse_label``; do not use
``cell_type_super`` alone when the h5ad is T-cell-only (single class).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scvi
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score

from project_utils import ensure_coarse_label

warnings.filterwarnings("ignore")


def harmony_corrected_pca(adata, batch_key: str, basis: str = "X_pca", out_key: str = "X_pca_harmony"):
    """Run harmonypy and store (n_obs × n_pc) in obsm.

    Scanpy's ``sc.external.pp.harmony_integrate`` assigns ``Z_corr.T``. Recent
    harmonypy releases return ``Z_corr`` already shaped (cells × PCs), so the
    transpose yields an invalid obsm; we normalize orientation here.
    """
    import harmonypy

    x = np.asarray(adata.obsm[basis], dtype=np.float64)
    harmony_out = harmonypy.run_harmony(x, adata.obs, batch_key)
    z = np.asarray(harmony_out.Z_corr)
    if z.shape[0] != adata.n_obs and z.shape[1] == adata.n_obs:
        z = z.T
    if z.shape[0] != adata.n_obs:
        raise ValueError(
            f"Harmony output shape {z.shape} incompatible with n_obs={adata.n_obs}"
        )
    adata.obsm[out_key] = z


# ── Config ────────────────────────────────────────────────────────────────────
PREPROCESSED_PATH = "data/adata_preprocessed.h5ad"
SEED              = 42
N_LATENT          = 10  # scVI bottleneck size (latent dim)
N_LAYERS          = 2
N_HIDDEN          = 128
PAPER_RESOLUTION  = 0.5   # keep identical to 01_preprocess for Leiden comparison
RESOLUTIONS       = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
MAX_EPOCHS        = 400

scvi.settings.seed = SEED
np.random.seed(SEED)
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=100, facecolor="white", figsize=(6, 5))

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print(f"scvi-tools {scvi.__version__}  |  PyTorch {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")


# ── 1. Load preprocessed data ─────────────────────────────────────────────────
adata = sc.read_h5ad(PREPROCESSED_PATH)
print(f"\nLoaded: {adata.n_obs} cells × {adata.n_vars} genes")

# Coarse labels for ARI (from cluster_label prefixes); idempotent if 01 already saved them.
ensure_coarse_label(adata)
COARSE_REF_COL = "coarse_label" if "coarse_label" in adata.obs.columns else None

adata_scvi = adata.copy()

# scVI fits on raw counts (NB likelihood); normalized matrix in .X is not used for training here.
if "counts" not in adata_scvi.layers:
    print("counts layer not found — using adata.raw.X")
    adata_scvi.layers["counts"] = adata_scvi.raw.X.copy()

# Identify columns
cell_type_cols = [c for c in adata_scvi.obs.columns
                  if any(k in c.lower() for k in ["cell_type", "annotation", "label"])]
batch_col = next((c for c in ["patient_id", "patient", "donor_id", "donor", "sample_id", "batch"]
                  if c in adata_scvi.obs.columns), None)
ct_col = cell_type_cols[0] if cell_type_cols else None
# ARI vs. paper: use fine-grained labels (cluster_label). cell_type_super is often
# one value per split h5ad (e.g. all T.super) and yields ARI = 0.
ARI_REF_COL = (
    "cluster_label"
    if "cluster_label" in adata_scvi.obs.columns
    else ct_col
)

print(f"Batch col: {batch_col}  |  Cell type col: {ct_col}")
print(f"ARI reference labels: granular=`{ARI_REF_COL}`, coarse=`{COARSE_REF_COL}`")

GENE_SYM_COL = next((c for c in ("feature_name", "gene_name") if c in adata_scvi.var.columns), None)


# ── 2. Set up and train scVI ──────────────────────────────────────────────────
# batch_key: scVI learns patient-specific scaling / mixing; same key as Harmony when possible.
scvi.model.SCVI.setup_anndata(
    adata_scvi,
    layer="counts",
    batch_key=batch_col,
)

model = scvi.model.SCVI(
    adata_scvi,
    n_latent=N_LATENT,
    n_layers=N_LAYERS,
    n_hidden=N_HIDDEN,
    gene_likelihood="nb",       # negative binomial — well-suited for UMI counts
    dispersion="gene",
)
print(model)

model.train(
    max_epochs=MAX_EPOCHS,
    early_stopping=True,
    early_stopping_patience=20,
    batch_size=512,
    plan_kwargs={"lr": 1e-3},
)
print("Training complete.")

# Training curve
train_elbo = model.history["elbo_train"]
val_elbo   = model.history["elbo_validation"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_elbo, label="Train ELBO")
ax.plot(val_elbo,   label="Val ELBO", linestyle="--")
ax.set(xlabel="Epoch", ylabel="ELBO", title="scVI Training Curve")
ax.legend()
plt.tight_layout()
plt.savefig("figures/scvi_training_curve.png", dpi=150, bbox_inches="tight")
plt.close()

model.save("data/scvi_model/", overwrite=True)
print("Model saved to data/scvi_model/")


# ── 3. Latent space → UMAP → Leiden ──────────────────────────────────────────
# Neighbors on latent space, not on PCA — this is the “scVI embedding” analysis path.
adata_scvi.obsm["X_scVI"] = model.get_latent_representation()
print(f"Latent shape: {adata_scvi.obsm['X_scVI'].shape}")

sc.pp.neighbors(adata_scvi, use_rep="X_scVI", n_neighbors=15, random_state=SEED)
sc.tl.umap(adata_scvi, min_dist=0.3, random_state=SEED)

sc.tl.leiden(adata_scvi, resolution=PAPER_RESOLUTION, random_state=SEED, key_added="leiden_scvi")
print(f"scVI Leiden (res={PAPER_RESOLUTION}): {adata_scvi.obs['leiden_scvi'].nunique()} clusters")


# ── 4. Side-by-side UMAP comparison ──────────────────────────────────────────
# Second load: keeps PCA UMAP / leiden_paper_res from 01 without scVI-side mutations.
adata_pca = sc.read_h5ad(PREPROCESSED_PATH)
ensure_coarse_label(adata_pca)
adata_harmony = adata_pca.copy()

# Harmony adjusts batch effects in PC space; graph + UMAP are recomputed from X_pca_harmony.
if batch_col and batch_col in adata_harmony.obs.columns:
    harmony_corrected_pca(adata_harmony, batch_key=batch_col, basis="X_pca")
    sc.pp.neighbors(adata_harmony, use_rep="X_pca_harmony", n_neighbors=15, random_state=SEED)
    sc.tl.umap(adata_harmony, min_dist=0.3, random_state=SEED)
    sc.tl.leiden(
        adata_harmony,
        resolution=PAPER_RESOLUTION,
        random_state=SEED,
        key_added="leiden_harmony",
    )
    print(
        f"PCA+Harmony Leiden (res={PAPER_RESOLUTION}): "
        f"{adata_harmony.obs['leiden_harmony'].nunique()} clusters"
    )
else:
    adata_harmony = None
    print("No batch column found; skipping PCA+Harmony branch.")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc.pl.umap(adata_pca, color="leiden_paper_res", ax=axes[0], show=False,
           title=f"PCA → UMAP ({adata_pca.obs['leiden_paper_res'].nunique()} clusters)",
           legend_loc="on data", legend_fontsize=6)
sc.pl.umap(adata_scvi, color="leiden_scvi", ax=axes[1], show=False,
           title=f"scVI → UMAP ({adata_scvi.obs['leiden_scvi'].nunique()} clusters)",
           legend_loc="on data", legend_fontsize=6)
plt.suptitle("PCA vs. scVI Latent Space", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("figures/pca_vs_scvi_umap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/pca_vs_scvi_umap.png")


# ── 5. Quantitative evaluation (ARI + silhouette) ────────────────────────────
def _ari_against(obs_df, cluster_col, ref_col):
    """Adjusted Rand index between Leiden (cluster_col) and paper labels (ref_col)."""
    if (not ref_col) or (ref_col not in obs_df.columns):
        return np.nan
    ref = obs_df[ref_col].astype("category")
    if ref.nunique() < 2:
        return np.nan
    ref_codes = ref.cat.codes.values
    clusters = obs_df[cluster_col].astype(int).values
    return adjusted_rand_score(ref_codes, clusters)


# One table row per method; silhouette uses the same subsample size for comparability.
rows = []
pca_labels = adata_pca.obs["leiden_paper_res"].astype(int).values
rows.append(
    {
        "Method": "PCA (reproduction)",
        "N_clusters": adata_pca.obs["leiden_paper_res"].nunique(),
        "ARI_coarse": _ari_against(adata_pca.obs, "leiden_paper_res", COARSE_REF_COL),
        "ARI_granular": _ari_against(adata_pca.obs, "leiden_paper_res", ARI_REF_COL),
        # First 30 PCs — same dimensionality cap as typical neighbors usage in 01.
        "Silhouette": silhouette_score(
            adata_pca.obsm["X_pca"][:, :30], pca_labels, sample_size=5000, random_state=SEED
        ),
    }
)

if adata_harmony is not None:
    harmony_labels = adata_harmony.obs["leiden_harmony"].astype(int).values
    rows.append(
        {
            "Method": "PCA + Harmony",
            "N_clusters": adata_harmony.obs["leiden_harmony"].nunique(),
            "ARI_coarse": _ari_against(adata_harmony.obs, "leiden_harmony", COARSE_REF_COL),
            "ARI_granular": _ari_against(adata_harmony.obs, "leiden_harmony", ARI_REF_COL),
            "Silhouette": silhouette_score(
                adata_harmony.obsm["X_pca_harmony"],
                harmony_labels,
                sample_size=5000,
                random_state=SEED,
            ),
        }
    )

scvi_labels = adata_scvi.obs["leiden_scvi"].astype(int).values
rows.append(
    {
        "Method": "scVI (extension)",
        "N_clusters": adata_scvi.obs["leiden_scvi"].nunique(),
        "ARI_coarse": _ari_against(adata_scvi.obs, "leiden_scvi", COARSE_REF_COL),
        "ARI_granular": _ari_against(adata_scvi.obs, "leiden_scvi", ARI_REF_COL),
        "Silhouette": silhouette_score(
            adata_scvi.obsm["X_scVI"], scvi_labels, sample_size=5000, random_state=SEED
        ),
    }
)

results = pd.DataFrame(rows)
for col in ["ARI_coarse", "ARI_granular", "Silhouette"]:
    results[col] = results[col].round(4)
print(
    f"\nARI columns use coarse `{COARSE_REF_COL}` and granular `{ARI_REF_COL}` "
    "(NaN means reference had <2 classes)."
)
print("\n" + results.to_string(index=False))
print(
    "\n[Report — Table 1 template] Fill with the values above:\n"
    "Method | Coarse ARI | Granular ARI | Silhouette\n"
    "PCA | ... | ... | ...\n"
    "PCA + Harmony | ... | ... | ...\n"
    "scVI | ... | ... | ...\n"
)
pca_row = results[results["Method"] == "PCA (reproduction)"]
harm_row = results[results["Method"] == "PCA + Harmony"]
if len(pca_row) and len(harm_row):
    g0 = pca_row["ARI_granular"].iloc[0]
    g1 = harm_row["ARI_granular"].iloc[0]
    if not (pd.isna(g0) or pd.isna(g1)) and float(g0) == float(g1):
        print(
            "[Report — Discussion] PCA and PCA+Harmony granular ARI match: Harmony "
            "adjusts batch geometry in PC space but Leiden assignments at this "
            "resolution can remain unchanged — worth stating explicitly.\n"
        )
results.to_csv("data/pca_vs_scvi_metrics.csv", index=False)

# Three panels: coarse ARI (lineage), granular ARI (paper subtypes), internal silhouette.
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric, title in zip(
    axes,
    ["ARI_coarse", "ARI_granular", "Silhouette"],
    [
        (
            f"ARI vs. `{COARSE_REF_COL}` (coarse)"
            if COARSE_REF_COL
            else "ARI coarse (missing coarse_label)"
        ),
        f"ARI vs. `{ARI_REF_COL}` (granular)",
        "Silhouette Score (↑ better)",
    ],
):
    ax.bar(results["Method"], results[metric], color=["steelblue", "slateblue", "coral"][: len(results)])
    ax.set(ylabel=metric, title=title, ylim=(0, 1))
    ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig("figures/pca_vs_scvi_metrics.png", dpi=150, bbox_inches="tight")
plt.close()


# ── 6. Marker gene check (scVI clusters) ──────────────────────────────────────
MARKER_GENES = {
    "Tumor":       ["EPCAM", "KRT8", "KRT18", "PAX8"],
    "CD8 T cell":  ["CD8A", "CD8B", "GZMB"],
    "Macrophage":  ["CD68", "CSF1R", "LYZ"],
    "Fibroblast":  ["COL1A1", "COL1A2", "DCN"],
    "Endothelial": ["PECAM1", "VWF"],
}
if GENE_SYM_COL is not None:
    _sym_set = set(adata_scvi.raw.var[GENE_SYM_COL].astype(str))
    avail = {ct: [g for g in genes if g in _sym_set] for ct, genes in MARKER_GENES.items()}
else:
    avail = {ct: [g for g in genes if g in adata_scvi.raw.var_names] for ct, genes in MARKER_GENES.items()}
avail = {ct: g for ct, g in avail.items() if g}

_dot_kw = dict(
    title="Marker genes per scVI cluster",
    use_raw=True,
    standard_scale="var",
    save="_scvi_marker_dotplot.png",
    show=False,
)
if GENE_SYM_COL is not None:
    _dot_kw["gene_symbols"] = GENE_SYM_COL
sc.pl.dotplot(adata_scvi, var_names=avail, groupby="leiden_scvi", **_dot_kw)


# ── 7. Resolution sweep on scVI (same grid as 01 for side-by-side reporting) ──
for res in RESOLUTIONS:
    sc.tl.leiden(adata_scvi, resolution=res, random_state=SEED, key_added=f"leiden_scvi_res_{res}")
    print(f"  scVI res={res} → {adata_scvi.obs[f'leiden_scvi_res_{res}'].nunique()} clusters")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
for i, res in enumerate(RESOLUTIONS):
    key = f"leiden_scvi_res_{res}"
    sc.pl.umap(adata_scvi, color=key, ax=axes[i], show=False,
               title=f"scVI res={res} ({adata_scvi.obs[key].nunique()} clusters)",
               legend_loc="on data", legend_fontsize=5)
plt.suptitle("scVI — Resolution Sweep", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("figures/scvi_resolution_sweep.png", dpi=150, bbox_inches="tight")
plt.close()


# ── 8. Save ───────────────────────────────────────────────────────────────────
adata_scvi.write_h5ad("data/adata_scvi.h5ad", compression="gzip")
print("\nSaved data/adata_scvi.h5ad")
print(adata_scvi)
