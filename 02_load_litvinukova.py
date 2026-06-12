"""Heart Cell Atlas (Litvinukova et al., 2020) の生 h5ad を読み込み、
   著者ラベルを統合カテゴリにマッピングしたうえで
   (cell_label x region x donor) で pseudobulk を取る。"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

np.random.seed(42)

H5AD = Path("data/reference/litvinukova/Global_raw.h5ad")
OUT_DIR = Path("reports/revision_20260617/scripts/intermediate")
OUT_CELLS = OUT_DIR / "litvinukova_cells.tsv"
OUT_PB_COUNTS = OUT_DIR / "litvinukova_pseudobulk_counts.tsv"
OUT_PB_META = OUT_DIR / "litvinukova_pseudobulk_meta.tsv"

LABEL_MAP = {
    "Ventricular Cardiomyocyte": "CM",
    "Atrial Cardiomyocyte": "CM",
    "Fibroblast": "FB",
    "Endothelial cell": "EC",
    "Lymphatic Endothelial cell": "EC",
    "Mural cell": "Mural",
    "Myeloid": "Immune",
    "Lymphoid": "Immune",
    "Mast cell": "Immune",
    "Neural cell": "Neural",
    "Mesothelial cell": "Epicardial",
    "Adipocyte": "Other",
}

# CM だけは著者ラベル上で aCM/vCM が区別されているため、サブタイプを保持する
AV_MAP = {
    "Ventricular Cardiomyocyte": "vCM",
    "Atrial Cardiomyocyte": "aCM",
}

MIN_CELLS = 20
EXCLUDED = {"Other"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {H5AD} (backed=r) ...")
    a = ad.read_h5ad(H5AD, backed="r")
    print(f"shape = {a.shape}")

    obs = a.obs[["cell_type", "region", "donor"]].copy()
    obs.index.name = "barcode"
    obs["unified"] = obs["cell_type"].map(LABEL_MAP)
    obs["av_subtype"] = obs["cell_type"].map(AV_MAP).fillna("")
    obs["cell_label"] = np.where(obs["av_subtype"] != "",
                                 obs["av_subtype"], obs["unified"])

    unknown = obs.loc[obs["unified"].isna(), "cell_type"].unique().tolist()
    if unknown:
        raise RuntimeError(f"unmapped cell_type labels: {unknown}")

    obs.reset_index().to_csv(OUT_CELLS, sep="\t", index=False)
    print(f"wrote cell-level annotation: {OUT_CELLS} ({len(obs):,} cells)")

    keep_mask = ~obs["unified"].isin(EXCLUDED)
    obs_keep = obs[keep_mask].copy()
    print(f"keep {len(obs_keep):,} / {len(obs):,} cells "
          f"(excluded {(~keep_mask).sum():,})")

    obs_keep["group_key"] = (obs_keep["cell_label"].astype(str) + "|"
                             + obs_keep["region"].astype(str) + "|"
                             + obs_keep["donor"].astype(str))
    obs_keep["pos"] = np.arange(a.n_obs)[keep_mask.values]

    group_info = (
        obs_keep.groupby("group_key", observed=True)
                .agg(cell_label=("cell_label", "first"),
                     region=("region", "first"),
                     donor=("donor", "first"),
                     n_cells=("pos", "size"))
                .reset_index()
    )
    valid = group_info[group_info["n_cells"] >= MIN_CELLS].copy()
    print(f"pseudobulk groups: {len(group_info)} -> {len(valid)} "
          f"(min {MIN_CELLS} cells)")

    gene_symbols = a.var["gene_name-new"].copy()
    gene_symbols.index = a.var.index

    pos_lookup = (
        obs_keep.groupby("group_key", observed=True)["pos"]
                .apply(lambda s: s.to_numpy()).to_dict()
    )

    X = a.X
    n_groups = len(valid)
    pb_arr = np.zeros((n_groups, a.shape[1]), dtype=np.float64)

    for i, gk in enumerate(valid["group_key"].tolist()):
        sub = X[pos_lookup[gk], :]
        if sp.issparse(sub):
            pb_arr[i, :] = np.asarray(sub.sum(axis=0)).ravel()
        else:
            pb_arr[i, :] = np.asarray(sub).sum(axis=0)
        if (i + 1) % 50 == 0 or i == n_groups - 1:
            print(f"  pseudobulk {i + 1}/{n_groups}")

    # ENSG -> gene symbol に集約。同一 symbol に複数 ENSG が落ちる場合は和を取る。
    pb_df = pd.DataFrame(pb_arr.T, index=gene_symbols.values,
                         columns=valid["group_key"].tolist())
    pb_df.index.name = "gene_symbol"
    n_before = pb_df.shape[0]
    pb_df = pb_df.groupby(level=0).sum()
    print(f"gene aggregation: {n_before:,} -> {pb_df.shape[0]:,} symbols")

    sample_ids = [f"A_{r['cell_label']}_{r['region']}_{r['donor']}"
                  for _, r in valid.iterrows()]
    pb_df.columns = sample_ids
    pb_df.to_csv(OUT_PB_COUNTS, sep="\t")

    valid["sample_id"] = sample_ids
    valid["dataset"] = "A"
    valid[["sample_id", "dataset", "cell_label", "region", "donor",
           "n_cells"]].to_csv(OUT_PB_META, sep="\t", index=False)

    print(f"wrote pseudobulk counts: {OUT_PB_COUNTS} "
          f"({pb_df.shape[0]:,} x {pb_df.shape[1]})")
    print("groups per cell_label:")
    print(valid.groupby("cell_label", observed=True).size().to_string())

    a.file.close()


if __name__ == "__main__":
    main()
