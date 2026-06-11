"""Cui et al. 2019 (GEO: GSE106118) の UMI 行列と
   mmc3.xlsx の著者アノテーションを統合し、
   (cell_label x region x week) で pseudobulk を取る。

   - chamber は cell ID から正規表現で取得し、Cui 本文に従って
     atrium / ventricle / valve / aorta / other に集約する。
   - 著者の Fibroblast-like cell には ACTA2/MYH11 陰性の細胞も含まれており
     A の Mural と直接対応しないため、FB に統合する。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

UMI_FILE = Path("data/reference/cui/GSE106118_UMI_count_merge.txt.gz")
MMC3 = Path("data/reference/cui/supplementary/mmc3.xlsx")
OUT_DIR = Path("reports/revision_20260617/scripts/intermediate")
OUT_CELLS = OUT_DIR / "cui_cells.tsv"
OUT_PB_COUNTS = OUT_DIR / "cui_pseudobulk_counts.tsv"
OUT_PB_META = OUT_DIR / "cui_pseudobulk_meta.tsv"

LABEL_MAP = {
    "CM": "CM",
    "Fibroblast-like cell": "FB",
    "EC": "EC",
    "EP": "Epicardial",
    "Macrophage": "Immune",
    "Mast cell": "Immune",
    "B/T cells": "Immune",
    "Valvar cell": "Other",
    "5W": "Other",
    "unlabeled": "Other",
}

CHAMBER_TO_REGION = {
    "LA": "atrium", "RA": "atrium",
    "LV": "ventricle", "RV": "ventricle",
    "EP": "ventricle", "ED": "ventricle",
    "AV": "valve", "BV": "valve", "PV": "valve", "TV": "valve",
    "AO": "aorta",
    "IS": "other_anatomical", "PO": "other_anatomical",
}

MIN_CELLS = 20
EXCLUDED = {"Other"}


def parse_cluster(s: str) -> tuple[str, str]:
    """mmc3 'Cluster' 列の "C2 (CM)" を (cluster_id, cell_type) に分割する。"""
    m = re.match(r"\s*(C\d+)\s*\((.+)\)\s*", str(s))
    if m:
        return m.group(1), m.group(2).strip()
    return str(s), str(s)


def parse_cell_id(name: str) -> dict[str, object]:
    """cell ID 例: HE22W_1_LA.1 から week / donor / chamber を取り出す。"""
    week_m = re.match(r"^HE(\d+)W", name)
    donor_m = re.match(r"^HE\d+W_(\d+)_", name)
    chamber_m = re.search(r"_(LA|LV|RA|RV|AO|AV|BV|PV|TV|IS|PO|ED|EP)(\d*)", name)
    return {
        "week": int(week_m.group(1)) if week_m else np.nan,
        "donor": int(donor_m.group(1)) if donor_m else np.nan,
        "chamber": chamber_m.group(1) if chamber_m else "UNKNOWN",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {UMI_FILE} ...")
    df = pd.read_csv(UMI_FILE, sep="\t", index_col=0, compression="gzip")
    print(f"raw shape (genes x cells): {df.shape}")

    parsed = [parse_cell_id(c) for c in df.columns.astype(str)]
    cell_meta = pd.DataFrame(parsed, index=df.columns.astype(str))
    cell_meta["region_cui"] = (cell_meta["chamber"]
                               .map(CHAMBER_TO_REGION).fillna("unknown"))
    cell_meta.index.name = "barcode"

    print(f"loading {MMC3} ...")
    clusters = pd.read_excel(MMC3, sheet_name="Clusters", header=0)
    clusters.columns = [str(c).strip() for c in clusters.columns]
    clusters = clusters.rename(columns={"Cell Name": "barcode",
                                        "Cluster": "author_cluster"})
    clusters["barcode"] = clusters["barcode"].astype(str)
    clusters["author_cell_type"] = [parse_cluster(s)[1]
                                    for s in clusters["author_cluster"]]

    cell_meta = cell_meta.join(
        clusters.set_index("barcode")[["author_cell_type"]], how="left"
    )
    cell_meta["author_cell_type"] = cell_meta["author_cell_type"].fillna("unlabeled")
    cell_meta["unified"] = cell_meta["author_cell_type"].map(LABEL_MAP)
    unk = cell_meta.loc[cell_meta["unified"].isna(),
                        "author_cell_type"].unique().tolist()
    if unk:
        raise RuntimeError(f"unmapped author labels: {unk}")

    # CM は region で aCM / vCM に分割。それ以外の region は CM_NA とする。
    av = pd.Series("", index=cell_meta.index)
    cm_mask = cell_meta["unified"] == "CM"
    av[cm_mask & (cell_meta["region_cui"] == "atrium")] = "aCM"
    av[cm_mask & (cell_meta["region_cui"] == "ventricle")] = "vCM"
    cell_meta["av_subtype"] = av
    cell_meta["cell_label"] = np.where(
        cell_meta["av_subtype"] != "",
        cell_meta["av_subtype"],
        np.where(cm_mask, "CM_NA", cell_meta["unified"])
    )

    cell_meta.reset_index().to_csv(OUT_CELLS, sep="\t", index=False)
    print(f"wrote cell-level annotation: {OUT_CELLS} ({len(cell_meta):,} cells)")
    print("author_cell_type counts:")
    print(cell_meta["author_cell_type"].value_counts().to_string())

    keep_mask = ~cell_meta["unified"].isin(EXCLUDED)
    cells_keep = cell_meta[keep_mask].copy()
    print(f"keep {len(cells_keep):,} / {len(cell_meta):,} cells")

    cells_keep["group_key"] = (cells_keep["cell_label"].astype(str) + "|"
                               + cells_keep["region_cui"].astype(str) + "|"
                               + cells_keep["week"].astype(str))

    group_info = (
        cells_keep.groupby("group_key", observed=True)
                  .agg(cell_label=("cell_label", "first"),
                       region=("region_cui", "first"),
                       week=("week", "first"),
                       n_cells=("cell_label", "size"))
                  .reset_index()
    )
    valid = group_info[group_info["n_cells"] >= MIN_CELLS].copy()
    print(f"pseudobulk groups: {len(group_info)} -> {len(valid)} "
          f"(min {MIN_CELLS} cells)")

    df_t = df.T.loc[cells_keep.index]
    df_t["group_key"] = cells_keep["group_key"].values
    pb_full = df_t.groupby("group_key", observed=True).sum().T
    pb_df = pb_full[valid["group_key"].tolist()].copy()
    pb_df.index.name = "gene_symbol"
    pb_df = pb_df.groupby(level=0).sum()

    sample_ids = [f"C_{r['cell_label']}_{r['region']}_W{int(r['week'])}"
                  for _, r in valid.iterrows()]
    pb_df.columns = sample_ids
    pb_df.to_csv(OUT_PB_COUNTS, sep="\t")

    valid["sample_id"] = sample_ids
    valid["dataset"] = "C"
    valid[["sample_id", "dataset", "cell_label", "region", "week",
           "n_cells"]].to_csv(OUT_PB_META, sep="\t", index=False)

    print(f"wrote pseudobulk counts: {OUT_PB_COUNTS} "
          f"({pb_df.shape[0]:,} x {pb_df.shape[1]})")
    print("groups per cell_label:")
    print(valid.groupby("cell_label", observed=True).size().to_string())


if __name__ == "__main__":
    main()
