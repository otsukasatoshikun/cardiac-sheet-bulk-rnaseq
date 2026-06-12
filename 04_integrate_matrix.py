"""bulk count と各 reference dataset の pseudobulk count を
   gene symbol で intersect して 1 つの行列にまとめ、log2(CPM+1) に正規化する。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

INT_DIR = Path("reports/revision_20260617/scripts/intermediate")
BULK_CNT = INT_DIR / "bulk_counts.tsv"
BULK_META = INT_DIR / "bulk_meta.tsv"
A_CNT = INT_DIR / "litvinukova_pseudobulk_counts.tsv"
A_META = INT_DIR / "litvinukova_pseudobulk_meta.tsv"
C_CNT = INT_DIR / "cui_pseudobulk_counts.tsv"
C_META = INT_DIR / "cui_pseudobulk_meta.tsv"

OUT_LOG2CPM = INT_DIR / "integrated_log2cpm.tsv"
OUT_RAW = INT_DIR / "integrated_counts_raw.tsv"
OUT_META = INT_DIR / "sample_metadata.tsv"


def main() -> None:
    bulk = pd.read_csv(BULK_CNT, sep="\t", index_col=0)
    a = pd.read_csv(A_CNT, sep="\t", index_col=0)
    c = pd.read_csv(C_CNT, sep="\t", index_col=0)
    print(f"bulk: {bulk.shape}, A: {a.shape}, C: {c.shape}")

    common = bulk.index.intersection(a.index).intersection(c.index)
    print(f"common gene symbols: {len(common):,}")

    integrated = pd.concat([bulk.loc[common], a.loc[common], c.loc[common]], axis=1)
    integrated.astype(np.int64, errors="ignore").to_csv(OUT_RAW, sep="\t")

    libsize = integrated.sum(axis=0)
    cpm = integrated.divide(libsize, axis=1) * 1e6
    log2cpm = np.log2(cpm + 1.0)
    log2cpm.to_csv(OUT_LOG2CPM, sep="\t")
    print(f"integrated log2(CPM+1): {log2cpm.shape}")

    bulk_meta = pd.read_csv(BULK_META, sep="\t")
    bulk_long = pd.DataFrame({
        "sample_id": bulk_meta["sample_id"],
        "dataset": "bulk",
        "cell_label": bulk_meta["condition"],
        "region": "",
        "donor": "",
        "week": np.nan,
        "n_cells": np.nan,
    })

    a_meta = pd.read_csv(A_META, sep="\t")
    a_long = pd.DataFrame({
        "sample_id": a_meta["sample_id"],
        "dataset": a_meta["dataset"],
        "cell_label": a_meta["cell_label"],
        "region": a_meta["region"],
        "donor": a_meta["donor"],
        "week": np.nan,
        "n_cells": a_meta["n_cells"],
    })

    c_meta = pd.read_csv(C_META, sep="\t")
    c_long = pd.DataFrame({
        "sample_id": c_meta["sample_id"],
        "dataset": c_meta["dataset"],
        "cell_label": c_meta["cell_label"],
        "region": c_meta["region"],
        "donor": "",
        "week": c_meta["week"],
        "n_cells": c_meta["n_cells"],
    })

    sample_meta = pd.concat([bulk_long, a_long, c_long], axis=0,
                            ignore_index=True)
    sample_meta.to_csv(OUT_META, sep="\t", index=False)

    print(f"sample metadata: {len(sample_meta)} samples")
    print(sample_meta["dataset"].value_counts().to_string())


if __name__ == "__main__":
    main()
