"""bulk RNA-seq 8 サンプルの生 count 行列を読み込み、QC した上で
   中間ファイル化する。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

BULK_COUNTS = Path("data/bulk/real_bulk_counts.tsv")
BULK_META = Path("data/bulk/real_bulk_meta.tsv")
OUT_DIR = Path("reports/revision_20260617/scripts/intermediate")
OUT_COUNTS = OUT_DIR / "bulk_counts.tsv"
OUT_META = OUT_DIR / "bulk_meta.tsv"

SAMPLE_ORDER = [
    "CM_1", "CM_2", "Non-myocyte_1", "Non-myocyte_2",
    "CTS_1", "CTS_2", "MCTS_1", "MCTS_2",
]

# wet 側で記録された旧命名 (sample_id MC_1/MC_2, condition "MC") は
# 解析全体で使う新命名 (Non-myocyte_1/Non-myocyte_2, "Non-myocyte") に
# 本スクリプト内で読み替える。生データ (real_bulk_*.tsv) 自体は変更しない。
# 04_integrate_matrix.py で condition 列が cell_label にコピーされるため、
# 本 step で condition を書き換えれば下流の cell_label も新命名で揃う。
RENAME_MAP_SAMPLE = {"MC_1": "Non-myocyte_1", "MC_2": "Non-myocyte_2"}
RENAME_MAP_CONDITION = {"MC": "Non-myocyte"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cnt = pd.read_csv(BULK_COUNTS, sep="\t", index_col=0)
    meta = pd.read_csv(BULK_META, sep="\t")

    cnt = cnt.rename(columns=RENAME_MAP_SAMPLE)
    meta["sample_id"] = meta["sample_id"].replace(RENAME_MAP_SAMPLE)
    meta["condition"] = meta["condition"].replace(RENAME_MAP_CONDITION)

    missing = [s for s in SAMPLE_ORDER if s not in cnt.columns]
    if missing:
        raise RuntimeError(f"sample columns not found: {missing}")
    cnt = cnt[SAMPLE_ORDER].copy()

    # expected counts は整数に丸めて非負であることを担保する
    if not np.all(cnt.values == cnt.values.astype(int)):
        raise RuntimeError("non-integer counts detected")
    if (cnt.values < 0).any():
        raise RuntimeError("negative counts detected")

    cnt = cnt.astype(int)
    cnt.to_csv(OUT_COUNTS, sep="\t")
    meta.to_csv(OUT_META, sep="\t", index=False)

    libsize = cnt.sum(axis=0)
    print(f"loaded {cnt.shape[0]:,} genes x {cnt.shape[1]} samples")
    print("library sizes (M):",
          ", ".join(f"{s}={int(libsize[s] / 1e6)}" for s in SAMPLE_ORDER))


if __name__ == "__main__":
    main()
