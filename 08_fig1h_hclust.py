"""Fig 1H: bulk Non-myocyte と reference の非心筋細胞中心を 32 markers で並べ、
   correlation distance + average linkage で列方向に階層クラスタリングする。
   dendrogram の見やすさのために Epicardial (fetal) と FB (fetal) の内部 node を
   入れ替える (構造は保持、描画順だけ反転)。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, leaves_list

np.random.seed(42)

INT_DIR = Path("reports/revision_20260617/scripts/intermediate")
FIG_DIR = Path("reports/revision_20260617/figures/v2_rerun")
IN_LOG2CPM = INT_DIR / "integrated_log2cpm.tsv"
IN_META = INT_DIR / "sample_metadata.tsv"
OUT_FIG = FIG_DIR / "fig1h.png"
OUT_VAL = INT_DIR / "hclust_values.tsv"

FIGSIZE = (12.5, 8.0)
TICK_X_SIZE = 13
TICK_Y_SIZE = 12
CBAR_LABEL_SIZE = 13
CBAR_TICK_SIZE = 12
CBAR_LEFT = 0.11
CBAR_BOTTOM = 0.78
CBAR_WIDTH = 0.012
CBAR_HEIGHT = 0.18
DENDRO_RATIO = (0.11, 0.17)
SUBPLOTS_LEFT = 0.10
SUBPLOTS_RIGHT = 0.90
SUBPLOTS_BOTTOM = 0.22

MARKERS = {
    "Fibroblast":  ["THY1", "PDGFRA", "COL1A1", "COL3A1", "VIM",
                    "POSTN", "DCN", "LUM"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "KDR", "TIE1"],
    "Mural":       ["ACTA2", "MYH11", "RGS5", "PDGFRB"],
    "Epicardial":  ["WT1", "TBX18", "TCF21"],
    "Immune":      ["PTPRC", "CD68", "CD163", "TYROBP", "HLA-DRA", "C1QA"],
    "Neural":      ["NEFL", "NEFM", "S100B", "PLP1", "MPZ", "SOX10"],
}
REF_LABELS = ["FB", "EC", "Mural", "Epicardial", "Immune", "Neural"]
BULK_SAMPLES = ["Non-myocyte_1", "Non-myocyte_2"]
CAT_COLORS = {
    "Fibroblast": "#8c564b", "Endothelial": "#2ca02c", "Mural": "#ff9896",
    "Epicardial": "#aec7e8", "Immune": "#7f7f7f", "Neural": "#9edae5",
}
SWAP_PAIR = ("Epicardial (fetal)", "FB (fetal)")


def _subtree_leaves(node_id: int, Z: np.ndarray, n: int) -> set[int]:
    if node_id < n:
        return {node_id}
    row = Z[node_id - n]
    return (_subtree_leaves(int(row[0]), Z, n)
            | _subtree_leaves(int(row[1]), Z, n))


def swap_pair_in_linkage(Z: np.ndarray, col_names: list[str],
                         pair_a: str, pair_b: str
                         ) -> tuple[np.ndarray, int | None]:
    """Z の中で pair_a と pair_b が初めて同じ subtree に入る node の
       左右を入れ替える。dendrogram の構造 (距離・cluster) は保たれる。"""
    n = len(col_names)
    Z_new = Z.copy()
    if pair_a not in col_names or pair_b not in col_names:
        return Z_new, None
    a_idx, b_idx = col_names.index(pair_a), col_names.index(pair_b)
    for i, row in enumerate(Z_new):
        L = _subtree_leaves(int(row[0]), Z_new, n)
        R = _subtree_leaves(int(row[1]), Z_new, n)
        if (a_idx in L and b_idx in R) or (a_idx in R and b_idx in L):
            Z_new[i, 0], Z_new[i, 1] = Z_new[i, 1], Z_new[i, 0]
            return Z_new, i
    return Z_new, None


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log2cpm = pd.read_csv(IN_LOG2CPM, sep="\t", index_col=0)
    meta = pd.read_csv(IN_META, sep="\t")

    bulk_log = log2cpm[BULK_SAMPLES].copy()

    ref_meta = meta[meta["dataset"].isin(["A", "C"])
                    & meta["cell_label"].isin(REF_LABELS)].copy()
    ref_log = log2cpm[ref_meta["sample_id"].tolist()]
    ref_T = ref_log.T.copy()
    ref_T["dataset"] = (ref_meta.set_index("sample_id")
                        .loc[ref_T.index, "dataset"].values)
    ref_T["cell_label"] = (ref_meta.set_index("sample_id")
                           .loc[ref_T.index, "cell_label"].values)
    ref_mean = (ref_T.groupby(["dataset", "cell_label"])
                .mean(numeric_only=True).T)
    ds_label = {"A": "adult", "C": "fetal"}
    ref_mean.columns = [f"{cl} ({ds_label[ds]})"
                        for ds, cl in ref_mean.columns]

    seen: set[str] = set()
    marker_order: list[str] = []
    cat_of: dict[str, str] = {}
    for cat, genes in MARKERS.items():
        for g in genes:
            if g not in seen:
                marker_order.append(g)
                cat_of[g] = cat
                seen.add(g)
    available = [g for g in marker_order
                 if g in bulk_log.index and g in ref_mean.index]
    missing = [g for g in marker_order if g not in available]
    if missing:
        print(f"missing markers: {missing}")
    print(f"markers used: {len(available)} / {len(marker_order)}")

    full = pd.concat([bulk_log.loc[available], ref_mean.loc[available]],
                     axis=1)
    # 行ラベルは "遺伝子名 + [カテゴリ]" の混在のため、遺伝子名部分のみを
    # mathtext で italic 化する ($\mathit{...}$)。カテゴリ部分は通常体。
    full.index = pd.Index([rf"$\mathit{{{g}}}$  [{cat_of[g]}]"
                           for g in available])
    full.to_csv(OUT_VAL, sep="\t")

    row_colors = [CAT_COLORS[cat_of[g]] for g in available]

    # まず一度クラスタリングして linkage を取り出し、
    # 表示順を入れ替えてから本番のクラスタマップを描く。
    g_tmp = sns.clustermap(
        full, z_score=0, cmap="RdBu_r", vmin=-2, vmax=2,
        row_cluster=False, col_cluster=True,
        metric="correlation", method="average",
    )
    Z_orig = g_tmp.dendrogram_col.linkage
    plt.close(g_tmp.fig)
    Z_swapped, swap_row = swap_pair_in_linkage(
        Z_orig, list(full.columns), SWAP_PAIR[0], SWAP_PAIR[1])
    if swap_row is None:
        Z_swapped = Z_orig

    # cbar_pos=None で seaborn デフォルトの colorbar を出さず、
    # あとから add_axes で確実な位置に colorbar を載せる。
    g = sns.clustermap(
        full, z_score=0, cmap="RdBu_r", vmin=-2, vmax=2,
        row_cluster=False, col_cluster=True,
        col_linkage=Z_swapped,
        metric="correlation", method="average",
        row_colors=row_colors,
        figsize=FIGSIZE,
        dendrogram_ratio=DENDRO_RATIO,
        cbar_pos=None,
        xticklabels=True, yticklabels=True,
    )
    plt.setp(g.ax_heatmap.get_xticklabels(),
             rotation=45, ha="right", fontsize=TICK_X_SIZE)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=TICK_Y_SIZE)
    g.fig.subplots_adjust(left=SUBPLOTS_LEFT, right=SUBPLOTS_RIGHT,
                          bottom=SUBPLOTS_BOTTOM)

    cax = g.fig.add_axes([CBAR_LEFT, CBAR_BOTTOM, CBAR_WIDTH, CBAR_HEIGHT])
    norm = mcolors.Normalize(vmin=-2, vmax=2)
    cbar = mcolorbar.ColorbarBase(cax, cmap="RdBu_r", norm=norm,
                                  orientation="vertical")
    cbar.set_label("row z-score", fontsize=CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    cbar.set_ticks([-2, -1, 0, 1, 2])

    g.savefig(OUT_FIG, dpi=180)
    plt.close(g.fig)
    print(f"wrote {OUT_FIG}")

    cluster_labels = fcluster(Z_swapped, t=3, criterion="maxclust")
    col_idx = {c: i for i, c in enumerate(full.columns)}
    leaves_order = [full.columns[i] for i in leaves_list(Z_swapped)]
    print("dendrogram order:")
    print("  " + " -> ".join(leaves_order))
    print("Non-myocyte neighbors in the same k=3 cluster:")
    for s in BULK_SAMPLES:
        cid = cluster_labels[col_idx[s]]
        peers = [c for c in full.columns
                 if c not in BULK_SAMPLES and cluster_labels[col_idx[c]] == cid]
        print(f"  {s}: {peers}")


if __name__ == "__main__":
    main()
