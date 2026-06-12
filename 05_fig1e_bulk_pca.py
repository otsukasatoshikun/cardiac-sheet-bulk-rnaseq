"""Fig 1E: bulk 8 サンプルの PCA。
   top 2000 HVG -> gene-wise z-score -> SVD で第 2 主成分まで取り出す。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

np.random.seed(42)

INT_DIR = Path("reports/revision_20260617/scripts/intermediate")
FIG_DIR = Path("reports/revision_20260617/figures/v2_rerun")
PALETTE_PATH = Path("config/figure_palette.yaml")

IN_LOG2CPM = INT_DIR / "integrated_log2cpm.tsv"
IN_META = INT_DIR / "sample_metadata.tsv"
OUT_FIG = FIG_DIR / "fig1e.png"
OUT_VAL = INT_DIR / "bulk_pca_values.tsv"

SAMPLE_ORDER = [
    "CM_1", "CM_2", "Non-myocyte_1", "Non-myocyte_2",
    "CTS_1", "CTS_2", "MCTS_1", "MCTS_2",
]
TOP_HVG = 2000


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log2cpm = pd.read_csv(IN_LOG2CPM, sep="\t", index_col=0)
    meta = pd.read_csv(IN_META, sep="\t")
    bulk_log = log2cpm[SAMPLE_ORDER].copy()

    with open(PALETTE_PATH, encoding="utf-8") as fh:
        palette = yaml.safe_load(fh)
    cond_color = {k: palette["cell_color"][k]
                  for k in ["CM", "Non-myocyte", "CTS", "MCTS"]}
    cond_of = dict(zip(meta["sample_id"], meta["cell_label"]))

    variance = bulk_log.var(axis=1, ddof=0).sort_values(ascending=False)
    top_genes = variance.head(TOP_HVG).index
    X = bulk_log.loc[top_genes].to_numpy()

    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True, ddof=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    Xc = Xz - Xz.mean(axis=1, keepdims=True)

    U, S, _Vt = np.linalg.svd(Xc.T, full_matrices=False)
    scores = U * S
    ev = (S ** 2) / np.sum(S ** 2) * 100.0
    print(f"PC1 = {ev[0]:.2f}%, PC2 = {ev[1]:.2f}%, PC3 = {ev[2]:.2f}%")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, s in enumerate(SAMPLE_ORDER):
        ax.scatter(scores[i, 0], scores[i, 1], s=140,
                   color=cond_color[cond_of[s]], edgecolor="black", zorder=3)
        ax.annotate(s, (scores[i, 0], scores[i, 1]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)", fontsize=12)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=k,
                          markerfacecolor=v, markeredgecolor="black",
                          markersize=10)
               for k, v in cond_color.items()]
    ax.legend(handles=handles, title="Cell types", loc="best")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    val_df = pd.DataFrame(scores[:, :3], index=SAMPLE_ORDER,
                          columns=["PC1", "PC2", "PC3"])
    val_df["condition"] = [cond_of[s] for s in SAMPLE_ORDER]
    val_df.to_csv(OUT_VAL, sep="\t", index_label="sample_id")
    with open(OUT_VAL, "a", encoding="utf-8") as fh:
        fh.write(f"\n# explained_variance_pct: "
                 f"PC1={ev[0]:.4f} PC2={ev[1]:.4f} PC3={ev[2]:.4f}\n")
    print(f"wrote {OUT_FIG}")
    print(f"wrote {OUT_VAL}")


if __name__ == "__main__":
    main()
