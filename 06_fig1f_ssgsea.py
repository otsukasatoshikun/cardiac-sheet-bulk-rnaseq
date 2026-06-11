"""Fig 1F: ssGSEA で atrial-ventricular スコアと maturation スコアを計算し、
   bulk サンプルを reference の aCM / vCM 中心と並べて散布する。"""
from __future__ import annotations

from pathlib import Path

import gseapy as gp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

np.random.seed(42)
SEED = 42

INT_DIR = Path("reports/revision_20260617/scripts/intermediate")
FIG_DIR = Path("reports/revision_20260617/figures/v2_rerun")
PALETTE_PATH = Path("config/figure_palette.yaml")
GMT = Path("config/signatures.gmt")

IN_LOG2CPM = INT_DIR / "integrated_log2cpm.tsv"
IN_META = INT_DIR / "sample_metadata.tsv"
OUT_FIG = FIG_DIR / "fig1f.png"
OUT_VAL = INT_DIR / "ssgsea_values.tsv"

USE_SETS = ["ATRIAL_CLEAN", "VENTRICULAR", "IMMATURE", "MATURE"]
BULK_CONDS = ["CM", "CTS", "MCTS"]
REF_LABELS = ["aCM", "vCM"]


def read_gmt(path: Path) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sets[parts[0]] = [g for g in parts[2:] if g]
    return sets


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log2cpm = pd.read_csv(IN_LOG2CPM, sep="\t", index_col=0)
    meta = pd.read_csv(IN_META, sep="\t")
    gene_sets = {k: v for k, v in read_gmt(GMT).items() if k in USE_SETS}

    ss = gp.ssgsea(
        data=log2cpm, gene_sets=gene_sets,
        sample_norm_method="rank", min_size=2, max_size=1000,
        permutation_num=0, outdir=None, seed=SEED, threads=4, verbose=False,
    )
    res = ss.res2d.copy()
    nes_col = "NES" if "NES" in res.columns else "ES"
    res[nes_col] = pd.to_numeric(res[nes_col], errors="coerce")
    nes_wide = (res.pivot(index="Name", columns="Term", values=nes_col)
                .reindex(columns=USE_SETS))
    nes_wide["av_score"] = nes_wide["VENTRICULAR"] - nes_wide["ATRIAL_CLEAN"]
    nes_wide["mat_score"] = nes_wide["MATURE"] - nes_wide["IMMATURE"]

    merged = meta.set_index("sample_id").join(nes_wide, how="left").reset_index()
    merged.to_csv(OUT_VAL, sep="\t", index=False)

    with open(PALETTE_PATH, encoding="utf-8") as fh:
        pal = yaml.safe_load(fh)
    cell_color = pal["cell_color"]
    ds_marker = pal["dataset_marker"]

    ref = merged[merged["dataset"].isin(["A", "C"])
                 & merged["cell_label"].isin(REF_LABELS)]
    cent = (ref.groupby(["dataset", "cell_label"])
            [["av_score", "mat_score"]].mean().reset_index())
    bulk = merged[(merged["dataset"] == "bulk")
                  & merged["cell_label"].isin(BULK_CONDS)]

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ds_label = {"A": "adult", "C": "fetal"}
    for _, r in cent.iterrows():
        c = cell_color.get(r["cell_label"], "#888888")
        m = ds_marker.get(r["dataset"], "o")
        ax.scatter(r["av_score"], r["mat_score"], s=180,
                   marker=m, c=c, edgecolors="black",
                   linewidths=0.7, alpha=0.9, zorder=3)
        # adult vCM は右端の凡例とぶつかりやすいのでラベルを左に逃がす
        if r["dataset"] == "A" and r["cell_label"] == "vCM":
            offset, ha = (-8, 5), "right"
        else:
            offset, ha = (7, 5), "left"
        ax.annotate(f"{r['cell_label']} ({ds_label[r['dataset']]})",
                    (r["av_score"], r["mat_score"]),
                    fontsize=9, xytext=offset, textcoords="offset points",
                    color=c, fontweight="bold", ha=ha, zorder=4)

    for _, r in bulk.iterrows():
        c = cell_color.get(r["cell_label"], "#333333")
        ax.scatter(r["av_score"], r["mat_score"], s=360,
                   marker=ds_marker["bulk"], c=c, edgecolors="black",
                   linewidths=0.8, zorder=5)
        ax.annotate(r["sample_id"], (r["av_score"], r["mat_score"]),
                    fontsize=9, fontweight="bold",
                    xytext=(6, -10), textcoords="offset points",
                    color=c, zorder=6)

    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.axvline(0, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("Atrial  <-  av_score  ->  Ventricular\n"
                  "[NES(VENTRICULAR) - NES(ATRIAL_CLEAN)]")
    ax.set_ylabel("Immature  <-  mat_score  ->  Mature\n"
                  "[NES(MATURE) - NES(IMMATURE)]")
    ax.grid(True, alpha=0.2)

    cell_order = ["aCM", "vCM", "CM", "CTS", "MCTS"]
    h_color = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=cell_color[c], markeredgecolor="k",
                      markersize=10, label=c) for c in cell_order]
    leg1 = ax.legend(handles=h_color, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), fontsize=9,
                     title="Cell types", title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg1)
    h_marker = [
        Line2D([0], [0], marker=ds_marker["A"], color="w",
               markerfacecolor="#888", markeredgecolor="k",
               markersize=10, label="Adult (Litvinukova 2020)"),
        Line2D([0], [0], marker=ds_marker["C"], color="w",
               markerfacecolor="#888", markeredgecolor="k",
               markersize=10, label="Fetal (Cui 2019)"),
        Line2D([0], [0], marker=ds_marker["bulk"], color="w",
               markerfacecolor="#888", markeredgecolor="k",
               markersize=14, label="this study"),
    ]
    ax.legend(handles=h_marker, loc="lower left",
              bbox_to_anchor=(1.02, 0.0), fontsize=9,
              title="Dataset", title_fontsize=10, framealpha=0.95)

    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG}")
    print(f"wrote {OUT_VAL}")
    print("bulk sample scores:")
    print(bulk[["sample_id", "av_score", "mat_score"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
