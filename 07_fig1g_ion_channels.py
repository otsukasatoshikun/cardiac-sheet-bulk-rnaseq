"""Fig 1G: bulk と reference vCM で 6 つの心筋イオンチャネル遺伝子の発現を比較する。
   各 group の平均 +/- SD を棒で示し、個別サンプル値を点で重ねる。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)
SEED = 42

INT_DIR = Path("reports/revision_20260617/scripts/intermediate")
FIG_DIR = Path("reports/revision_20260617/figures/v2_rerun")
IN_LOG2CPM = INT_DIR / "integrated_log2cpm.tsv"
IN_META = INT_DIR / "sample_metadata.tsv"
OUT_FIG = FIG_DIR / "fig1g.png"
OUT_VAL = INT_DIR / "ion_channels_values.tsv"

ION_GENES = ["SCN5A", "CACNA1C", "KCNH2", "KCNQ1", "KCNJ2", "KCNJ12"]

SERIES = [
    ("CMs",      "CMs (n=2)",            "#e377c2"),
    ("CTSs",     "CTSs (n=2)",           "#17becf"),
    ("MCTSs",    "MCTSs (n=2)",          "#ff7f0e"),
    ("FetalvCM", "Fetal vCM (Cui 2019, n=10)",         "#2ca02c"),
    ("AdultvCM", "Adult vCM (Litviňuková 2020, n=70)", "#1f77b4"),
]

BULK_MAP = {
    "CMs":   ["CM_1", "CM_2"],
    "CTSs":  ["CTS_1", "CTS_2"],
    "MCTSs": ["MCTS_1", "MCTS_2"],
}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log2cpm = pd.read_csv(IN_LOG2CPM, sep="\t", index_col=0)
    meta = pd.read_csv(IN_META, sep="\t")
    missing = [g for g in ION_GENES if g not in log2cpm.index]
    if missing:
        raise RuntimeError(f"missing genes: {missing}")
    log_ion = log2cpm.loc[ION_GENES]

    series_samples = dict(BULK_MAP)
    series_samples["FetalvCM"] = meta.loc[
        (meta["dataset"] == "C") & (meta["cell_label"] == "vCM"),
        "sample_id"].tolist()
    series_samples["AdultvCM"] = meta.loc[
        (meta["dataset"] == "A") & (meta["cell_label"] == "vCM"),
        "sample_id"].tolist()
    for k, ss in series_samples.items():
        if not ss:
            raise RuntimeError(f"no samples for series {k}")

    means: dict[str, dict[str, float]] = {}
    sds: dict[str, dict[str, float]] = {}
    indiv: dict[str, dict[str, list[float]]] = {}
    for key, _, _ in SERIES:
        sub = log_ion[series_samples[key]]
        means[key] = sub.mean(axis=1).to_dict()
        sds[key] = (sub.std(axis=1, ddof=1).to_dict()
                    if sub.shape[1] >= 2 else {g: 0.0 for g in ION_GENES})
        indiv[key] = {g: sub.loc[g].tolist() for g in ION_GENES}

    rows = []
    for g in ION_GENES:
        for key, label, _ in SERIES:
            rows.append({"gene": g, "series": key, "label": label,
                         "mean_log2cpm": means[key][g],
                         "sd_log2cpm": sds[key][g],
                         "n": len(series_samples[key])})
    pd.DataFrame(rows).to_csv(OUT_VAL, sep="\t", index=False)

    n_genes = len(ION_GENES)
    n_series = len(SERIES)
    bar_width = 0.14
    x_base = np.arange(n_genes)

    fig, ax = plt.subplots(figsize=(15, 6.5))
    rng = np.random.default_rng(SEED)
    for i, (key, label, color) in enumerate(SERIES):
        offset = (i - n_series / 2 + 0.5) * bar_width
        vals = [means[key][g] for g in ION_GENES]
        errs = [sds[key][g] for g in ION_GENES]
        ax.bar(x_base + offset, vals, width=bar_width,
               color=color, edgecolor="black", linewidth=0.6,
               label=label, zorder=3,
               yerr=errs, ecolor="black",
               error_kw=dict(elinewidth=1.0, capsize=2, capthick=1.0))
        n_pts = len(series_samples[key])
        for j, g in enumerate(ION_GENES):
            pts = indiv[key][g]
            if n_pts <= 2:
                xs = np.full(n_pts, x_base[j] + offset)
                size, alpha = 22, 1.0
            else:
                xs = (x_base[j] + offset
                      + rng.uniform(-bar_width * 0.32,
                                    bar_width * 0.32, n_pts))
                size = 10 if n_pts <= 20 else 6
                alpha = 0.75 if n_pts <= 20 else 0.5
            ax.scatter(xs, pts, s=size, c="black",
                       edgecolors="white", linewidths=0.3,
                       alpha=alpha, zorder=5)

    ax.set_xticks(x_base)
    # 遺伝子名は論文表記の慣習に従って斜体 (italic) で表示する
    ax.set_xticklabels(ION_GENES, fontsize=12, fontweight="bold",
                       fontstyle="italic")
    ax.set_ylabel("log2(CPM + 1)", fontsize=12)
    ax.set_xlabel("Ion channel gene", fontsize=12)
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(axis="y", alpha=0.25, zorder=1)
    # y 軸上限を拡張 (凡例を ax 内右上 = KCNJ2/KCNJ12 領域の上に配置するため余白を確保)
    ymax = max(means[key][g] + sds[key][g]
               for key, _, _ in SERIES for g in ION_GENES)
    ax.set_ylim(top=ymax * 1.32)
    # 凡例を ax 内右上 (KCNJ2/KCNJ12 領域の上) に inset 配置
    ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.985),
              fontsize=9, framealpha=0.95,
              title="Sample group", title_fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG}")
    print(f"wrote {OUT_VAL}")

    summary = pd.DataFrame(
        {key: [means[key][g] for g in ION_GENES] for key, _, _ in SERIES},
        index=ION_GENES).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()
