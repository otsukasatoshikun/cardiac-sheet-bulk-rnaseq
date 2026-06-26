# Bulk RNA-seq × Public scRNA-seq Comparative Analysis of Multi-layered Cardiac Tissue Sheets

This repository contains the analysis code accompanying the cardiomyocyte tissue sheet study by Kawatou et al. (in revision at iScience). The code reproduces panels **E, F, G, and H of Figure 1** from raw count data and public single-cell RNA-seq references.

## Overview

We compared bulk RNA-seq profiles of four hiPSC-derived sample types (cardiomyocytes, non-myocytes, single-layer cardiac tissue sheets [CTSs], and multi-layered cardiac tissue sheets [MCTSs]) with two public scRNA-seq atlases of the human heart, in order to characterize the transcriptional state of the constituent cell types of MCTSs.

| Panel | Analysis |
|---|---|
| Fig 1E | Principal component analysis of bulk samples |
| Fig 1F | Atrial-ventricular and maturation scoring by ssGSEA |
| Fig 1G | Expression of six cardiac ion-channel genes |
| Fig 1H | Hierarchical clustering of non-myocyte populations across a 32-gene marker panel |

Full methodological details are provided in the STAR Methods section of the manuscript.

## Repository contents

```
.
├── 01_load_bulk.py            Load and QC the bulk count matrix
├── 02_load_litvinukova.py     Process the adult Heart Cell Atlas (Litviňuková et al., 2020)
├── 03_load_cui.py             Process the fetal cardiac dataset (Cui et al., 2019)
├── 04_integrate_matrix.py     Merge bulk + pseudobulk into log2(CPM + 1) matrix
├── 05_fig1e_bulk_pca.py       Generate Fig 1E (bulk PCA)
├── 06_fig1f_ssgsea.py         Generate Fig 1F (atrial-ventricular vs maturation score scatter)
├── 07_fig1g_ion_channels.py   Generate Fig 1G (six ion channels)
├── 08_fig1h_hclust.py         Generate Fig 1H (32-gene marker hclust)
├── config/
│   ├── cell_type_dict.yaml    Cell-type label harmonization rules
│   ├── signatures.gmt         Gene-set definitions (ATRIAL, VENTRICULAR, IMMATURE, MATURE)
│   ├── figure_palette.yaml    Color palette for figures
│   └── cardiac_env.yml        Conda environment specification
└── README.md
```

## Input data

The scripts expect the following raw data files. None of these are redistributed in this repository.

| Dataset | Source | Required files |
|---|---|---|
| Bulk RNA-seq (this study) | NCBI GEO: **[GSE335163](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE335163)** *(private until publication; reviewer access available upon request)* | `data/bulk/real_bulk_counts.tsv`, `data/bulk/real_bulk_meta.tsv` |
| Adult Heart Cell Atlas (Litviňuková et al., 2020) | https://www.heartcellatlas.org/ | `data/reference/litvinukova/Global_raw.h5ad` |
| Fetal cardiac dataset (Cui et al., 2019) | NCBI GEO: GSE106118 | `GSE106118_UMI_count_merge.txt.gz`, `mmc3.xlsx` (donor/region metadata from Cui et al., 2019 supplementary table) |

## Environment setup

A conda environment specification is provided. The analysis was developed and tested under Python 3.11 on Windows 11.

```bash
conda env create -f config/cardiac_env.yml
conda activate cardiac
```

Key dependencies (versions used in the published analysis):

- Python 3.11
- NumPy 2.4.5
- pandas 2.3.3
- SciPy 1.17.1
- scanpy / anndata (for `Global_raw.h5ad` input)
- GSEApy 1.2.1
- matplotlib / seaborn

A random seed of 42 is used throughout (`np.random.seed(42)` / `set.seed(42)` where applicable).

## How to run

Execute scripts in numerical order (each writes its outputs to `intermediate/` and `figures/`):

```bash
python 01_load_bulk.py
python 02_load_litvinukova.py
python 03_load_cui.py
python 04_integrate_matrix.py
python 05_fig1e_bulk_pca.py
python 06_fig1f_ssgsea.py
python 07_fig1g_ion_channels.py
python 08_fig1h_hclust.py
```

Expected runtime on a desktop workstation: approximately 10-20 minutes total, dominated by loading the Heart Cell Atlas `Global_raw.h5ad` (~10 GB).

## Expected outputs

- Integrated count matrix: 19,794 genes × 675 samples
- Fig 1E: PC1 63.5 %, PC2 24.3 %
- Fig 1F: bulk CM-containing samples positioned in the ventricular-positive region
- Fig 1G: KCNJ2 and SCN5A markedly lower in bulk samples than in adult ventricular cardiomyocytes
- Fig 1H: bulk MC samples cluster with fibroblast centroids

## Citation

If you use this code, please cite:

> Kawatou et al. (in revision). [Title to be finalized]. *iScience*.

## License

This code is released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

## Contact

For questions, please contact the lead contact listed in the published manuscript.

---

*Code archived at Zenodo (DOI: [10.5281/zenodo.20652760](https://doi.org/10.5281/zenodo.20652760)) at the time of publication.*

