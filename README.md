# q2-haarlikedist

QIIME 2 plugin implementing the **Haar-like distance (HLD)** and **Adaptive Haar-like Distance (AHLD) v2** for phylogenetically aware beta-diversity analysis of microbiome data.

This plugin provides a scalable implementation of Haar-like wavelet projections on phylogenetic trees, called AHLD2, enabling the computation of interpretable distances between microbial community samples while highlighting the clades that contribute most strongly to observed differences.

---

# Key Features

- **Phylogenetically aware distance metric**
- **Haar-like wavelet representation** of phylogenetic trees
- **Interpretability** through identification of influential internal nodes
- **Supervised extension (AHLD)** using Random Forest models
- **Scalable design** for large microbiome datasets
- **Integration with QIIME 2** workflows

---

# Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/YOUR_USERNAME/q2-haarlikedist.git
cd q2-haarlikedist
pip install -e .
```

The plugin should be installed inside an active **QIIME 2 environment**.

Example:

```bash
conda activate qiime2
pip install -e .
```

After installation, the plugin will be available through the QIIME 2 CLI.

---

# Example Usage

Typical inputs include:

- a QIIME 2 feature table
- a phylogenetic tree whose tips correspond to the observed features
- a metadata file containing the variable of interest

Example command:

```bash
qiime haarlikedist adaptive-visual \
  --i-biom-table table.qza \
  --i-tree tree.qza \
  --p-label variable-of-interest \
  --m-metadata-file metadata.tsv \
  --o-visualization output.qzv
```

Refer to the plugin help text for detailed descriptions of parameters and available outputs.

---

# Notes

- The feature table and tree must share the same set of features (tips).
- The quality of the phylogenetic tree can affect interpretability.
- AHLD highlights candidate discriminative clades but does not determine causal biological mechanisms.
- The method should be considered a **marker-screening and hypothesis-generation tool**.

---

# Citation

If you use this software in your research, please cite the relevant Haar-like distance and QIIME 2 publications.

Suggested references:

- Ghasemloo Gheidari et al., “Crowd augmentation in citizen science enables high-resolution gut biomarker discovery”, preprint, to-be-submitted
- Ghasemloo Gheidari et al., “Large-scale beta diversity analysis of the gut microbiome uncovers markers”, preprint, to be submitted
- Gorman, E. & Lladser, M. E. *Sparsification of large ultrametric matrices: insights into the microbial Tree of Life.*
- Gorman, E. D. & Lladser, M. E. *Interpretable metric learning in comparative metagenomics: The adaptive Haar-like distance.*
- Bolyen, E. et al. *Reproducible, interactive, scalable and extensible microbiome data science using QIIME 2.* Nature Biotechnology (2019).

---

# About QIIME 2

QIIME 2 is a platform for **reproducible and extensible microbiome data science**.

More information:
https://qiime2.org/

---

# License

See the repository license file for details.
