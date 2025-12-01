# Algorithmic Extractors
This repository contains the source code for the feature extraction methodologies presented in the research article: **"Évaluation d’une Stratégie d’Hybridation Primitives-Transformeur pour Quantifier la Sévérité de Pneumonies"**.

The project focuses on generating Discrete Wavelet Transform (DWT), Gabor, and Histogram of Oriented Gradients (HOG) feature maps from Chest X-ray images (specifically the RALO dataset) to evaluate their impact on transformer-based severity quantification models.

## 📄 Abstract

Recent research on automatic severity scoring of pneumonia in chest X-ray images focuses primarily on improving AI model architectures. This work argues for a shift towards a data-centered approach, focusing on algorithmic feature-extraction to provide richer training sets to these models. We conduct a comparative analysis to determine the effects that this pre-processing stage, hybridizing primitives with transformers, has on the performance of automatic pneumonia scoring tasks.

## 📂 Repository Structure

The codebase is organized into three main extraction scripts.

* **`create_dwt.py`**: Generates DWT visualizations using various wavelets (Haar, Daubechies, Symlets, Coiflets, Biorthogonal). It computes approximation and detail coefficients and tiles them into a single visual map.

* **`create_gabor.py`**: Applies Gabor filters with varying orientations ($\theta$) and frequencies ($f$). **Note:** This script is currently optimized for GPU execution using `CuPy`.

* **`create_hog.py`**: Computes Histogram of Oriented Gradients (HOG) features and generates their visual representation.

**Note on LBP Variants:** The Local Binary Pattern (LBP) variants discussed in the article and presented in the results were generated using the implementation available at: https://github.com/carolinepacheco/lbp-library.

## 🛠️ Installation & Requirements

To run these scripts, you will need **Python 3.x**.

The `create_gabor.py` script utilizes **CuPy** for GPU acceleration. Ensure you have the appropriate CUDA toolkit installed for your system.

Install the required dependencies via pip:

```bash
pip install numpy scikit-image matplotlib PyWavelets opencv-python cupy-cuda11x
# Note: Replace 'cupy-cuda11x' with the version matching your CUDA installation (e.g., cupy-cuda12x)
```

## 📊 Appendix: Supplementary Results

The following tables present detailed ablation studies and performance comparisons discussed in the article.

### Comparative Analysis of Feature Extraction Methods

*Comparison of results when replacing the first image channel with an image generated using feature extraction methods. T and F represent theta and frequency variables, respectively.*

| **Méthode** | **MAE LO ↓** | **PC LO ↑** | **MAE GE ↓** | **PC GE ↑** |
| :--- | :---: | :---: | :---: | :---: |
| **Variante de ViTReg-IP** | 0.828 | 0.737 | **0.838** | **0.840** |
| **DWT Variants** | | | | |
| DWT bior 1.3 | 0.881 | 0.699 | 0.868 | 0.832 |
| DWT bior 2.2 | 0.867 | 0.718 | 0.882 | 0.826 |
| DWT coif 1 | 0.867 | 0.705 | 0.872 | 0.826 |
| DWT coif 5 | 0.859 | 0.703 | 0.878 | 0.825 |
| DWT haar (db1) | 0.855 | 0.694 | 0.881 | 0.825 |
| DWT db2 | 0.842 | 0.711 | 0.891 | 0.825 |
| DWT db4 | 0.853 | 0.709 | 0.875 | 0.831 |
| DWT db8 | 0.848 | 0.713 | 0.894 | 0.824 |
| DWT dmey | 0.845 | 0.721 | 0.860 | 0.828 |
| DWT rbio2.2 | 0.863 | 0.712 | 0.887 | 0.823 |
| DWT sym4 | 0.856 | 0.706 | 0.881 | 0.826 |
| DWT sym8 | 0.852 | 0.711 | 0.888 | 0.823 |
| **HOG** | 0.855 | 0.710 | 0.892 | 0.820 |
| **Gabor** | | | | |
| GABOR Theta=0, Frequency=0.05 | 0.850 | 0.711 | 0.960 | 0.795 |
| GABOR Theta=0, Frequency=0.1 | 0.836 | 0.714 | 0.931 | 0.804 |
| GABOR Theta=0, Frequency=0.3 | 0.841 | 0.718 | 0.902 | 0.822 |
| GABOR Theta=0, Frequency=0.6 | 0.836 | 0.721 | 0.900 | 0.825 |
| GABOR Theta=0, Frequency=0.9 | 0.833 | 0.716 | 1.063 | 0.733 |
| GABOR Theta=0.45, Frequency=0.05 | 0.845 | 0.722 | 0.950 | 0.790 |
| GABOR Theta=0.45, Frequency=0.1 | 0.845 | 0.709 | 0.927 | 0.804 |
| GABOR Theta=0.45, Frequency=0.3 | 0.851 | 0.717 | 0.896 | 0.819 |
| GABOR Theta=0.45, Frequency=0.6 | 0.837 | 0.710 | 0.896 | 0.821 |
| GABOR Theta=0.45, Frequency=0.9 | 0.844 | 0.709 | 1.052 | 0.737 |
| GABOR Theta=0.9, Frequency=0.05 | 0.863 | 0.718 | 0.998 | 0.773 |
| GABOR Theta=0.9, Frequency=0.1 | 0.847 | 0.708 | 0.966 | 0.797 |
| GABOR Theta=0.9, Frequency=0.3 | 0.845 | 0.707 | 0.917 | 0.819 |
| GABOR Theta=0.9, Frequency=0.6 | 0.851 | 0.716 | 0.889 | 0.819 |
| GABOR Theta=0.9, Frequency=0.9 | 0.845 | 0.714 | 1.001 | 0.767 |
| **LBP Variants** | | | | |
| BGLBP | 0.931 | 0.626 | 0.987 | 0.774 |
| CSLBP | 0.847 | 0.720 | 0.905 | 0.815 |
| CSLDP | 0.841 | 0.716 | 0.899 | 0.820 |
| CSSILTP | 0.887 | 0.672 | 0.969 | 0.784 |
| ELBP | 0.897 | 0.638 | 0.941 | 0.773 |
| OLBP | 0.920 | 0.629 | 0.955 | 0.782 |
| SILTP | 0.900 | 0.666 | 0.906 | 0.811 |
| VARLBP | 0.863 | 0.709 | 0.914 | 0.812 |
| SCSLBP | 0.827 | 0.723 | 0.892 | 0.818 |
| **XCSLBP** | **0.820** | **0.724** | 0.918 | 0.813 |

## 📝 Citation
Anonymous while the article is in review process
If you use this code or the results in your research, please cite the original article:

> [Author Names]. "Évaluation d’une Stratégie d’Hybridation Primitives-Transformeur pour Quantifier la Sévérité de Pneumonies". 2026.
```
