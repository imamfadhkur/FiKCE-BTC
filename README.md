# FIKCE-BTC: Filtered Keyword Contextual Embedding for Bloom's Taxonomy Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9DC3E.svg)](https://huggingface.co/)

This repository contains the official implementation and supporting material for our paper titled **FIKCE-BTC (Filtered Keyword Contextual Embedding for Bloom's Taxonomy Classification)**.

The method integrates class-specific keyword extraction based on **TF-IDF filtering** with Transformer-based semantic representations (BERT and RoBERTa) to automatically classify cognitive questions according to Bloom's Taxonomy.

---

## Repository Structure

This repository organizes experiments into two reproducible scenarios: **with undersampling** (balanced data) and **without undersampling** (natural data distribution).

```text
FIKCE-BTC/
├── data/
│   └── README.md                             # Dataset download instructions
├── notebooks/
│   ├── with_undersampling/
│   │   ├── 01_baseline_models.ipynb          # Baseline experiments (CNN, LSTM, CNN-LSTM)
│   │   ├── 02_proposed_bert_fikce.ipynb      # Proposed method: BERT + FIKCE
│   │   └── 03_proposed_roberta_fikce.ipynb   # Proposed method: RoBERTa + FIKCE
│   ├── without_undersampling/
│   │   ├── 01_baseline_models.ipynb          # Baseline experiments (without undersampling)
│   │   ├── 02_proposed_bert_fikce.ipynb      # Proposed method: BERT + FIKCE (without undersampling)
│   │   └── 03_proposed_roberta_fikce.ipynb   # Proposed method: RoBERTa + FIKCE (without undersampling)
│   └── 04_clustering_analysis.ipynb          # Exploratory clustering analysis (K-Means & UMAP)
├── requirements.txt                          # Library and dependency specifications
└── README.md                                 # This file
```

---

## Installation & Environment Setup

To run the code in this repository, use a virtual environment such as **conda** or **venv**.

### 1. Clone the Repository

```bash
git clone https://github.com/imamfadhkur/FiKCE-BTC.git
cd FIKCE-BTC
```

### 2. Install Dependencies

Make sure you're on **Python 3.10** or later, then run:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download the **Bloom's Taxonomy Classification** dataset from Kaggle.

Place the downloaded `.csv` file in the `data/` folder as described in `data/README.md`.

---

## Main Experimental Results

The tables below summarize model performance on the held-out test set ($N = 1,754$) using leak-free training-split keyword extraction and early stopping on a 10% validation split ($N = 702$).

### Without Undersampling (Table 4, $N_{\text{train}} = 6,311$)

| Embedding Model | Keyword Features | Architecture | Accuracy | Precision | Recall | F1-Score | Implementation Script |
|-----------------|------------------|--------------|---------:|----------:|-------:|---------:|-----------------------|
| BERT | None (Baseline) | CNN | 0.774 | 0.786 | 0.774 | 0.775 | `01_baseline_models.ipynb` |
| BERT | None (Baseline) | LSTM | 0.790 | 0.793 | 0.790 | 0.789 | `01_baseline_models.ipynb` |
| BERT | None (Baseline) | CNN-LSTM | 0.795 | 0.801 | 0.795 | 0.795 | `01_baseline_models.ipynb` |
| BERT | **TF-IDF + Filtered Keyword** | CNN | 0.778 | 0.780 | 0.778 | 0.777 | `02_proposed_bert_fikce.ipynb` |
| BERT | **TF-IDF + Filtered Keyword** | LSTM | **0.795** | **0.800** | **0.795** | **0.794** | `02_proposed_bert_fikce.ipynb` |
| BERT | **TF-IDF + Filtered Keyword** | CNN-LSTM | 0.792 | 0.795 | 0.792 | 0.792 | `02_proposed_bert_fikce.ipynb` |
| RoBERTa | None (Baseline) | CNN | 0.795 | 0.797 | 0.795 | 0.794 | `01_baseline_models.ipynb` |
| RoBERTa | None (Baseline) | LSTM | 0.769 | 0.772 | 0.769 | 0.770 | `01_baseline_models.ipynb` |
| RoBERTa | None (Baseline) | CNN-LSTM | 0.786 | 0.790 | 0.786 | 0.785 | `01_baseline_models.ipynb` |
| RoBERTa | **TF-IDF + Filtered Keyword** | CNN | 0.769 | 0.779 | 0.769 | 0.771 | `03_proposed_roberta_fikce.ipynb` |
| RoBERTa | **TF-IDF + Filtered Keyword** | LSTM | **0.790** | **0.796** | **0.790** | **0.789** | `03_proposed_roberta_fikce.ipynb` |
| RoBERTa | **TF-IDF + Filtered Keyword** | CNN-LSTM | 0.784 | 0.787 | 0.784 | 0.783 | `03_proposed_roberta_fikce.ipynb` |

### With Undersampling on Training Split Only (Table 5, $N_{\text{train}} = 3,378$)

| Embedding Model | Keyword Features | Architecture | Accuracy | Precision | Recall | F1-Score | Implementation Script |
|-----------------|------------------|--------------|---------:|----------:|-------:|---------:|-----------------------|
| BERT | None (Baseline) | CNN | 0.742 | 0.749 | 0.742 | 0.743 | `01_baseline_models.ipynb` |
| BERT | None (Baseline) | LSTM | 0.744 | 0.745 | 0.744 | 0.744 | `01_baseline_models.ipynb` |
| BERT | None (Baseline) | CNN-LSTM | 0.750 | 0.759 | 0.750 | 0.751 | `01_baseline_models.ipynb` |
| BERT | **TF-IDF + Filtered Keyword** | CNN | 0.742 | 0.751 | 0.742 | 0.743 | `02_proposed_bert_fikce.ipynb` |
| BERT | **TF-IDF + Filtered Keyword** | LSTM | 0.745 | 0.748 | 0.745 | 0.745 | `02_proposed_bert_fikce.ipynb` |
| BERT | **TF-IDF + Filtered Keyword** | CNN-LSTM | **0.750** | **0.759** | **0.750** | **0.751** | `02_proposed_bert_fikce.ipynb` |
| RoBERTa | None (Baseline) | CNN | 0.752 | 0.758 | 0.752 | 0.744 | `01_baseline_models.ipynb` |
| RoBERTa | None (Baseline) | LSTM | 0.733 | 0.744 | 0.733 | 0.735 | `01_baseline_models.ipynb` |
| RoBERTa | None (Baseline) | CNN-LSTM | 0.730 | 0.748 | 0.730 | 0.733 | `01_baseline_models.ipynb` |
| RoBERTa | **TF-IDF + Filtered Keyword** | CNN | **0.754** | **0.759** | **0.754** | **0.750** | `03_proposed_roberta_fikce.ipynb` |
| RoBERTa | **TF-IDF + Filtered Keyword** | LSTM | 0.736 | 0.739 | 0.736 | 0.737 | `03_proposed_roberta_fikce.ipynb` |
| RoBERTa | **TF-IDF + Filtered Keyword** | CNN-LSTM | 0.745 | 0.749 | 0.745 | 0.746 | `03_proposed_roberta_fikce.ipynb` |

> **Note**
>
> Full outputs for all 18 configurations per setting (including standard unfiltered TF-IDF, Table 2 filtered keywords, and Table 6a/6b per-class and confusion-pair analyses) are saved directly in the executed notebooks under `notebooks/without_undersampling/` and `notebooks/with_undersampling/`.

---

## Reproduction Steps

To reproduce the full set of experiments, run the notebooks in the following order.

1. Go to the `notebooks/with_undersampling/` folder.

2. Run:

   ```text
   01_baseline_models.ipynb
   ```

   to obtain the baseline performance of the semantic models.

3. Run:

   ```text
   02_proposed_bert_fikce.ipynb
   03_proposed_roberta_fikce.ipynb
   ```

   to observe how the **Filtered Keyword (FIKCE)** feature affects the deep-learning classifiers.

4. Repeat the same steps in the:

   ```text
   notebooks/without_undersampling/
   ```

   folder to evaluate model performance on the imbalanced dataset.

5. For the interpretation analysis, run:

   ```text
   notebooks/04_clustering_analysis.ipynb
   ```

   This notebook produces:

   - A 2D UMAP visualization.
   - Evaluation of the optimal K-Means value using the Silhouette Score.
   - Exploratory cluster distribution analysis of Bloom's Taxonomy cognitive levels across semantic question groups.

---
