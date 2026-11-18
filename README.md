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
│   └── blooms_taxonomy_dataset.csv          # Included Bloom's Taxonomy question dataset
├── notebooks/
│   ├── with_undersampling/
│   │   ├── 01_baseline_models.ipynb         # Semantic-only BERT/RoBERTa baselines
│   │   ├── 02_proposed_bert_fikce.ipynb     # BERT + TF-IDF/FIKCE experiments
│   │   └── 03_proposed_roberta_fikce.ipynb  # RoBERTa + TF-IDF/FIKCE experiments
│   ├── without_undersampling/
│   │   ├── 01_baseline_models.ipynb         # Semantic-only baselines on imbalanced data
│   │   ├── 02_proposed_bert_fikce.ipynb     # BERT + TF-IDF/FIKCE experiments
│   │   └── 03_proposed_roberta_fikce.ipynb  # RoBERTa + TF-IDF/FIKCE experiments
│   └── 04_clustering_analysis.ipynb          # UMAP, K-Means, and HDBSCAN analysis
├── output/
│   ├── results_bundle_baseline_undersample*.zip
│   ├── results_bundle_bert_undersample*.zip
│   └── results_bundle_roberta_undersample*.zip
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## Installation & Environment Setup

Use Python 3.10 or later and a virtual environment. A CUDA-enabled GPU is recommended for the Transformer notebooks; CPU execution is supported but will be slower.

The experiment notebooks download `bert-base-uncased` and `roberta-base` from Hugging Face and require internet access on the first run.

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

### 3. Dataset

The repository includes `data/blooms_taxonomy_dataset.csv`. It contains **8,767 questions** with the following two columns:

- `Questions`: the source question text.
- `Category`: Bloom's Taxonomy label (`BT1` through `BT6`).

The notebooks accept the local dataset path, a Kaggle dataset path, or an automatically discovered single CSV under `/kaggle/input/`.

The original category distribution is:

| Category | Samples |
|----------|--------:|
| BT1 | 2582 |
| BT2 | 1801 |
| BT3 | 1508 |
| BT4 | 1293 |
| BT5 | 783 |
| BT6 | 800 |

For the clustering notebook, the notebooks also download NLTK resources such as stopwords and lemmatization data when needed.

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
> Full executed outputs for all 24 configurations per setting (6 baseline, 9 BERT, and 9 RoBERTa configurations) are saved in the notebooks under `notebooks/without_undersampling/` and `notebooks/with_undersampling/`. The notebooks include standard TF-IDF comparisons, filtered FIKCE keywords, per-class analyses, and confusion-pair analyses.

---

## Saved Result Bundles

The `output/` directory contains six ZIP bundles generated by the executed notebooks:

- Baseline BERT/RoBERTa results for undersampled and natural training splits.
- BERT FIKCE experiment results for both training splits.
- RoBERTa FIKCE experiment results for both training splits.

Each bundle contains summary CSV/JSON files, model-specific result folders, metrics, misclassified examples, and comparison plots where applicable.

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
