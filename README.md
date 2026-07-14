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
│   └── 04_clustering_analysis.ipynb          # Subject-domain analysis (K-Means & UMAP)
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

The table below summarizes model performance under the **undersampling** scenario (balanced dataset).

| Embedding Model | Additional Feature | Architecture | Accuracy (%) | F1-Score | Implementation Script |
|-----------------|---------------------|--------------|-------------:|---------:|------------------------|
| BERT | None (Baseline) | CNN | 79.15 | 79.06 | `01_baseline_models.ipynb` |
| BERT | None (Baseline) | CNN-LSTM | 80.11 | 79.67 | `01_baseline_models.ipynb` |
| BERT | **FIKCE (Proposed)** | CNN-LSTM | **82.45** | **82.26** | `02_proposed_bert_fikce.ipynb` |
| RoBERTa | None (Baseline) | LSTM | 79.36 | 79.38 | `01_baseline_models.ipynb` |
| RoBERTa | **FIKCE (Proposed)** | CNN-LSTM | 83.94 | 83.81 | `03_proposed_roberta_fikce.ipynb` |
| RoBERTa | **FIKCE (Proposed)** | CNN | **85.21** | **85.14** | `03_proposed_roberta_fikce.ipynb` |

> **Note**
>
> All other experimental variations, including combinations with standard TF-IDF and the **without-undersampling** scenario, are available and reproducible from the `notebooks/` directory.

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
   - An analysis of subject-domain variation based on the distribution of questions.

---
