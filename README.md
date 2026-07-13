# FIKCE-BTC: Filtered Keyword Contextual Embedding for Bloom's Taxonomy Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9DC3E.svg)](https://huggingface.co/)

Repositori ini berisi kode implementasi resmi dan material pendukung untuk riset **FIKCE-BTC (Filtered Keyword Contextual Embedding for Bloom's Taxonomy Classification)**.

Penelitian ini mengusulkan metode integrasi ekstraksi kata kunci berbasis **TF-IDF Filtering** spesifik-kelas dengan model representasi semantik berbasis Transformer (BERT dan RoBERTa) untuk mengklasifikasikan pertanyaan kognitif secara otomatis berdasarkan Taksonomi Bloom.

---

## 📁 Struktur Repositori

Repositori ini diatur untuk memudahkan reproduksi eksperimen dengan skenario penggunaan **undersampling** (untuk data seimbang) dan tanpa **undersampling** (data natural).

```text
FIKCE-BTC/
├── data/
│   └── README.md                             # Instruksi pengunduhan dataset
├── notebooks/
│   ├── with_undersampling/
│   │   ├── 01_baseline_models.ipynb          # Eksperimen Baseline (CNN, LSTM, CNN-LSTM)
│   │   ├── 02_proposed_bert_fikce.ipynb      # Proposed Method: BERT + FIKCE
│   │   └── 03_proposed_roberta_fikce.ipynb   # Proposed Method: RoBERTa + FIKCE
│   ├── without_undersampling/
│   │   ├── 01_baseline_models.ipynb          # Eksperimen Baseline (Tanpa Undersampling)
│   │   ├── 02_proposed_bert_fikce.ipynb      # Proposed Method: BERT + FIKCE (Tanpa Undersampling)
│   │   └── 03_proposed_roberta_fikce.ipynb   # Proposed Method: RoBERTa + FIKCE (Tanpa Undersampling)
│   └── 04_clustering_analysis.ipynb          # Analisis Subject-Domain (K-Means & UMAP)
├── requirements.txt                          # Spesifikasi library & dependencies
└── README.md                                 # File ini
```

---

## 🚀 Instalasi & Persiapan Lingkungan

Untuk menjalankan kode di dalam repositori ini, disarankan menggunakan virtual environment seperti **conda** atau **venv**.

### 1. Clone Repositori

```bash
git clone https://github.com/username-anda/FIKCE-BTC.git
cd FIKCE-BTC
```

### 2. Instalasi Dependencies

Pastikan Anda menggunakan **Python 3.10** atau lebih baru, kemudian jalankan:

```bash
pip install -r requirements.txt
```

### 3. Persiapan Dataset

Unduh dataset **Bloom's Taxonomy Classification** dari Kaggle.

Selanjutnya, letakkan file `.csv` yang telah diunduh ke dalam folder `data/` sesuai instruksi pada `data/README.md`.

---

## 📊 Hasil Eksperimen Utama

Berikut adalah ringkasan hasil performa model pada skenario **undersampling** (dataset seimbang).

| Embedding Model | Fitur Tambahan | Arsitektur | Accuracy (%) | F1-Score | Skrip Implementasi |
|-----------------|----------------|------------|-------------:|---------:|--------------------|
| BERT | None (Baseline) | CNN | 79.15 | 79.06 | `01_baseline_models.ipynb` |
| BERT | None (Baseline) | CNN-LSTM | 80.11 | 79.67 | `01_baseline_models.ipynb` |
| BERT | **FIKCE (Proposed)** | CNN-LSTM | **82.45** | **82.26** | `02_proposed_bert_fikce.ipynb` |
| RoBERTa | None (Baseline) | LSTM | 79.36 | 79.38 | `01_baseline_models.ipynb` |
| RoBERTa | **FIKCE (Proposed)** | CNN-LSTM | 83.94 | 83.81 | `03_proposed_roberta_fikce.ipynb` |
| RoBERTa | **FIKCE (Proposed)** | CNN | **85.21** | **85.14** | `03_proposed_roberta_fikce.ipynb` |

> **Catatan**
>
> Seluruh variasi eksperimen lainnya, termasuk kombinasi dengan TF-IDF standar dan skenario **tanpa undersampling**, tersedia dan dapat direproduksi melalui direktori `notebooks/`.

---

## 🛠️ Tahapan Reproduksi (Step-by-Step)

Untuk mereproduksi seluruh eksperimen, jalankan notebook secara berurutan berikut ini.

1. Masuk ke folder `notebooks/with_undersampling/`.

2. Jalankan:

   ```text
   01_baseline_models.ipynb
   ```

   untuk memperoleh performa dasar model semantik.

3. Jalankan:

   ```text
   02_proposed_bert_fikce.ipynb
   03_proposed_roberta_fikce.ipynb
   ```

   untuk mengamati integrasi fitur **Filtered Keyword (FIKCE)** ke dalam classifier berbasis Deep Learning.

4. Lakukan langkah yang sama pada folder:

   ```text
   notebooks/without_undersampling/
   ```

   untuk mengevaluasi performa model pada kondisi dataset yang tidak seimbang (*imbalanced dataset*).

5. Untuk analisis interpretasi, jalankan:

   ```text
   notebooks/04_clustering_analysis.ipynb
   ```

   Notebook ini menghasilkan:

   - Visualisasi UMAP 2D.
   - Evaluasi nilai terbaik K-Means menggunakan **Silhouette Score**.
   - Analisis variasi domain mata pelajaran berdasarkan distribusi soal.

---
