# Mushroom Classification — Take-Home Challenge

## Overview
Proyek ini menjawab pertanyaan: **"Can Machine Learning Identify Whether a Mushroom is Edible or Poisonous?"**
Menggunakan dataset `mushrooms.csv` (8.124 baris, 22 fitur kategorikal), dibangun beberapa model Machine Learning dan Deep Learning untuk mengklasifikasikan jamur ke dalam kategori **Edible** atau **Poisonous** berdasarkan ciri fisiknya.

## Metodologi
1. **Data Understanding** — memeriksa struktur, tipe data, jumlah unique value, dan distribusi target.
2. **Exploratory Data Analysis (EDA)** — analisis distribusi target, distribusi fitur, dan hubungan fitur terhadap target (`odor`, `bruises`, `gill-size`, `habitat`).
3. **Data Preprocessing** — penanganan missing value (`?` pada `stalk-root` → kategori `"missing"`), pengecekan duplikat, penghapusan fitur konstan (`veil-type`), dan One-Hot Encoding untuk seluruh fitur kategorikal.
4. **Modeling** — Logistic Regression (baseline), Decision Tree, Random Forest, dan Neural Network (TensorFlow/Keras).
5. **Evaluation & Comparison** — Accuracy, Precision, Recall, F1 Score, dan Confusion Matrix untuk setiap model.
6. **Hyperparameter Tuning** — RandomizedSearchCV pada Random Forest.
7. **Model Interpretation** — Feature Importance & Permutation Importance.
8. **Regression Exploration** — target numerik eksperimental "Rarity Score" dengan Linear Regression & Random Forest Regressor, disertai pembahasan mengapa Classification lebih tepat untuk masalah ini.

## Hasil (ringkasan)
| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.9982 | 1.0000 | 0.9962 | 0.9981 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Neural Network | *(isi setelah menjalankan notebook)* | | | |

Fitur paling berpengaruh secara konsisten: **`odor`** (terutama kategori "tidak berbau") dan **`gill-size`**, sejalan dengan temuan EDA.

## Kesimpulan
Machine Learning mampu mengidentifikasi status edible/poisonous suatu jamur dengan akurasi sangat tinggi (mendekati sempurna) menggunakan fitur fisik sederhana. Model berbasis pohon (Decision Tree, Random Forest) paling optimal untuk kasus ini karena mampu menangkap aturan kategorikal non-linear secara alami. Meski demikian, performa sempurna ini merupakan karakteristik khas dataset UCI Mushroom yang sangat bersih — untuk penerapan dunia nyata tetap disarankan verifikasi oleh ahli, mengingat risiko fatal dari kesalahan False Negative.

## Struktur Repository
```
Mushroom-Classification/
│
├── data/
│   └── mushrooms.csv
│
├── notebook/
│   └── Nama_NIM_TakeHomeML.ipynb
│
└── README.md
```

## Catatan Menjalankan Notebook
Bagian Deep Learning (section 12) membutuhkan `tensorflow`. Jalankan notebook secara berurutan dari atas ke bawah (Run All) di Jupyter Notebook, JupyterLab, atau Google Colab.
