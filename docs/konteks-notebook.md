# Konteks: Notebook Fase 1 (`adaptive_fusion new.ipynb`)

> Ringkasan pekerjaan teman tim di notebook Colab. Ini **titik awal** eksperimen GMM router
> (`docs/spec-gmm-router.md`). Angka di sini dikutip dari output notebook sebagai **referensi**,
> belum direproduksi di repo, jadi jangan dipakai di laporan (aturan 6 `AGENTS.md`).

## Pipeline notebook (Bagian 1–10)

| Bagian | Isi | Catatan |
|---|---|---|
| 1 | Setup Colab, `HOME_DIR=/content/drive/MyDrive/Magister/DataMining` | Path Colab di-hardcode |
| 2 | Amazon Reviews 2023, kategori **Appliances**. Subset 1000 review yang punya gambar, distribusi rating dipertahankan (DuckDB) | `LIMIT` tanpa `ORDER BY` |
| 3 | Label biner: rating 1–3 → 0 (negatif), 4–5 → 1 (positif) | |
| 4 | Gambar pertama per review → **DINOv2-base** CLS (768-d) | Berhasil 484/1000 (sisanya gagal unduh → NaN) |
| 5 | **BGE-base-en-v1.5** (768-d, dinormalisasi), `title` dan `text` di-encode **terpisah** | 3 modalitas: image, title, text |
| 6 | Filter sampel valid (484), split stratified 64/16/20, `random_state=42` | Train 309 / Val 78 / Test 97 |
| 7 | 3 model fusion: `mean`, `concat`, `cross_attn` (detail di bawah) | |
| 8 | Training satu model per fusion, checkpoint terbaik = **val loss** terendah | |
| 9 | Evaluasi global di test | |
| 10 | Analisis per-sampel di **test**: confidence tiap fusion, pemenang per sampel | Output = `appliances_fusion_persample.csv` |

## Arsitektur (sama untuk ketiga fusion, hanya modul fusion yang berbeda)

```
image 768 ─ BranchProjection ─┐
title 768 ─ BranchProjection ─┼─ Fusion ─ post: Linear→LayerNorm→GELU (512) ─ head ─ 2 logit
text  768 ─ BranchProjection ─┘
BranchProjection = Linear(768→512) → LayerNorm → GELU
head             = Linear(512→256) → GELU → Dropout(0.2) → Linear(256→2)
```
- `mean`: rata-rata elemen dari 3 vektor cabang (512) → post.
- `concat`: gabung 3 vektor cabang (1536) → post.
- `cross_attn`: `nn.MultiheadAttention(512, 4 head)`, **query = image**, key/value = [title, text]
  (2 token) → LayerNorm → post.

Hyperparameter: `D_BRANCH=512, D_FUSION=512, HIDDEN=256, BATCH_SIZE=128, EPOCHS=300, LR=1e-3,
WEIGHT_DECAY=1e-4 (AdamW), PATIENCE=300, SEED=42`, loss `CrossEntropyLoss` tanpa bobot kelas.

## Statistik data (dihitung ulang dari artefak di zip, split identik)

| Split | n | negatif (0) | positif (1) |
|---|---|---|---|
| Semua valid | 484 | 100 | 384 |
| Train | 309 | 64 | 245 |
| Val | 78 | 16 | 62 |
| Test | 97 | 20 | 77 |

- Tidak ada overlap indeks antar split.
- Norma embedding: image ≈ 46.5 (DINOv2 tidak dinormalisasi), title/text = 1.0 (BGE dinormalisasi).
  → skala antar modalitas sangat berbeda.
- Split bisa direproduksi persis: `subset_idx` di CSV per-sampel cocok dengan split yang dibangun ulang.

## Hasil referensi notebook (seed 42, satu run)

| Metode | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| cross_attn | 0.9485 | 0.9669 | 0.9916 |
| mean | 0.9381 | 0.9610 | 0.9753 |
| concat | 0.9381 | 0.9610 | 0.9831 |

Analisis per-sampel (test, 97 sampel):
- `winner` (benar + confidence tertinggi): mean 71, cross_attn 15, concat 7, all_wrong 4.
- `best_loss_method` (= argmax P(kelas benar), **label router yang dipakai**): mean 71, cross_attn 16, concat 10.
- Oracle per-sampel 0.9588 vs fusion global terbaik 0.9485 → **selisih 0.0103 = 1 sampel**.
- Pemenang bergeser per kelas: positif didominasi `mean` (93%), negatif didominasi `cross_attn` (74%).

## Run kedua yang lebih besar (`appliances_fusion_persample (2).csv` di root repo)

Teman tim juga menjalankan notebook dengan subset yang jauh lebih besar. Yang tersedia di repo hanya CSV
per-sampel (test). Embedding, `appliances.csv`, checkpoint, dan nilai `MAX_ROWS` untuk run ini
**belum ada** di repo (lihat Q0 di spec).

| | Run zip (1000 baris) | Run besar (CSV) |
|---|---|---|
| Sampel test | 97 | **9.944** (≈ 20% dari ~49.700 sampel valid) |
| `subset_idx` maks | 999 | 84.948 |
| Kelas test (neg/pos) | 20 / 77 (21% neg) | 3.689 / 6.255 (37% neg), artinya cara sampling berbeda |
| Rating test 1/2/3/4/5 | — | 2.484 / 580 / 625 / 897 / 5.358 |
| Akurasi mean / concat / cross_attn | 0.9381 / 0.9381 / 0.9485 | 0.9449 / **0.9474** / 0.9465 |
| Oracle per-sampel | 0.9588 | 0.9607 |
| Selisih oracle − global terbaik | 0.0103 (1 sampel) | **0.0133 (~132 sampel)** |
| `best_loss_method` (label router) | mean 71, cross_attn 16, concat 10 | concat 7.467, cross_attn 1.784, mean 693 |
| `winner` | mean 71, cross_attn 15, concat 7, all_wrong 4 | concat 7.414, cross_attn 1.591, mean 548, all_wrong 391 |

Per kelas (label router, run besar): positif → concat 86%, cross_attn 11%, mean 4%;
negatif → concat 57%, cross_attn 30%, mean 13%.

Implikasi: di run besar, metode dominan bergeser dari `mean` ke `concat`. Ruang perbaikan oracle
(~132 sampel) cukup besar untuk diukur. CSV ini **juga dihitung di test**, jadi aturan kebocorannya
sama: hanya untuk referensi dan evaluasi.

## Artefak di `datasets/amazon_review_2023-*.zip`

`Appliances.jsonl(.gz)`, `meta_Appliances.jsonl(.gz)`, `appliances.csv` (subset 1000 baris),
`appliances_{image,title,text}_emb.npy` (1000×768, baris gagal = NaN),
`fusion_{mean,concat,cross_attn}.pt` (checkpoint notebook),
`appliances_fusion_metrics.csv`, `appliances_fusion_persample.csv`, `appliances_fusion_persample_summary.csv`.

## Masalah yang perlu diingat

1. **Model hafal train.** Train loss mencapai 0.0000. Val loss terbaik ada di epoch 4 (mean),
   5 (concat), dan 7 (cross_attn) dari 300. Early stopping tidak pernah aktif karena `PATIENCE = EPOCHS`.
   Akibatnya, confidence di data train ≈ 1.0 untuk semua metode dan **tidak bisa** dipakai sebagai label router.
2. **CSV per-sampel dihitung di TEST.** File ini tidak boleh dipakai untuk melatih atau memilih router.
   Pemakaiannya hanya untuk evaluasi akhir dan cek kewajaran.
3. **Test set kecil (97) di run zip.** Satu sampel = 1.03% akurasi, dan ruang perbaikan maksimal (oracle)
   hanya ~1 sampel. Hasilnya kemungkinan tidak konklusif. Run besar (9.944 test) tidak punya masalah ini,
   tetapi artefaknya belum ada di repo.
4. **Reproduksibilitas data.** Hasil unduh gambar bisa berbeda antar run, dan `LIMIT` tanpa `ORDER BY`
   tidak menjamin baris yang sama. Karena itu embedding cache (`.npy`) di zip diperlakukan sebagai
   data sumber yang dibekukan.
5. **Cross-attention lemah.** Query 1 token dan key/value 2 token, sehingga attention praktis menjadi
   weighted sum (sudah dicatat di notebook).
6. **Satu seed saja.** Charter meminta minimal 3 seed.
