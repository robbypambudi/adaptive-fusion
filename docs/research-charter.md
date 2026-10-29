# Piagam Riset

> Diisi bersama tim di **Fase 0**. Semua agen dan anggota tim mengacu ke dokumen ini.
> Bagian bertanda **(usulan)** belum disetujui tim; konfirmasi lalu hapus tandanya.
> Konteks: `docs/konteks-notebook.md` (notebook Fase 1) dan `docs/spec-gmm-router.md` (GMM router).

## Tim
| Nama | Tugas |
|---|---|
| (isi nama teman) | Fase 1 notebook: data, embedding, 3 fusion global, analisis per-sampel |
| Robby | GMM-based fusion router (Dynamic Fusion) dan perbandingan dengan Global Fusion |

## Pertanyaan & hipotesis
- **Pertanyaan riset:** Apakah metode fusion terbaik berbeda antar sampel, dan apakah memilih fusion
  per sampel (*sample-wise conditional fusion*) lewat router berbasis GMM menghasilkan akurasi lebih
  tinggi daripada satu fusion global?
- **Hipotesis (usulan):** Dynamic Fusion (GMM + MLP router) mencapai akurasi test rata-rata (≥ 3 seed)
  lebih tinggi daripada fusion global terbaik (dipilih di val), dan lebih tinggi daripada router mayoritas.
  Hipotesis dianggap salah jika Dynamic ≤ Global terbaik atau tidak berbeda dari router mayoritas.

## Fusion yang diteliti
- **Sumber/modalitas yang digabung:** gambar produk (DINOv2-base, 768-d), judul review (BGE-base-en-v1.5,
  768-d), teks review (BGE-base-en-v1.5, 768-d).
- **Jenis fusion:** intermediate (mean, concat, cross-attention) + adaptif per sampel (router).
- **Ide metode usulan:** GMM pada embedding (Scaler → PCA → GMM, fit di train) menghasilkan probabilitas
  cluster per sampel. MLP router memetakan probabilitas ini ke metode fusion terbaik. Labelnya argmax
  P(kelas benar) dari prediksi out-of-fold di train.

## Dataset
| Nama | Sumber/URL | Lisensi | Ukuran |
|---|---|---|---|
| Amazon Reviews 2023, kategori Appliances | https://amazon-reviews-2023.github.io/ (McAuley Lab, UCSD) | (cek di halaman dataset) | Run zip: 1000 review bergambar → 484 valid. Run besar: ~49.700 valid (9.944 test). Data utama: **belum diputuskan (Q0 di spec)** |

**Label:** rating 1–3 → 0 (negatif), 4–5 → 1 (positif). Run zip: 100 negatif, 384 positif.
Run besar (test): 3.689 negatif, 6.255 positif.

**Cara split:** stratified 64/16/20 dengan `random_state=42` (identik notebook) → run zip: train 309,
val 78, test 97. Split tetap untuk semua seed **(usulan, Q2 di spec)**.

## Baseline
- `mean` (versi notebook: projection cabang → rata-rata → post-projection)
- `concat` (versi notebook)
- `cross_attn` (query = image, key/value = title + text)
- Router mayoritas, router acak (untuk menguji apakah router benar-benar belajar)
- Oracle per sampel (batas atas, bukan pembanding yang adil)
- Unimodal terbaik: belum ada di notebook **(usulan: opsional)**

## Evaluasi
- **Metrik utama:** accuracy di test set.
- **Metrik tambahan:** F1 kelas positif, precision, recall, ROC-AUC; akurasi router dan distribusi
  pilihan router; uji McNemar Dynamic vs Global terbaik.
- **Jumlah seed per eksperimen:** 3 (42, 43, 44) **(usulan)**.

## Rencana eksperimen
| ID | Tujuan | Yang dibandingkan |
|---|---|---|
| EXP-001 | Baseline Global Fusion | mean, concat, cross_attn (model Full, seluruh test) |
| EXP-002 | Uji metode usulan | Dynamic (GMM + MLP router) vs Global terbaik, router mayoritas/acak, oracle |
| EXP-003 | Ablation **(usulan)** | MLP pada fitur PCA tanpa GMM; GMM hard-cluster → metode terbaik per cluster tanpa MLP; variasi K |
