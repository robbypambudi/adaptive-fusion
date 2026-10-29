# EXP-002: Dynamic Fusion dengan GMM + MLP Router

**Status:** belum dijalankan. Spesifikasi lengkap: `docs/spec-gmm-router.md`.

## Tujuan
Menguji apakah memilih fusion per sampel lewat router (posterior GMM → MLP) lebih akurat daripada
Global Fusion terbaik (EXP-001).

## Setup
- Label router: argmax P(kelas benar) dari prediksi OOF 5-fold di train (checkpoint fold dipilih di
  inner-val, bukan di fold held-out atau val global).
- Fitur router: Scaler → PCA → GMM pada embedding image+title+text, fit di train.
- Seleksi K/PCA/covariance/hparam MLP: di val. Test dievaluasi sekali.
- Model fusion yang dipilih router = model Full dari EXP-001 (seed yang sama).
- Pembanding: Global mean/concat/cross_attn, Global terbaik (by val), router mayoritas, router acak, oracle.
- Seed: 42, 43, 44.
- Output: `results/<dataset_id>/<run_id>/gmm_router/seed<s>.json`, `results/<dataset_id>/gmm_router_summary.json`.

## Perkiraan hasil (ditulis sebelum menjalankan)
- Label router sangat timpang (run zip: `mean` ≈ 73%; run besar: `concat` ≈ 75%). Ada kemungkinan
  besar router cenderung selalu memilih metode mayoritas, sehingga Dynamic ≈ Global metode itu.
- Run zip: oracle hanya 1 sampel di atas fusion global terbaik. Perkiraan: Dynamic berada dalam ±2
  sampel dari Global terbaik, dan McNemar tidak signifikan.
- Run besar: oracle +0.0133 di atas global terbaik (concat 0.9474). Perkiraan: Dynamic merebut sebagian
  kecil selisih itu (+0.000 s.d. +0.004), karena posterior GMM dari embedding hanya menangkap sebagian
  pola "kelas negatif → cross_attn".
- Jika Dynamic > Global terbaik secara konsisten di 3 seed, itu bukti awal yang mendukung hipotesis.
  Jika tidak, hasil negatif tetap dilaporkan.

## Hasil
(diisi dari `results/` dengan skrip rekap)

## Kesimpulan
