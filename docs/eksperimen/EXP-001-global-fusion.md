# EXP-001: Global Fusion (baseline)

**Status:** belum dijalankan di repo (referensi: notebook Fase 1).

## Tujuan
Mengukur akurasi tiap fusion (`mean`, `concat`, `cross_attn`) bila dipakai untuk **seluruh** sampel test.
Model Full dari eksperimen ini juga dipakai Dynamic Fusion di EXP-002.

## Setup
- Data/split: `configs/data/amazon_appliances.yaml`, split tetap seed 42 (309/78/97).
- Model: arsitektur notebook (`docs/konteks-notebook.md`), dilatih di train, checkpoint by val loss.
- Config: `configs/` (dibuat di T2/T3).
- Seed: 42, 43, 44.
- Model Full dilatih di T3 (hanya val). Angka **test** Global dihitung di T7 bersamaan dengan Dynamic,
  setelah semua model dan router dibekukan.
- Output: `results/<dataset_id>/<run_id>/gmm_router/seed<s>.json` (bagian `test.global`).

## Perkiraan hasil (ditulis sebelum menjalankan)
- Akurasi ketiga metode sekitar 0.93–0.95, selisih antar metode ≤ 2 sampel test.
- Angka notebook (cross_attn 0.9485; mean/concat 0.9381) hanya pembanding setelah T7 selesai.
  Angka ini **tidak** dipakai sebagai syarat lulus atau dasar perbaikan kode, karena itu berarti tuning di test.

## Hasil
(diisi dari `results/` dengan skrip rekap)

## Kesimpulan
