# Fase 3: Validasi Kode

**Tujuan:** memastikan kode buatan AI benar sebelum menjalankan eksperimen besar.

## Langkah
1. Subagent `reviewer` memeriksa seluruh `src/`, terutama poin di bawah.
2. Jalankan sanity check di data kecil:
   - Model bisa overfit 1 batch kecil (loss mendekati 0).
   - Label diacak → akurasi turun ke level tebakan acak.
   - Satu modalitas di-nol-kan → hasil berubah (fusion benar-benar memakai semua modalitas).
   - Seed sama dijalankan 2x → hasil sama.
3. Perbaiki masalah yang ditemukan (kembali ke cara kerja Fase 2).
4. Tulis ringkasan temuan & perbaikan di deskripsi PR.

## Checklist kebocoran data
- [ ] Normalisasi/preprocessing di-fit hanya di train
- [ ] Tidak ada sampel/grup yang sama di train dan test
- [ ] Test set tidak dipakai untuk tuning atau early stopping
- [ ] Augmentasi hanya untuk train
- [ ] Baseline dan metode usulan dilatih dengan setting yang setara
