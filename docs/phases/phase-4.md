# Fase 4: Jalankan & Pastikan Benar

**Tujuan:** menjalankan eksperimen sesuai rencana dan memastikan hasilnya benar.

## Langkah
1. Untuk setiap eksperimen, buat catatan singkat di `docs/eksperimen/EXP-xxx.md`: tujuan, config, seed, dan **perkiraan hasil** sebelum dijalankan.
2. Jalankan setiap config dengan minimal 3 seed. Hasil tersimpan di `results/`.
3. Rekap hasil (mean ± std) dari file di `results/` dengan skrip, bukan disalin manual.
4. Minta subagent `reviewer` mencocokkan angka di laporan dengan file `results/`.
5. Tulis kesimpulan: apakah hipotesis terbukti? Hasil negatif juga dilaporkan.

## Checklist selesai
- [ ] Semua eksperimen di charter sudah dijalankan (atau alasan batal dicatat)
- [ ] Setiap angka di laporan bisa ditelusuri ke file di `results/`
- [ ] Ada perintah untuk mereproduksi hasil utama
