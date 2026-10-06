# Fase 1: Setup Data

**Tujuan:** data siap pakai, sama untuk semua anggota, dan split-nya benar.

## Langkah
1. Tulis skrip unduh/siapkan data di `src/adaptive_fusion/data/`. Data asli disimpan di `data/raw/` dan tidak diubah.
2. Preprocessing → `data/processed/`. Statistik normalisasi dihitung dari train saja.
3. Buat split train/val/test dengan seed tetap. Jika data punya grup (subjek, sesi, waktu), split per grup.
4. Simpan pengaturan data di `configs/data/<nama>.yaml`.
5. Loader mengembalikan `dict` nama_modalitas → tensor, plus label.
6. Tulis tes sederhana di `tests/data/`: dimensi benar, tidak ada NaN, tidak ada sampel yang sama di train dan test.
7. Catat di README cara anggota lain mendapatkan data (link drive/bucket).

## Checklist selesai
- [ ] Anggota lain bisa menyiapkan data dengan perintah yang sama
- [ ] Tes data lolos
- [ ] Tidak ada overlap antar split
