# Fase 2: Ngoding

**Tujuan:** kode metode fusion dan pipeline training/evaluasi.

## Urutan tugas (satu tugas = satu PR kecil)
1. Encoder per modalitas
2. Metode fusion usulan di `src/adaptive_fusion/fusion/<nama>.py` (pakai `@register_fusion`)
3. Model lengkap: encoder → fusion → classifier/head
4. `src/adaptive_fusion/train.py`: training, simpan model terbaik berdasarkan **validation**, evaluasi test sekali di akhir, simpan hasil ke `results/<nama-run>.json`
5. Config eksperimen di `configs/`

## Cara kerja per tugas
1. Minta AI membuat rencana singkat dulu.
2. Subagent `coder` menulis kode + tes.
3. `make lint && make test`.
4. Subagent `reviewer` memeriksa. Kalau ada masalah, perbaiki lalu review ulang (maksimal 3 kali; setelah itu tanya manusia).

## Checklist selesai
- [ ] Semua metode di charter (termasuk baseline) bisa dijalankan
- [ ] `make test` lolos, termasuk tes kontrak fusion
- [ ] Training bisa dijalankan dengan data kecil dalam beberapa menit (smoke test)
