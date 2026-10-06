# AGENTS.md: Instruksi untuk Agen AI

Proyek riset **metode fusion**. Kode ditulis oleh agen AI, keputusan riset diambil oleh tim.

## Sebelum mulai
1. Baca `docs/research-charter.md` (pertanyaan riset, dataset, metrik, baseline).
2. Baca prosedur fase yang sedang dikerjakan di `docs/phases/phase-<N>.md`.
3. Kerjakan hanya lingkup fase itu.

## Perintah
- Install: `make setup`
- Tes: `make test`
- Lint: `make lint`

## Aturan
1. Jangan ubah isi `data/raw/`. Hasil olahan masuk ke `data/processed/`.
2. Pilih model/hyperparameter pakai **validation set**. Test set dipakai sekali di akhir.
3. Statistik preprocessing (normalisasi, dll.) dihitung dari **train** saja.
4. Metode fusion baru mewarisi `FusionModule`, didaftarkan dengan `@register_fusion`, dan harus lolos `tests/test_fusion_contract.py`. Jangan melemahkan tes agar lolos.
5. Selalu pakai `seed_everything(seed)` dan simpan hyperparameter di `configs/`, bukan di kode.
6. Angka di laporan harus berasal dari file hasil di `results/`, bukan diketik manual.
7. Jika ada yang ambigu, tanya ke manusia. Jangan berasumsi.

## Gaya kode
Python ≥ 3.11, type hints, `ruff`. Komentar boleh bahasa Indonesia, nama variabel bahasa Inggris.
