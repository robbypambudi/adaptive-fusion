# Keputusan Tim: GMM Router

> Jawaban atas pertanyaan terbuka di `docs/spec-gmm-router.md` §3. Dicatat di file terpisah agar spec yang
> sudah disetujui reviewer (SHA256 `64af459e…`, lihat `PLAN-REVIEW-LOG.md`) tidak berubah.

| # | Keputusan (2026-10-29) | Catatan |
|---|---|---|
| Q0 | Pengembangan dan run tahap dev (T1–T6) memakai **run zip** (`appliances_zip1000`). Run besar menjadi data utama setelah artefaknya didapat dari teman tim (config data kedua). | Embedding run besar belum ada di repo. |
| Q1 | `concat` dan `mean` di `baselines.py` **diganti** dengan versi notebook (projection cabang → fusion → post-projection). `cross_attn` ditambahkan. Nama registry tetap. | Tes kontrak tetap berlaku untuk ketiganya. |
| Q2 | Split **tetap** seed 42 untuk semua seed. Seed hanya mengubah inisialisasi model, fold OOF, GMM, dan router. | |
| Q3 | `EPOCHS = PATIENCE = 300` dipertahankan (setara notebook), checkpoint terbaik by loss seleksi. | Dicatat sebagai keterbatasan. |
| Q4 | Seleksi K/PCA/covariance/router berdasarkan **akurasi Dynamic di val**. Jika seri: K terkecil, lalu urutan grid. | |
| Q5 | 3 seed: 42, 43, 44. | |
| Q6 | Run besar diasumsikan memakai resep split yang sama sampai teman tim mengonfirmasi. | |
| Q0 (revisi) | Pengguna minta dataset penuh: **semua 84.949 review Appliances bergambar** dibangun ulang sendiri dari zip (`scripts/build_full_features.py`, config `appliances_full`). Hasil unduhan gambar bisa berbeda dari run besar teman tim, jadi angkanya tidak identik dengan run itu. | Pilihan pengguna (bukan menunggu artefak teman tim). |
| Lingkup build | Kode + tes + run nyata T1–T6 (train/val). **Test set baru dipakai (T7) setelah inspeksi kode lolos.** | Jawaban pengguna saat build. |
