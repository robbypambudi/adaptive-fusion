---
name: gmm-router
description: Kerjakan satu tugas (T1–T7) dari spesifikasi GMM-based fusion router dengan siklus coder → reviewer.
disable-model-invocation: true
argument-hint: "<T1..T7, mis. 'T1'>"
---

Kerjakan tugas GMM router berikut: **$ARGUMENTS**

## Konteks wajib (baca dulu, berurutan)
1. `AGENTS.md`
2. `docs/research-charter.md`
3. `docs/konteks-notebook.md`: apa yang sudah dibuat teman tim di notebook, angka referensi, dan masalah yang diketahui.
4. `docs/spec-gmm-router.md`: keputusan D1–D4, pertanyaan terbuka Q1–Q6, pipeline, checklist kebocoran, dan rencana tugas T1–T7.
5. `docs/phases/phase-2.md`

Jika argumen kosong, lihat tabel §8 di spec, usulkan tugas berikutnya yang belum selesai, lalu tanyakan dulu.

## Aturan khusus tugas ini
- Kerjakan **hanya** satu tugas T<n>. Jangan mengerjakan tugas berikutnya.
- Jika tugas menyentuh pertanyaan terbuka (Q1–Q6) yang belum dijawab di spec, **tanyakan ke pengguna
  dulu** dan tampilkan usulan default-nya. Setelah dijawab, pindahkan jawabannya ke tabel "Keputusan" di spec.
- Jangan pernah memakai `appliances_fusion_persample*.csv` (dihitung di TEST) untuk training, seleksi, atau tuning.
- Scaler, PCA, dan GMM di-fit di train saja. Label router train berasal dari OOF. Seleksi dilakukan di val. Test dipakai sekali di T7.
- Router bukan `FusionModule` (letaknya di `src/adaptive_fusion/router/`). Ketiga fusion tetap `FusionModule` dan wajib lolos tes kontrak.
- Isi `data/raw/` tidak diubah. Hasil olahan masuk ke `data/processed/`, dan angka ke `results/`.

## Langkah
1. Buat rencana singkat: file yang disentuh, tes yang ditambah, dan pertanyaan terbuka yang relevan.
   Jika lebih dari 3 file, minta persetujuan dulu.
2. Delegasikan implementasi ke subagent `coder`. Sertakan path spec dan nomor tugas.
3. Jalankan `make lint && make test`.
4. Delegasikan ke subagent `reviewer` dengan checklist kebocoran §7 di spec. Jika hasilnya PERLU
   PERBAIKAN, kembalikan temuan ke `coder` (maksimal 3 putaran; setelah itu tanya pengguna).
5. Ringkas hasil untuk pengguna: file yang diubah, hasil tes, keputusan yang diambil, dan tugas berikutnya.
