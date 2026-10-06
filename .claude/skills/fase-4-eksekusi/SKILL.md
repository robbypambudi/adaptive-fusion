---
name: fase-4-eksekusi
description: Fase 4. Jalankan eksperimen sesuai rencana dan verifikasi hasilnya.
disable-model-invocation: true
argument-hint: "[EXP-ID, opsional]"
---

Kerjakan Fase 4 sesuai `docs/phases/phase-4.md`. Target: $ARGUMENTS

1. Pastikan catatan eksperimen di `docs/eksperimen/` sudah ada (termasuk perkiraan hasil). Jika belum, buat bersama pengguna.
2. Tampilkan rencana run (config × seed, perkiraan waktu) dan minta persetujuan.
3. Jalankan eksperimen. Jangan ubah kode di fase ini; jika ada bug, laporkan dan kembali ke Fase 2.
4. Rekap hasil dari file `results/` dengan skrip (mean ± std).
5. Minta subagent `reviewer` mencocokkan angka di laporan dengan `results/`.
6. Tampilkan hasil dan checklist Fase 4.
