---
name: fase-3-validasi
description: Fase 3. Validasi seluruh kode (review, sanity check, cek kebocoran data).
disable-model-invocation: true
---

Kerjakan Fase 3 sesuai `docs/phases/phase-3.md`.

1. Delegasikan review menyeluruh `src/` ke subagent `reviewer`, dengan checklist kebocoran di `phase-3.md`.
2. Jalankan sanity check (overfit 1 batch, label diacak, modalitas di-nol-kan, seed sama 2x). Simpan skripnya di `tests/` atau `src/adaptive_fusion/` agar bisa diulang.
3. Perbaiki temuan PENTING lewat subagent `coder`, lalu review ulang.
4. Tampilkan ringkasan temuan, hasil sanity check, dan checklist Fase 3.
