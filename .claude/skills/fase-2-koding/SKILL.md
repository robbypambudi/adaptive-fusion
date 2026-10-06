---
name: fase-2-koding
description: Fase 2. Implementasi satu tugas kode dengan siklus coder → reviewer.
disable-model-invocation: true
argument-hint: "<tugas, mis. 'implementasi gated fusion'>"
---

Kerjakan tugas Fase 2 berikut sesuai `docs/phases/phase-2.md`: **$ARGUMENTS**

Jika tugas kosong, usulkan tugas berikutnya dari urutan di `phase-2.md` dan tanyakan dulu.

1. Buat rencana singkat (file yang disentuh, tes yang ditambah). Jika lebih dari 3 file, minta persetujuan dulu.
2. Delegasikan ke subagent `coder`.
3. Jalankan `make lint && make test`.
4. Delegasikan ke subagent `reviewer`. Jika hasilnya PERLU PERBAIKAN, kembalikan temuan ke `coder`. Maksimal 3 putaran; setelah itu tanya pengguna.
5. Ringkas hasilnya untuk pengguna.
