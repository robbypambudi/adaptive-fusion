@AGENTS.md

## Khusus Claude Code
- Slash command per fase: `/fase-0-setup`, `/fase-1-data`, `/fase-2-koding <tugas>`, `/fase-3-validasi`, `/fase-4-eksekusi`.
- Subagent: `coder` (menulis kode) dan `reviewer` (memeriksa kode, read-only).
- Setelah menulis kode, minta `reviewer` memeriksanya sebelum dianggap selesai.
