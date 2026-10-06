# Sumber Rujukan

Template ini disusun berdasarkan sumber-sumber berikut.

## Agentic workflow
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic): mulai dari workflow paling sederhana; pola *evaluator-optimizer* (coder → reviewer).
- [Claude Code: Best practices for agentic coding](https://www.anthropic.com/engineering/claude-code-best-practices) (Anthropic): alur rencana → kode → verifikasi, `CLAUDE.md` untuk tim.
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) (Anthropic): instruksi ringkas dan subagent dengan konteks terpisah.
- [Claude Code docs: Subagents](https://code.claude.com/docs/en/sub-agents) dan [Skills](https://code.claude.com/docs/en/skills): format `.claude/agents/` dan `.claude/skills/`.
- [AGENTS.md](https://agents.md): standar terbuka instruksi agen yang bisa dipakai lintas tool.
- [GitHub Spec Kit](https://github.com/github/spec-kit): sepakati spesifikasi dulu (→ piagam riset), baru koding.

## Struktur proyek ML
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) (DrivenData): struktur folder dan aturan data mentah tidak diubah.
- [ML Code Completeness Checklist](https://github.com/paperswithcode/releasing-research-code) (Papers with Code, dipakai NeurIPS): kode training/evaluasi, hasil, dan perintah reproduksi.

## Validitas riset
- [REFORMS checklist](https://reforms.cs.princeton.edu/) (Kapoor, Narayanan, dkk., *Science Advances* 2024): checklist validasi Fase 3.
- [Leakage and the Reproducibility Crisis in ML-based Science](https://reproducible.cs.princeton.edu/) (Kapoor & Narayanan, *Patterns* 2023): jenis-jenis kebocoran data.
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (Gebru dkk.): apa yang perlu dicatat tentang dataset.
- [PyTorch: Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html): pengaturan seed.
