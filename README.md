# adaptive-fusion

Template riset **metode fusion**. Kodenya dibangun dengan bantuan AI (Claude Code) dan dikerjakan bersama tim dalam 5 fase.

---

## Cara Pakai (Langkah demi Langkah)

### 1. Siapkan komputer (sekali saja)

Yang dibutuhkan:
- Python 3.11 atau lebih baru
- Git
- [Claude Code](https://code.claude.com/docs) (atau agen AI lain seperti Codex/Cursor)

Lalu jalankan di terminal:

```bash
git clone <url-repo-ini>
cd adaptive-fusion
make setup     # install semua yang dibutuhkan
make test      # cek apakah semua berjalan normal (harus "passed")
```

> Tidak punya GPU? Install PyTorch versi CPU dulu sebelum `make setup`:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 2. Buka Claude Code

```bash
claude
```

Setelah terbuka, AI otomatis membaca aturan proyek (`AGENTS.md`). Kamu tinggal mengetik perintah fase.

### 3. Kerjakan fase satu per satu

Ketik perintah di bawah ini di dalam Claude Code, **berurutan**:

| Fase | Ketik ini | Apa yang terjadi |
|---|---|---|
| **0. Setup** | `/fase-0-setup` | AI membantu tim mengisi **piagam riset**: pertanyaan, dataset, metrik, baseline. AI hanya memberi usulan; tim yang memutuskan. |
| **1. Data** | `/fase-1-data` | AI menyiapkan dataset: unduh, preprocessing, split train/val/test, dan tes data. |
| **2. Ngoding** | `/fase-2-koding implementasi gated fusion` | AI menulis kode untuk **satu tugas**, lalu AI lain (`reviewer`) memeriksanya. Ulangi untuk tugas berikutnya. |
| **3. Validasi** | `/fase-3-validasi` | AI memeriksa seluruh kode: mencari bug, kebocoran data, dan menjalankan sanity check. |
| **4. Eksekusi** | `/fase-4-eksekusi EXP-001` | AI menjalankan eksperimen, merekap hasil, dan mengecek bahwa angka di laporan cocok dengan hasil asli. |

**Kapan pindah ke fase berikutnya?** Buka checklist fase itu di [`docs/phases/`](docs/phases/). Jika semua sudah dicentang dan tim setuju, lanjut ke fase berikutnya.

### 4. Simpan pekerjaan dan minta review tim

Setiap selesai satu tugas:

```bash
git checkout -b fase-2/gated-fusion     # buat branch baru (format: fase-N/nama-tugas)
git add .
git commit -m "Tambah gated fusion"
git push -u origin fase-2/gated-fusion
```

Lalu buat **Pull Request** di GitHub. Tulis di deskripsinya bagian mana yang dibuat AI, supaya anggota tim lain tahu apa yang perlu diperiksa.

---

## Tips Supaya Hasil AI Bagus

1. **Isi piagam riset dengan lengkap dulu.** AI bekerja jauh lebih baik jika tahu tujuan risetnya.
2. **Beri tugas kecil.** Contoh yang baik: `/fase-2-koding buat encoder untuk data citra`. Contoh yang buruk: "buatkan semua kodenya".
3. **Selalu jalankan `make test`** setelah AI mengubah kode.
4. **Jangan langsung percaya.** Baca ringkasan dari `reviewer` dan cek sendiri bagian yang penting.
5. **Jika AI terus mengulang kesalahan yang sama**, tambahkan aturan baru di `AGENTS.md`.

---

## Perintah yang Sering Dipakai

| Perintah | Fungsi |
|---|---|
| `make setup` | Install semua kebutuhan |
| `make test` | Jalankan semua tes |
| `make lint` | Cek kerapian kode |

---

## Isi Folder

```
AGENTS.md, CLAUDE.md       aturan untuk AI (jangan dihapus)
.claude/agents/            2 asisten AI: coder (menulis kode) & reviewer (memeriksa)
.claude/skills/            perintah /fase-0 sampai /fase-4
docs/research-charter.md   piagam riset (diisi di Fase 0)
docs/phases/               langkah & checklist tiap fase
docs/eksperimen/           catatan tiap eksperimen (Fase 4)
docs/SOURCES.md            sumber rujukan template ini
configs/                   pengaturan data & eksperimen
data/raw/                  data asli (jangan diubah, tidak masuk git)
data/processed/            data hasil olahan (tidak masuk git)
src/adaptive_fusion/       kode program
tests/                     tes otomatis
results/                   hasil eksperimen
```

---

## Pertanyaan Umum

**Saya tidak pakai Claude Code. Apakah bisa?**
Bisa. Minta agen AI kamu membaca `AGENTS.md` dan file fase di `docs/phases/phase-N.md`, lalu ikuti langkahnya.

**Bagaimana cara anggota tim lain mendapatkan datasetnya?**
Data tidak disimpan di git karena ukurannya besar. Cara mendapatkannya (link drive/bucket) dicatat di Fase 1.

**Bagaimana cara menambah metode fusion baru?**
Minta AI dengan `/fase-2-koding implementasi <nama metode>`. Metode baru otomatis dites oleh `tests/test_fusion_contract.py`.

**Tes gagal setelah AI mengubah kode, bagaimana?**
Minta AI memperbaiki kodenya, **bukan tesnya**. Tes yang dilemahkan supaya lolos membuat hasil riset tidak bisa dipercaya.
