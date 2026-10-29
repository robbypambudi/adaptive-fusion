# Spesifikasi: GMM-Based Fusion Router

> Dokumen acuan untuk tugas GMM router (penanggung jawab: Robby). Baca `docs/konteks-notebook.md`
> dulu. Semua agen yang mengerjakan tugas ini mengikuti dokumen ini. Jika ada yang tidak tercakup,
> **tanya**, jangan berasumsi.

## 1. Tujuan

Menjawab pertanyaan: *apakah memilih metode fusion **per sampel** (Dynamic Fusion) memberi akurasi
test lebih tinggi daripada memakai **satu** fusion untuk semua sampel (Global Fusion)?*

TODO asli dari tim:
- [ ] **Prepare Data:** confidence tiap fusion (concat, mean, cross_attn) sebagai label router.
- [ ] **GMM:** probabilitas tiap cluster (K) per sampel.
- [ ] **MLP Router:** input = probabilitas GMM, target = label fusion.
- [ ] **Dynamic Fusion:** prediksi metode fusion terbaik per sampel dengan router.
- [ ] **Testing:** akurasi Dynamic Fusion di test set.
- [ ] **Global Fusion:** evaluasi concat, mean, dan cross_attn masing-masing di seluruh test set.
- [ ] **Comparison:** Dynamic Fusion vs Global Fusion.
- Catatan: hindari data leakage dengan memisahkan train, validation, dan test.

## 2. Keputusan yang sudah disepakati (2026-10-29)

| # | Keputusan | Alasan |
|---|---|---|
| D1 | Label router berasal dari prediksi **out-of-fold (OOF) 5-fold di train**. Model fusion dilatih ulang per fold, hanya memakai data train. | Model notebook hafal train (confidence ≈ 1.0). CSV per-sampel notebook berasal dari test, jadi memakainya = kebocoran. |
| D2 | GMM di-fit pada **embedding image+title+text (2304-d) → StandardScaler → PCA → GMM**. Ketiganya di-fit di train saja. | Tidak memakai label sehingga tersedia saat test. Scaler wajib karena norma image ≈ 46 vs text = 1. |
| D3 | Label router = **argmax P(kelas benar)** di antara 3 metode (= `best_loss_method` notebook). Seri diputus dengan urutan `mean → concat → cross_attn`. | Setiap sampel dapat label; sama dengan definisi notebook. |
| D4 | Kode di-port ke repo: `src/adaptive_fusion/`, hyperparameter di `configs/`, angka di `results/`. Notebook tetap jadi referensi Fase 1. | Mengikuti `AGENTS.md`. |

## 3. Keputusan terbuka (tanya tim sebelum tugas terkait dikerjakan)

| # | Pertanyaan | Usulan default | Tugas |
|---|---|---|---|
| **Q0** | **Run mana yang jadi data utama?** Run zip (1000 baris → 97 test) atau run besar (~49.700 valid → 9.944 test, lihat `docs/konteks-notebook.md`)? Run besar butuh embedding `.npy`, `appliances.csv`, nilai `MAX_ROWS`, dan cara sampling dari teman tim. | Run besar sebagai data utama (bisa mengukur selisih ~132 sampel). Run zip dipakai untuk smoke test dan pengembangan selama artefak run besar belum ada. | T1 (blokir) |
| Q1 | Nama dan implementasi baseline: ganti `concat`/`mean` di `baselines.py` dengan versi notebook, atau daftarkan versi notebook dengan nama baru? | Ganti isi `concat`/`mean` dengan versi notebook (nama tetap, tes registry tetap lolos), tambah `cross_attn`. | T2 |
| Q2 | Split data: tetap seed 42 (identik notebook) untuk semua seed, atau ikut berubah per seed? | Split **tetap** seed 42; seed hanya mengubah inisialisasi model, fold OOF, GMM, router. | T1 |
| Q3 | `PATIENCE=300` (praktis tanpa early stopping, checkpoint terbaik by val loss): pertahankan? | Pertahankan agar setara notebook. Catat sebagai keterbatasan. | T2 |
| Q4 | Kriteria pemilihan K/PCA/router di val: akurasi Dynamic Fusion di val, atau akurasi router di val? | Akurasi Dynamic Fusion di val; jika seri, pilih K terkecil. | T6 |
| Q5 | Jumlah seed final. | 3 seed (42, 43, 44), sesuai charter. | Fase 4 |
| Q6 | Untuk run besar: apakah split juga stratified 64/16/20 `random_state=42` seperti run zip? OOF 5-fold (15 model fusion per seed) dan GMM `full` pada ~31.800 sampel train perlu dicek waktunya. | Asumsikan split sama sampai teman tim mengonfirmasi. | T1 |

Router **bukan** `FusionModule`. Router memilih model terlatih per sampel saat inferensi, tidak
menggabungkan fitur, jadi tidak masuk `FUSION_REGISTRY` dan tidak diuji oleh
`tests/test_fusion_contract.py`. Tempatnya di `src/adaptive_fusion/router/`. Ketiga fusion (mean,
concat, cross_attn) tetap `FusionModule` dan wajib lolos tes kontrak.

## 4. Data dan split

- Sumber: artefak notebook diekstrak **apa adanya** ke `data/raw/amazon_review_2023/` (run zip) atau
  `data/raw/amazon_review_2023_large/` (run besar, setelah artefaknya didapat). Jangan diubah. Embedding
  `.npy` dianggap data sumber yang dibekukan (tidak di-encode ulang). Run yang dipakai dipilih lewat
  config data, bukan di kode (Q0).
- Angka di bawah ini untuk run zip. Untuk run besar, ukuran dan distribusi kelas dicek ulang di T1.
- Sampel valid: baris dengan embedding image, title, dan text semuanya finite → 484 sampel.
- Label: `rating >= 4` → 1, selain itu 0.
- Split: reproduksi **persis** urutan pemanggilan notebook:
  ```python
  idx = np.arange(len(y))
  idx_train, idx_test, y_train, y_test = train_test_split(idx, y, test_size=0.2, random_state=42, stratify=y)
  idx_train, idx_val, y_train, y_val = train_test_split(idx_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
  ```
  Ukuran yang diharapkan: 309/78/97, kelas train [64, 245], val [16, 62], test [20, 77].
- **Dua ruang indeks** (jangan dicampur):
  - *Indeks terfilter* `0..n_valid-1` (posisi di array 484 baris setelah mask valid). Ini dipakai untuk
    akses tensor dan split.
  - *Indeks sumber* = nomor baris di `appliances.csv` (`source_row = np.where(valid)[0][indeks_terfilter]`).
    Kolom `subset_idx` di CSV notebook memakai ruang ini.
  - Cek kecocokan dengan notebook: `np.where(valid)[0][idx_test]` (urut `test_pos`) == `subset_idx` di CSV.
    Ini cek **identitas sampel**, bukan memakai isi/label CSV tersebut.
- **Identitas dataset.** Setiap run data punya `dataset_id` dari config (mis. `appliances_zip1000`,
  `appliances_large`). Semua artefak turunan diberi namespace `dataset_id`:
  `data/processed/<dataset_id>/split.json`, berisi indeks terfilter train/val/test, pemetaan
  `valid → source_row`, dan *fingerprint* sumber (SHA256 file `.npy` dan `appliances.csv`, n_valid, seed split).

## 4a. Isolasi artefak

- Path: `data/processed/<dataset_id>/...`, `results/<dataset_id>/...`, `checkpoints/<dataset_id>/...`
  (checkpoint tidak di-commit).
- **Run immutable.** Setiap tahap menulis ke direktori `<run_id>`, dengan
  `run_id = sha256(dataset fingerprint, split_sha256, config_sha256 tahap itu, run_id upstream, seed)[:12]`.
  Mengubah config atau upstream menghasilkan `run_id` baru, sehingga tidak ada artefak lama yang ditimpa
  diam-diam. Tahap menolak menulis jika `<run_id>` sudah ada dengan isi berbeda.
- Setiap file hasil/confidence/config terpilih menyimpan *manifest*: `dataset_id`, fingerprint sumber,
  `split_sha256`, `config_sha256` (config tahap itu + config training upstream), seed, `run_id`, dan
  **SHA256 setiap checkpoint/artefak upstream** yang dibaca.
- Setiap konsumen (T4–T7, skrip rekap) **wajib memverifikasi ulang** hash file upstream terhadap manifest
  dan menolak artefak yang tidak cocok: dataset, fingerprint, split, config, atau hash checkpoint.
  Khususnya, T7 harus memakai checkpoint Full yang **hash-nya sama** dengan yang dipakai T4 (confidence val)
  dan T6 (seleksi).
- Skrip rekap menolak gabungan hasil dari `dataset_id` atau protokol berbeda (config training, grid seleksi,
  R permutasi). Pilihan router per seed boleh berbeda karena itu hasil seleksi.

## 5. Pipeline (per seed)

```
                 ┌──────────────── TRAIN (309) ────────────────┐   VAL (78)          TEST (97)
Tahap A  OOF     5-fold stratified pada train (outer). Per fold f:
                   bagian latih fold = 4/5 train → dibagi lagi stratified 80/20 menjadi
                   inner-train dan inner-val (random_state=seed).
                   Latih 3 fusion di inner-train, checkpoint by loss di inner-val.
                   Prediksi P(kelas benar) di fold f (outer held-out).
                 → conf_oof [309, 3] → label router y_r = argmax (D3)
Tahap B  Full    Latih 3 fusion di seluruh train, checkpoint by loss di VAL global.
                 → prediksi di val (label router val). Model dibekukan; TEST belum disentuh.
Tahap C  GMM     Scaler → PCA → GMM fit di train (embedding saja, tanpa label)
                 → posterior P(k | x) untuk train / val (test baru di Tahap E)
Tahap D  Router  MLP: posterior [K] → 3 logit, dilatih di (posterior_train, y_r_train)
                 Pilih K, dim PCA, covariance, hparam MLP di VAL (Q4) → konfigurasi dibekukan
Tahap E  Test    SEKALI, setelah A–D beku: prediksi semua model Full di test, posterior GMM test,
                 pilihan router m_i, Dynamic = model Full m_i pada sampel i, Global, baseline
                 router, oracle, dan label router test (untuk akurasi router), dihitung bersamaan
```

**Isolasi seleksi checkpoint (Tahap A).** Untuk fold f, data yang boleh dipakai untuk optimasi
**dan** pemilihan checkpoint hanya bagian latih fold f (inner-train + inner-val). Fold f yang
di-hold-out, val global, dan test **tidak boleh** dipakai. Setiap model fold menyimpan provenance:
`optimizer_indices`, `selection_indices`, `predict_indices`. Ketiganya disjoint, dan `predict_indices`
tidak beririsan dengan dua lainnya maupun dengan val/test.

Detail:
- **Confidence** = `softmax(logits)[kelas benar]` dari model fusion dalam `eval()` + `no_grad()`.
  Label kelas benar hanya dipakai untuk **membuat label router**, tidak pernah jadi input router.
- **Label router val** berasal dari model Full (Tahap B). Val juga dipakai untuk memilih checkpoint
  model Full, jadi confidence val sedikit optimis. Ini dicatat sebagai keterbatasan, bukan kebocoran test.
- **GMM:** `sklearn.mixture.GaussianMixture`, `random_state=seed`. Grid usulan (di config):
  K ∈ {2, 3, 4, 5, 6, 8, 10}, PCA ∈ {16, 32, 64}, `covariance_type` ∈ {diag, full}
  (`full` hanya untuk PCA ≤ 32; 309 sampel terlalu sedikit untuk kovarians penuh berdimensi tinggi).
- **MLP router:** `K → hidden → 3`, ReLU/GELU, dropout. CrossEntropy, opsional dengan bobot kelas
  (label sangat timpang: `mean` ≈ 73% di run zip, `concat` ≈ 75% di run besar). Early stopping by val loss. Dilatih dengan `seed_everything(seed)`.
- **Dynamic Fusion:** keluaran utama `y_pred_dyn[i] = pred_{m_i}[i]`. Probabilitas untuk ROC-AUC =
  probabilitas kelas positif dari model terpilih.

## 6. Pembanding wajib di tabel hasil

| Baris | Definisi |
|---|---|
| Global `mean` / `concat` / `cross_attn` | Model Full, satu metode untuk semua sampel test |
| **Dynamic (GMM+MLP)** | Metode usulan |
| Router mayoritas | Selalu memilih label router terbanyak di train. Jika Dynamic ≈ baris ini, router tidak belajar apa-apa |
| Router acak | Pilih metode acak per sampel (seed tetap) |
| Oracle | Per sampel pilih metode yang benar jika ada (batas atas; memakai label test, **hanya** sebagai referensi) |

Metrik: accuracy (utama), F1 kelas positif, precision, recall, ROC-AUC. Untuk router: akurasi router
terhadap label router test, distribusi metode yang dipilih, dan jumlah sampel yang diubah keputusannya
vs fusion global terbaik. Untuk perbandingan: mean ± std antar seed, ditambah uji McNemar berpasangan
(Dynamic vs Global terbaik per seed). Global terbaik ditentukan dengan akurasi **val**, bukan test.

## 7. Checklist kebocoran (wajib dicek reviewer)

- [ ] Scaler, PCA, dan GMM di-fit hanya di embedding **train**.
- [ ] Label router train berasal dari OOF: model fold tidak pernah melihat sampel yang ia beri label,
      baik untuk optimasi maupun pemilihan checkpoint (provenance indeks dicek, §5).
- [ ] Tidak ada metrik test yang dihitung sebelum T7. Tidak ada keputusan (perbaikan kode, hparam,
      checkpoint) yang diambil berdasarkan angka test.
- [ ] Semua artefak yang dikonsumsi punya `dataset_id`, fingerprint, `split_sha256`, config, dan hash
      checkpoint upstream yang cocok (§4a). Model Full di T7 = model yang sama dengan di T4/T6 (hash).
- [ ] Kontrol permutasi mengulang seluruh prosedur seleksi per permutasi (§8).
- [ ] `appliances_fusion_persample*.csv` (keduanya dihitung di test) **tidak** dipakai untuk training, seleksi, atau tuning.
- [ ] Input router tidak mengandung label atau apa pun yang diturunkan dari label.
- [ ] K, dim PCA, covariance, dan hparam MLP dipilih di **val**. Test dievaluasi sekali setelah semuanya dibekukan.
- [ ] Global terbaik dipilih berdasarkan val. Oracle hanya dilaporkan sebagai batas atas.
- [ ] Model Global dan model yang dipakai Dynamic adalah **model Full yang sama** (perbandingan setara).
- [ ] Semua run memakai `seed_everything(seed)`; hparam ada di `configs/`.

## 8. Rencana tugas (satu tugas = satu PR, jalankan dengan `/gmm-router T<n>`)

| Tugas | Isi | Output | Tes minimal |
|---|---|---|---|
| **T1** Data | Tambah dependensi (`scikit-learn`, `pandas`) di `pyproject.toml`. Loader `src/adaptive_fusion/data/amazon.py`: baca `.npy` + CSV dari `data/raw/<dataset>/`, mask valid, label, split (§4) → `data/processed/<dataset_id>/split.json` (indeks terfilter, pemetaan `source_row`, fingerprint). Config `configs/data/<dataset_id>.yaml`. | dict `{"image","title","text"}` → tensor, plus label | 484 sampel, 309/78/97, distribusi kelas cocok, tanpa overlap, tanpa NaN; `source_row[idx_test]` urut `test_pos` == `subset_idx` CSV notebook; loader menolak fingerprint yang tidak cocok |
| **T2** Fusion + training | Port 3 fusion notebook sebagai `FusionModule` (projection cabang di dalam modul; `cross_attn` punya kwarg `query_source`, default sumber pertama hasil urut = `image`). Model `encoder→fusion→head`, `train.py` (checkpoint by loss pada indeks seleksi yang diberikan pemanggil, simpan provenance indeks). Jawab Q1, Q3. | `src/adaptive_fusion/fusion/*.py`, `models.py`, `train.py` | Tes kontrak fusion lolos; smoke test training di data kecil; **parity** dengan notebook diuji pada fixture sintetis (bobot sama → output sama), bukan dengan angka test |
| **T3** Model Full | Latih model Full 3 metode (checkpoint by val global), simpan checkpoint + metrik **val** saja. Sanity: best val loss seed 42 sebanding dengan log notebook (mean 0.0634, concat 0.0604, cross_attn 0.0409). Ini cek kewajaran, bukan syarat lulus yang memicu tuning. **Tidak ada evaluasi test.** | `checkpoints/<dataset_id>/<run_id>/full/<metode>_seed<s>.pt`, `results/<dataset_id>/<run_id>/full_val/<metode>_seed<s>.json` (+ manifest, §4a) | Tes: fungsi T3 tidak menerima/memuat indeks test |
| **T4** OOF + label router | 5-fold OOF dengan inner-val per fold (§5) → `conf_oof`, `y_router` train; confidence + label router **val** dari model Full. | `data/processed/<dataset_id>/<run_id>/router/seed<s>/confidences.npz` (+ provenance, manifest dengan hash checkpoint Full) | Konsumen menolak cache jika hash checkpoint Full berubah; setiap indeks train muncul tepat sekali di OOF; per fold, `optimizer_indices`, `selection_indices`, `predict_indices` disjoint dan tidak beririsan dengan val/test |
| **T5** Fitur GMM | `src/adaptive_fusion/router/gmm.py`: Scaler → PCA → GMM, fit di train, `transform` → posterior | posterior `[n, K]` | Baris posterior berjumlah 1; fit hanya memakai indeks train (tes dengan data sintetis) |
| **T6** MLP router + seleksi | `router/mlp.py` + grid di `configs/router/gmm_mlp.yaml`; pilih di val (Q4); bekukan konfigurasi terpilih | `results/<dataset_id>/<run_id>/router_selection/seed<s>.json` (+ manifest, distribusi permutasi) | Router bisa overfit batch kecil; **kontrol permutasi** (lihat bawah); K=1 → router konstan sehingga Dynamic = satu metode Global persis |
| **T7** Test + perbandingan | Satu-satunya tahap yang menyentuh test: Tahap E §5, tabel §6, skrip rekap mean ± std + McNemar | `results/<dataset_id>/<run_id>/gmm_router/seed<s>.json`, `results/<dataset_id>/gmm_router_summary.json` | T7 memverifikasi hash checkpoint = yang dipakai T4/T6; rekap menolak artefak dengan identitas/protokol tidak cocok; reviewer mencocokkan angka laporan dengan `results/` |

**Kontrol permutasi (T6, di val).** Untuk setiap permutasi r = 1..R (usulan R = 20, seed berbeda):
acak label router train, lalu **ulangi seluruh prosedur seleksi** yang sama dengan router asli (grid
K/PCA/covariance/hparam MLP penuh, kriteria Q4 di val). Catat skor val terbaik hasil seleksi itu.
Fitur GMM tidak bergantung label, jadi posterior per konfigurasi grid di-cache dan dipakai ulang; hanya
MLP yang dilatih ulang per permutasi. Hasilnya adalah distribusi nol skor val **setelah seleksi**, yang
setara dengan skor router asli (bias seleksi ikut terhitung). Router asli dinyatakan mengungguli kontrol
jika skornya di atas persentil ke-95 distribusi ini. Laporkan juga p-value empiris `(1 + #{null ≥ asli}) / (R + 1)`.
Dengan R = 20, p minimum ≈ 0.048, jadi ini bukti lemah dan dilaporkan apa adanya.
Ekspektasi akurasi router acak-independen = Σⱼ pⱼ·qⱼ (pⱼ = frekuensi label val, qⱼ = frekuensi pilihan
router), **bukan** akurasi mayoritas. Ini diagnostik yang dilaporkan, bukan syarat tes otomatis.
Router mayoritas tetap baseline terpisah.

Format `results/<dataset_id>/gmm_router/seed<s>.json` (usulan):
```json
{
  "dataset_id": "appliances_zip1000",
  "source_fingerprint": "<sha256>",
  "split_sha256": "<sha256>",
  "config_sha256": "<sha256>",
  "run_id": "<12 hex>",
  "upstream": {"t3_run_id": "...", "t4_run_id": "...", "t6_run_id": "..."},
  "checkpoints": {"mean": {"path": "checkpoints/<dataset_id>/<t3_run_id>/full/mean_seed42.pt", "sha256": "<sha256>"}, "concat": {}, "cross_attn": {}},
  "seed": 42,
  "selected": {"K": 4, "pca_dim": 32, "covariance_type": "diag", "router": {"hidden": 32, "lr": 0.001}},
  "val": {"dynamic_acc": 0.0, "global_acc": {"mean": 0.0, "concat": 0.0, "cross_attn": 0.0}},
  "test": {
    "global": {"mean": {"accuracy": 0.0, "f1": 0.0, "roc_auc": 0.0}, "concat": {}, "cross_attn": {}},
    "dynamic": {"accuracy": 0.0, "f1": 0.0, "roc_auc": 0.0},
    "majority_router": {"accuracy": 0.0}, "random_router": {"accuracy": 0.0}, "oracle": {"accuracy": 0.0},
    "router_acc": 0.0, "routing_counts": {"mean": 0, "concat": 0, "cross_attn": 0},
    "mcnemar_vs_best_global": {"best_global_by_val": "cross_attn", "b": 0, "c": 0, "p_value": 1.0}
  }
}
```

## 9. Pemetaan ke fase

- **Fase 0:** charter diisi dari notebook dan spec ini (`docs/research-charter.md`).
- **Fase 1:** notebook sudah selesai; T1 memindahkannya ke repo.
- **Fase 2:** T2–T7.
- **Fase 3:** `/fase-3-validasi` dengan checklist §7, ditambah sanity check router: kontrol permutasi
  (§8) dan K=1 → Dynamic = satu metode Global persis.
- **Fase 4:** EXP-001 (Global) dan EXP-002 (Dynamic) di `docs/eksperimen/`, 3 seed.

## 10. Yang di luar lingkup tugas ini

Mengganti encoder, menambah data, memperbaiki cross_attn (patch token), dan mengubah definisi label
sentimen. Semua ini dicatat sebagai saran, tidak dikerjakan tanpa persetujuan tim.
