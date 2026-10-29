# Ide Riset

> Tulis ide apa saja di sini, bebas dan tidak perlu rapi. Boleh poin-poin, link paper, atau pertanyaan.
> Saat menjalankan `/fase-0-setup`, AI membaca file ini lalu membantu menyusunnya menjadi `docs/research-charter.md`.

## Ide utama
- Sample-wise conditional fusion: tiap sampel mungkin cocok dengan metode fusion yang berbeda.
- Notebook Fase 1 menunjukkan pemenang per sampel bergeser: review positif didominasi `mean`,
  review negatif didominasi `cross_attn`.
- GMM-based router: probabilitas cluster GMM per sampel → MLP → pilih fusion (mean/concat/cross_attn).
  Bandingkan dengan satu fusion global.

## Data yang mungkin dipakai
- Amazon Reviews 2023, kategori Appliances (gambar + judul + teks review). Lihat `docs/konteks-notebook.md`.

## Paper / referensi yang menarik
-

## Pertanyaan yang belum terjawab
- Run zip: 97 sampel test, selisih oracle vs global terbaik hanya 1 sampel.
  Run besar: 9.944 test, selisih ~132 sampel. Artefak run besar perlu diminta ke teman tim.
- Apakah cross-attention dengan patch token DINOv2 mengubah pola pemenang?

## Catatan lain
- Keputusan dan pertanyaan terbuka untuk router ada di `docs/spec-gmm-router.md`.
