.PHONY: setup test lint data router-dev router-test data-full router-dev-full router-test-full

DATA_CONFIG ?= configs/data/appliances_zip1000.yaml
EXP_CONFIG ?= configs/experiments/gmm_router.yaml
ZIP ?= $(firstword $(wildcard datasets/amazon_review_2023-*.zip))

setup:  ## Install paket + dependensi dev
	python -m pip install -e ".[dev]"

test:   ## Jalankan semua tes
	python -m pytest -q

lint:   ## Cek gaya kode
	python -m ruff check .

data:   ## Ekstrak artefak notebook dari zip ke data/raw + buat split
	python -m adaptive_fusion.data.amazon --config $(DATA_CONFIG) $(if $(ZIP),--extract-zip "$(ZIP)")

router-dev:   ## GMM router T3–T6 (train/val saja, semua seed)
	python -m adaptive_fusion.experiments.gmm_router dev --config $(EXP_CONFIG)

router-test:  ## GMM router T7 + rekap (menyentuh TEST; jalankan sekali setelah inspeksi lolos)
	python -m adaptive_fusion.experiments.gmm_router test --config $(EXP_CONFIG)
	python -m adaptive_fusion.experiments.gmm_router summary --config $(EXP_CONFIG)

data-full:   ## Dataset penuh: semua review bergambar → unduh gambar + encode (butuh GPU & internet)
	python scripts/build_full_features.py --zip "$(ZIP)" --out data/raw/amazon_review_2023_full
	python -m adaptive_fusion.data.amazon --config configs/data/appliances_full.yaml

router-dev-full:   ## GMM router T3–T6 pada dataset penuh (train/val saja)
	$(MAKE) router-dev EXP_CONFIG=configs/experiments/gmm_router_full.yaml

router-test-full:  ## T7 + rekap pada dataset penuh (menyentuh TEST; sekali, setelah inspeksi lolos)
	$(MAKE) router-test EXP_CONFIG=configs/experiments/gmm_router_full.yaml
