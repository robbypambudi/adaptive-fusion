.PHONY: setup test lint

setup:  ## Install paket + dependensi dev
	python -m pip install -e ".[dev]"

test:   ## Jalankan semua tes
	python -m pytest -q

lint:   ## Cek gaya kode
	python -m ruff check .
