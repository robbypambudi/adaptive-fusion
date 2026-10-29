"""Fase 1 (dataset penuh): semua review Appliances yang punya gambar → CSV + embedding.

Mengikuti notebook ``adaptive_fusion new.ipynb`` Bagian 2–5, tetapi tanpa subsample:
- baris = SEMUA review dengan ``images`` tidak kosong (urutan file), label dari ``rating``;
- gambar = gambar pertama (``small_image_url``) → DINOv2-base CLS (768-d); gagal unduh → NaN;
- teks = ``title`` dan ``text`` di-encode terpisah dengan BGE-base-en-v1.5 (dinormalisasi).

Keluaran (tulis-sekali; skrip menolak menimpa file yang sudah ada) di ``--out``:
``appliances.csv``, ``appliances_{image,title,text}_emb.npy``, ``images/`` (cache unduhan),
``download_log.json``. Unduhan bisa dilanjutkan (gambar yang sudah ada dilewati).

    python scripts/build_full_features.py --zip datasets/amazon_review_2023-*.zip \
        --out data/raw/amazon_review_2023_full
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image

IMAGE_MODEL = "facebook/dinov2-base"
TEXT_MODEL = "BAAI/bge-base-en-v1.5"
COLUMNS = ["rating", "title", "text", "parent_asin", "user_id", "images", "review_time"]


def read_image_reviews(zip_path: Path) -> pd.DataFrame:
    """Semua review dengan minimal satu gambar, urutan sesuai file."""
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith("/Appliances.jsonl"))
        with zf.open(name) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                r = json.loads(line)
                if not r.get("images"):
                    continue
                rows.append(
                    {
                        "rating": float(r["rating"]),
                        "title": r.get("title") or "",
                        "text": r.get("text") or "",
                        "parent_asin": r.get("parent_asin"),
                        "user_id": r.get("user_id"),
                        "images": json.dumps(r["images"]),
                        "review_time": pd.to_datetime(r["timestamp"], unit="ms"),
                        "first_image": r["images"][0].get("small_image_url"),
                    }
                )
    return pd.DataFrame(rows)


def download_images(urls: list[str | None], image_dir: Path, workers: int) -> dict[str, int]:
    """Unduh paralel ke ``images/<baris>.jpg``; file yang sudah ada dilewati (bisa dilanjutkan)."""
    image_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=workers))

    def fetch(i: int, url: str | None) -> tuple[int, str]:
        target = image_dir / f"{i}.jpg"
        if target.exists():
            return i, "cached"
        if not url:
            return i, "no_url"
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=15)
                if resp.status_code == 200 and resp.content:
                    tmp = target.with_suffix(".part")
                    tmp.write_bytes(resp.content)
                    tmp.rename(target)
                    return i, "ok"
                if resp.status_code in (403, 404, 410):
                    return i, f"http_{resp.status_code}"
            except requests.RequestException:
                pass
            time.sleep(1.5 * (attempt + 1))
        return i, "failed"

    status: dict[int, str] = {}
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, i, u) for i, u in enumerate(urls)]
        for k, fut in enumerate(as_completed(futures), 1):
            i, s = fut.result()
            status[i] = s
            if k % 5000 == 0:
                print(f"  unduh {k}/{len(urls)} ({time.time() - start:.0f}s)", flush=True)
    counts: dict[str, int] = {}
    for s in status.values():
        counts[s] = counts.get(s, 0) + 1
    return counts


@torch.no_grad()
def encode_images(n: int, image_dir: Path, device: torch.device, batch_size: int) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL)
    model = AutoModel.from_pretrained(IMAGE_MODEL).to(device).eval()
    out = np.full((n, 768), np.nan, dtype=np.float32)
    batch_idx: list[int] = []
    batch_img: list[Image.Image] = []

    def flush() -> None:
        if not batch_idx:
            return
        inputs = processor(images=batch_img, return_tensors="pt").to(device)
        cls = model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy()
        out[batch_idx] = cls
        batch_idx.clear()
        batch_img.clear()

    bad = 0
    for i in range(n):
        path = image_dir / f"{i}.jpg"
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:  # gambar rusak → tetap NaN, seperti notebook
            bad += 1
            continue
        batch_idx.append(i)
        batch_img.append(img)
        if len(batch_idx) == batch_size:
            flush()
        if (i + 1) % 10000 == 0:
            print(f"  encode gambar {i + 1}/{n}", flush=True)
    flush()
    print(f"  gambar rusak/tidak terbaca: {bad}")
    return out


def encode_texts(texts: list[str], device: torch.device) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(TEXT_MODEL, device=str(device))
    texts = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    return model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)


def write_atomic(path: Path, writer) -> None:
    """Tulis ke file sementara lalu rename: file keluaran selalu utuh atau tidak ada."""
    tmp = path.with_name(f".{path.name}.tmp")
    writer(tmp)
    tmp.rename(path)


def save_npy(path: Path, arr: np.ndarray) -> None:
    with open(path, "wb") as f:  # np.save ke handle: nama file sementara tidak diberi akhiran .npy
        np.save(f, arr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/raw/amazon_review_2023_full"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--image-batch", type=int, default=128)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setiap tahap dilanjutkan dari keluaran yang sudah selesai (REV-007); keluaran yang sudah
    # ada dipakai ulang dan tidak pernah ditimpa.
    csv_path, url_path = out / "appliances.csv", out / "first_image_urls.json"
    if csv_path.exists() and url_path.exists():
        subset = pd.read_csv(csv_path, index_col=0)
        print(f"CSV sudah ada: {len(subset)} baris")
    else:
        subset = read_image_reviews(args.zip)
        write_atomic(url_path, lambda p: subset["first_image"].to_json(p, orient="values"))
        write_atomic(csv_path, lambda p: subset[COLUMNS].to_csv(p))
        print(f"{len(subset)} review dengan gambar -> {csv_path}")
    urls = json.loads(url_path.read_text())
    if len(urls) != len(subset):
        raise ValueError("first_image_urls.json tidak cocok dengan appliances.csv")

    log_path = out / "download_log.json"
    if log_path.exists():
        print(f"unduhan sudah selesai: {json.loads(log_path.read_text())}")
    else:
        print(f"unduh {len(urls)} gambar ({args.workers} thread)")
        counts = download_images(urls, out / "images", args.workers)
        print(f"  status unduhan: {counts}")
        write_atomic(log_path, lambda p: p.write_text(json.dumps(counts, indent=1)))

    image_path = out / "appliances_image_emb.npy"
    if image_path.exists():
        image_emb = np.load(image_path)
        print(f"embedding gambar sudah ada: {image_emb.shape}")
    else:
        print(f"encode gambar ({IMAGE_MODEL}, {device})")
        image_emb = encode_images(len(urls), out / "images", device, args.image_batch)
        write_atomic(image_path, lambda p: save_npy(p, image_emb))
    for col in ("title", "text"):
        path = out / f"appliances_{col}_emb.npy"
        if path.exists():
            print(f"embedding {col} sudah ada")
            continue
        print(f"encode {col} ({TEXT_MODEL})")
        emb = encode_texts(subset[col].fillna("").astype(str).tolist(), device)
        write_atomic(path, lambda p, e=emb: save_npy(p, e))

    valid = np.isfinite(image_emb).all(axis=1)
    labels = (subset["rating"].to_numpy() >= 4).astype(int)
    print(
        f"selesai: {valid.sum()} / {len(valid)} sampel valid; "
        f"kelas valid (neg/pos) = {np.bincount(labels[valid], minlength=2).tolist()}"
    )


if __name__ == "__main__":
    main()
