"""Laporan sementara EXP-002 (tahap dev, train/val saja) → PDF + gambar.

Semua angka dibaca dari file ``results/`` yang manifest-nya diverifikasi ulang (aturan 6 AGENTS.md).
Test set belum disentuh, jadi laporan ini hanya berisi hasil validasi.

    python scripts/laporan_sementara.py [--config configs/experiments/gmm_router.yaml]
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from adaptive_fusion.experiments.gmm_router import (  # noqa: E402
    _load_upstream,
    build_context,
    load_run,
)
from adaptive_fusion.utils.artifacts import manifest_path  # noqa: E402

# Palet kategorikal referensi (urutan tetap): mean, concat, cross_attn, lalu Dynamic.
COLORS = {"mean": "#2a78d6", "concat": "#eb6834", "cross_attn": "#1baf7a", "dynamic": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#6b6a64", "#e4e3de"
METHODS = ("mean", "concat", "cross_attn")


def load_results(ctx, seeds, select_runs=None):
    """Tanpa ``select_runs``: cari run dari identitas kode/config saat ini. Dengan ``select_runs``:
    muat run select eksplisit + run full upstream-nya dan verifikasi hash (REV-004)."""
    if select_runs:
        pairs = []
        for rid in select_runs:
            select = load_run(ctx, "select", rid)
            pairs.append((select, load_run(ctx, "full", select["identity"]["upstream"]["full"])))
    else:
        pairs = [(_load_upstream(ctx, "select", s), _load_upstream(ctx, "full", s)) for s in seeds]
    # REV-006: satu baris per seed, dan semua run harus berasal dari protokol yang sama
    # (config_sha256 tahap select & full mencakup config, kode, dan environment produsen).
    seeds_seen = [s["identity"]["seed"] for s, _ in pairs]
    if len(set(seeds_seen)) != len(seeds_seen):
        raise ValueError(f"seed ganda dalam laporan: {seeds_seen}")
    for stage, idx in (("select", 0), ("full", 1)):
        shas = {pair[idx]["identity"]["config_sha256"] for pair in pairs}
        if len(shas) != 1:
            raise ValueError(f"run '{stage}' berasal dari protokol berbeda: {sorted(shas)}")
    rows = []
    for select, full in pairs:
        seed = select["identity"]["seed"]
        sel = json.loads(manifest_path(select, "files", "selection", ctx.base).read_text())
        val_files = {
            m: json.loads(manifest_path(full, "files", f"val_{m}", ctx.base).read_text())
            for m in METHODS
        }
        rows.append({"seed": seed, "select": sel, "full": val_files, "manifest": select})
    return rows


def style_axes(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fit_xlim(ax, values, n_val):
    """Rentang sumbu x mengikuti data (bukan dipaku ke 1.0), dengan margin >= 2 sampel."""
    lo, hi = min(values), max(values)
    pad = max(2 / n_val, 0.15 * (hi - lo))
    ax.set_xlim(lo - pad, hi + pad)


def fig_val_accuracy(rows, n_val, path):
    """Dot plot: akurasi val per baris pembanding, satu titik per seed + garis rata-rata."""
    labels = ["Global mean", "Global concat", "Global cross_attn", "Router mayoritas", "Dynamic"]
    keys = ["mean", "concat", "cross_attn", "majority", "dynamic"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    for i, key in enumerate(keys):
        vals = []
        for r in rows:
            v = r["select"]["val"]
            vals.append(
                v["dynamic_acc"]
                if key == "dynamic"
                else v["majority_router"]["dynamic_acc"]
                if key == "majority"
                else v["global_acc"][key]
            )
        color = COLORS.get(key, MUTED)
        y = len(keys) - 1 - i
        offsets = [-0.12, 0.0, 0.12][: len(vals)]
        ax.scatter(
            vals,
            [y + o for o in offsets],
            s=46,
            color=color,
            zorder=3,
            edgecolor="white",
            linewidth=1.2,
        )
        mean = sum(vals) / len(vals)
        ax.plot([mean, mean], [y - 0.28, y + 0.28], color=INK, linewidth=2, zorder=4)
        ax.text(
            1.002,
            y,
            f"{mean:.4f}",
            va="center",
            ha="left",
            fontsize=8,
            color=INK,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=9)
    fit_xlim(ax, [x for c in ax.collections for x in c.get_offsets()[:, 0]], n_val)
    ax.set_xlabel(
        f"Akurasi validasi (n = {n_val:,}; 1 sampel = {1 / n_val:.5f})", fontsize=9, color=INK
    )
    ax.set_title(
        "Akurasi validasi per seed (titik) dan rata-rata (garis)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_router_labels(rows, path):
    """100% stacked bar: distribusi label router (metode terbaik per sampel) train-OOF vs val."""
    bars, data = [], []
    for r in rows:
        counts = r["select"]["router_label_counts"]
        for split, name in (("train_oof", "train (OOF)"), ("val", "val")):
            c = counts[split]
            total = sum(c)
            bars.append(f"seed {r['seed']} · {name}")
            data.append([x / total for x in c])
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    for i, shares in enumerate(data):
        y = len(data) - 1 - i
        left = 0.0
        for j, m in enumerate(METHODS):
            ax.barh(
                y,
                shares[j],
                left=left,
                height=0.62,
                color=COLORS[m],
                edgecolor="white",
                linewidth=2,
                label=m if i == 0 else None,
            )
            if shares[j] >= 0.08:
                ax.text(
                    left + shares[j] / 2,
                    y,
                    f"{shares[j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                )
            left += shares[j]
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels(list(reversed(bars)), fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(
        "Label router: metode dengan P(kelas benar) tertinggi", fontsize=10, color=INK, loc="left"
    )
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_permutation(rows, n_val, path):
    """Per seed: skor label asli (atas) vs 20 skor label diacak setelah seleksi (bawah)."""
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200)
    for i, r in enumerate(rows):
        perm = r["select"]["permutation"]["dynamic_acc"]
        y = len(rows) - 1 - i
        null = perm["null_scores"]
        jitter = [((k % 5) - 2) * 0.035 for k in range(len(null))]
        ax.scatter(
            null,
            [y - 0.2 + j for j in jitter],
            s=16,
            color=MUTED,
            alpha=0.55,
            zorder=2,
            label="label diacak (20x, seleksi diulang)" if i == 0 else None,
        )
        ax.scatter(
            [perm["observed"]],
            [y + 0.18],
            s=80,
            marker="D",
            color=COLORS["dynamic"],
            edgecolor=INK,
            linewidth=0.8,
            zorder=3,
            label="label asli" if i == 0 else None,
        )
        n_ge = sum(1 for s in null if s >= perm["observed"])
        ax.text(
            1.002,
            y,
            f"{n_ge}/{len(null)} acak >= asli\np = {perm['p_value']:.2f}",
            va="center",
            fontsize=7.5,
            color=INK,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"seed {r['seed']}" for r in reversed(rows)], fontsize=9)
    values = [
        x
        for r in rows
        for x in (
            *r["select"]["permutation"]["dynamic_acc"]["null_scores"],
            r["select"]["permutation"]["dynamic_acc"]["observed"],
        )
    ]
    fit_xlim(ax, values, n_val)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Akurasi Dynamic di validasi setelah seleksi", fontsize=9, color=INK)
    ax.set_title(
        "Kontrol permutasi: apakah router belajar sinyal nyata?", fontsize=10, color=INK, loc="left"
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.3), ncol=2)
    style_axes(ax)
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    fig.savefig(path)
    plt.close(fig)


DATASET_LABEL = {
    "appliances_zip1000": "run zip (subsample notebook: 1000 review, 484 sampel valid)",
    "appliances_full": "dataset penuh (semua review Appliances yang punya gambar)",
}


def build_pdf(ctx, rows, figs, out_pdf):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, textColor="#555555")
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=17, leading=21, alignment=0)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=10)

    dataset_id = ctx.data.dataset_id
    n_val = rows[0]["select"]["val"]["n"]
    n_train = len(ctx.dev.train)
    seeds = ", ".join(str(r["seed"]) for r in rows)
    # Dari hasil tersimpan (produsen), bukan dari config pembaca (REV-006).
    n_candidates = len(rows[0]["select"]["candidates"])
    n_perm = rows[0]["select"]["permutation"]["dynamic_acc"]["n_permutations"]
    if any(
        len(r["select"]["candidates"]) != n_candidates
        or r["select"]["permutation"]["dynamic_acc"]["n_permutations"] != n_perm
        for r in rows
    ):
        raise ValueError("jumlah kandidat/permutasi berbeda antar seed")

    def mean_of(fn):
        vals = [fn(r) for r in rows]
        return sum(vals) / len(vals)

    def samples(x: float) -> int:
        return round(x * n_val)

    dyn = mean_of(lambda r: r["select"]["val"]["dynamic_acc"])
    global_means = {m: mean_of(lambda r, m=m: r["select"]["val"]["global_acc"][m]) for m in METHODS}
    best_mean = max(global_means.values())
    router_acc = mean_of(lambda r: r["select"]["val"]["router_acc"])
    majority_router_acc = mean_of(lambda r: r["select"]["val"]["majority_router"]["router_acc"])
    majority_names = sorted({r["select"]["val"]["majority_router"]["method"] for r in rows})
    perms = [r["select"]["permutation"]["dynamic_acc"] for r in rows]
    n_exceed = sum(1 for p in perms if p["exceeds_p95"])

    per_seed = []
    for r in rows:
        v = r["select"]["val"]
        best = max(v["global_acc"].values())
        names = "/".join(m for m in METHODS if v["global_acc"][m] == best)
        per_seed.append((r["seed"], v["dynamic_acc"], best, names))
    diffs = [samples(d - b) for _, d, b, _ in per_seed]
    if all(x == 0 for x in diffs):
        verdict = "<b>sama persis</b> dengan"
    elif all(x > 0 for x in diffs):
        verdict = "sedikit <b>di atas</b>"
    else:
        verdict = "dibandingkan dengan"
    per_seed_txt = "; ".join(
        f"seed {s}: {d:.4f} vs {n} {b:.4f} ({x:+d} sampel)"
        for (s, d, b, n), x in zip(per_seed, diffs, strict=True)
    )
    accs = [
        x
        for r in rows
        for x in (*r["select"]["val"]["global_acc"].values(), r["select"]["val"]["dynamic_acc"])
    ]
    spread = samples(max(accs) - min(accs))
    # Selisih antar ketiga fusion global dalam satu seed (bukan antar seed).
    global_gap = max(
        samples(
            max(r["select"]["val"]["global_acc"].values())
            - min(r["select"]["val"]["global_acc"].values())
        )
        for r in rows
    )
    val_shares = [[c / n_val for c in r["select"]["router_label_counts"]["val"]] for r in rows]
    top_val = [METHODS[max(range(3), key=lambda j, s=s: s[j])] for s in val_shares]
    perm_txt = ", ".join(
        f"seed {r['seed']}: asli {p['observed']:.4f}, acak rata-rata {p['null_mean']:.4f}, "
        f"p = {p['p_value']:.2f}"
        for r, p in zip(rows, perms, strict=True)
    )
    evidence = (
        "<b>belum ada bukti</b> bahwa router berbasis GMM belajar memilih fusion yang tepat"
        if n_exceed <= len(rows) // 2
        else "ada indikasi awal bahwa router belajar sinyal nyata"
    )

    story = [
        Paragraph("Laporan Sementara: GMM-Based Fusion Router", h1),
        Paragraph(
            f"Amazon Reviews 2023 (Appliances), {DATASET_LABEL.get(dataset_id, dataset_id)} "
            f"&middot; seed {seeds} &middot; {date.today().isoformat()}",
            small,
        ),
        Spacer(1, 8),
        Paragraph(
            "<b>Status:</b> hasil <b>validasi</b> saja. Test set belum dipakai; test dijalankan "
            "sekali setelah kode lolos inspeksi.",
            body,
        ),
        Paragraph("Ringkasan", h2),
        Paragraph(
            f"Di validasi ({n_val:,} sampel), Dynamic Fusion {verdict} fusion global terbaik "
            f"pada setiap seed ({per_seed_txt}). Rata-rata Dynamic {dyn:.4f} vs rata-rata satu "
            f"metode terbaik {best_mean:.4f}. Skor Dynamic di validasi bias optimis karena "
            f"konfigurasinya dipilih dari {n_candidates} kandidat di validasi yang sama.",
            body,
        ),
        Paragraph(
            "Pembanding yang adil adalah kontrol permutasi, yaitu seluruh seleksi diulang dengan "
            f"label router diacak ({perm_txt}). Hanya {n_exceed} dari {len(rows)} seed yang "
            f"melewati persentil ke-95 (dengan {n_perm} permutasi, p terkecil yang mungkin "
            f"{1 / (n_perm + 1):.3f}). Router memilih metode terbaik per sampel dengan akurasi "
            f"{router_acc:.2f}, sedangkan selalu memilih {'/'.join(majority_names)} mendapat "
            f"{majority_router_acc:.2f}. Jadi {evidence}.",
            body,
        ),
        Paragraph("Apa yang dilakukan", h2),
        Paragraph(
            "1. Tiga model fusion (mean, concat, cross_attn) dilatih di train "
            f"({n_train:,} sampel), checkpoint dipilih di validasi.<br/>"
            "2. Label router = metode dengan probabilitas kelas benar tertinggi, dihitung dari "
            "prediksi 5-fold di train (model tidak pernah melihat sampel yang ia beri label).<br/>"
            "3. GMM pada embedding gambar + judul + teks (standarisasi &rarr; PCA &rarr; GMM, "
            "fit di train) memberi probabilitas cluster per sampel.<br/>"
            "4. MLP router: probabilitas cluster &rarr; metode fusion. K, PCA, kovarians, dan "
            f"MLP dipilih di validasi ({n_candidates} kandidat).<br/>"
            f"5. Kontrol permutasi: seluruh seleksi diulang {n_perm} kali dengan label router "
            "diacak.",
            body,
        ),
        Paragraph("Hasil", h2),
        Image(str(figs["accuracy"]), width=17 * cm, height=17 * cm * 3.2 / 7.2),
        Paragraph(
            f"Gambar 1. Akurasi validasi. Satu sampel = {1 / n_val:.5f}; seluruh selisih di "
            f"gambar ini paling banyak {spread} sampel dari {n_val:,}.",
            small,
        ),
        Spacer(1, 6),
    ]

    header = [
        "Seed",
        "GMM terpilih (PCA, kov., K)",
        "Router",
        "Dynamic",
        "Global terbaik",
        "Akurasi router",
        "p permutasi",
    ]
    table = [header]
    for r in rows:
        s, v = r["select"], r["select"]["val"]
        g = s["selected"]["gmm"]
        rt = s["selected"]["router"]
        best = max(METHODS, key=lambda m: v["global_acc"][m])
        table.append(
            [
                str(r["seed"]),
                f"{g['pca_dim']}, {g['covariance_type']}, {g['n_components']}",
                f"{rt['hidden']} / {rt['class_weight']}",
                f"{v['dynamic_acc']:.4f}",
                f"{best} {v['global_acc'][best]:.4f}",
                f"{v['router_acc']:.2f}",
                f"{s['permutation']['dynamic_acc']['p_value']:.2f}",
            ]
        )
    t = Table(table, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.grey),
                ("ALIGN", (3, 1), (-1, -1), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    shares_txt = "; ".join(
        f"seed {r['seed']}: {m} {max(s):.0%}"
        for r, m, s in zip(rows, top_val, val_shares, strict=True)
    )
    n_ge = [sum(1 for x in p["null_scores"] if x >= p["observed"]) for p in perms]
    story += [
        KeepTogether(
            [
                Paragraph("Tabel 1. Konfigurasi terpilih dan skor validasi per seed.", small),
                Spacer(1, 3),
                t,
            ]
        ),
        Spacer(1, 10),
        Image(str(figs["labels"]), width=17 * cm, height=17 * cm * 3.0 / 7.2),
        Paragraph(
            "Gambar 2. Distribusi label router (metode terbaik per sampel). Label terbanyak di "
            f"validasi: {shares_txt}. Label train (OOF, model fold) dan validasi (model penuh) "
            "berbeda distribusinya.",
            small,
        ),
        Spacer(1, 8),
        Image(str(figs["permutation"]), width=17 * cm, height=17 * cm * 2.9 / 7.2),
        Paragraph(
            "Gambar 3. Skor router dengan label asli (kuning) vs router dengan label acak "
            "(abu-abu). Jumlah permutasi dengan skor >= asli per seed: "
            + ", ".join(f"{k}/{n_perm}" for k in n_ge)
            + ".",
            small,
        ),
        Paragraph("Kesimpulan sementara", h2),
        Paragraph(
            "&bull; Pipeline berjalan lengkap di data ini. Checklist kebocoran data lolos "
            "pemeriksaan reviewer internal dan seluruh tes otomatis lolos; inspeksi independen "
            "(Codex) atas perbaikan terakhir belum selesai.<br/>"
            "&bull; Ketiga fusion global hampir setara di validasi (dalam satu seed, selisihnya "
            f"paling banyak {global_gap} dari {n_val:,} sampel), sehingga label "
            "'metode terbaik per sampel' sebagian besar ditentukan oleh selisih confidence "
            "yang sangat kecil.<br/>"
            "&bull; Langkah berikutnya: (1) inspeksi kode oleh Codex, (2) jalankan test sekali "
            "untuk perbandingan akhir Dynamic vs Global.",
            body,
        ),
        Spacer(1, 6),
        Paragraph(
            f"Sumber angka: results/{dataset_id}/&lt;run_id&gt;/router_selection/seed*.json "
            "dan full_val/*.json (hash diverifikasi saat laporan dibuat). Run select: "
            + ", ".join(f"seed {r['seed']}={r['manifest']['identity']['run_id']}" for r in rows),
            small,
        ),
    ]

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title=f"Laporan Sementara GMM Router ({dataset_id})",
        author="adaptive-fusion",
    )
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/gmm_router.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--select-run", action="append", help="run select eksplisit per seed")
    args = parser.parse_args()

    ctx = build_context(args.config)
    rows = load_results(ctx, ctx.cfg.seeds, args.select_run)
    n_val = rows[0]["select"]["val"]["n"]
    out_dir = args.out / ctx.data.dataset_id
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "accuracy": fig_dir / "gambar1_akurasi_validasi.png",
        "labels": fig_dir / "gambar2_label_router.png",
        "permutation": fig_dir / "gambar3_kontrol_permutasi.png",
    }
    fig_val_accuracy(rows, n_val, figs["accuracy"])
    fig_router_labels(rows, figs["labels"])
    fig_permutation(rows, n_val, figs["permutation"])
    out_pdf = out_dir / "laporan_sementara_gmm_router.pdf"
    build_pdf(ctx, rows, figs, out_pdf)
    print(f"PDF -> {out_pdf}")
    for p in figs.values():
        print(f"gambar -> {p}")


if __name__ == "__main__":
    main()
