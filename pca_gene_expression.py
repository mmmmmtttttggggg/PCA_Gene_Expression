#!/usr/bin/env python3
"""
PCA on GSE5325-style gene expression (Nature-style figure panels a–f).

Reads:
  - data/class.tsv           : one label per line (1 = ER+, 0 = ER-), same order as rows
  - data/filtered.tsv.gz     : samples × genes, header = probe/gene IDs
  - data/columns.tsv.gz      : probe annotation (comment lines with '#')

Writes (default: this directory):
  - pca_panel_a_gata3_xbp1_scatter.png
  - pca_panel_b_gata3_xbp1_pc_axes.png
  - pca_panel_c_pc1_projection_by_er.png
  - pca_panel_d_scree_plot.png
  - pca_panel_e_biplot_pc1_pc2.png
  - pca_panel_f_biplot_pc1_pc2_alt_palette.png
  - table_pca_scores_and_projection.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assignment genes + optional second loading gene (Nature primer uses CCNB2)
GENES_TWO_GENE_SCATTER = ("GATA3", "XBP1")  # panel a/b: x, y


def default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "CBB-Projects" / "pca" / "on_gene_expression_data" / "data"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent


def require_files(data_dir: Path) -> None:
    missing = [
        p.name
        for p in (data_dir / "class.tsv", data_dir / "filtered.tsv.gz", data_dir / "columns.tsv.gz")
        if not p.is_file()
    ]
    if missing:
        print("Missing data files in", data_dir, ":", ", ".join(missing), file=sys.stderr)
        sys.exit(1)


def load_class(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None, sep="\t").iloc[:, 0].astype(int).to_numpy()


def load_probe_annotation(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer", dtype=str, comment="#")


def resolve_probe_column(expr: pd.DataFrame, gene: str, annot: pd.DataFrame) -> str:
    if "ID" not in annot.columns or "GeneSymbol" not in annot.columns:
        raise ValueError("columns.tsv.gz must contain ID and GeneSymbol columns.")
    target = gene.upper().strip()
    headers = set(expr.columns.astype(str).str.strip())
    candidates: list[str] = []
    for _, row in annot.iterrows():
        sym = str(row.get("GeneSymbol", "")).strip().upper()
        if not sym or sym == "NAN":
            continue
        primary = sym.split("|")[0].strip()
        if primary != target:
            continue
        candidates.append(str(row["ID"]).strip())
    for pid in candidates:
        if pid in headers:
            return pid
    raise KeyError(
        f"No probe column in the expression matrix for gene {gene!r} "
        f"(tried {len(candidates)} mapped probe(s))."
    )


def load_expression(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", compression="infer")
    raw.columns = raw.columns.astype(str).str.strip()
    if raw.shape[1] < 2:
        return raw
    first_col = raw.iloc[:, 0]
    rest = raw.iloc[:, 1:]
    try:
        rest.to_numpy(dtype=float)
    except (ValueError, TypeError):
        return raw
    try:
        first_col.to_numpy(dtype=float)
    except (ValueError, TypeError):
        return raw.set_index(raw.columns[0])
    return raw


def resolve_biplot_second_gene(expr: pd.DataFrame, annot: pd.DataFrame) -> tuple[str, str]:
    """Return (gene_symbol, probe_col) for biplot second arrow; prefer CCNB2 else GATA3."""
    for sym in ("CCNB2", "GATA3"):
        try:
            return sym, resolve_probe_column(expr, sym, annot)
        except KeyError:
            continue
    raise RuntimeError("Could not resolve CCNB2 or GATA3 for biplot loadings.")


def draw_pc_lines_raw_space(
    ax: plt.Axes,
    gata3: np.ndarray,
    xbp1: np.ndarray,
    span: float = 4.0,
) -> None:
    """Overlay PC1 and PC2 from PCA on centered raw (GATA3, XBP1); axes match panel a."""
    xy = np.column_stack([gata3, xbp1])
    pca2 = PCA(n_components=2, random_state=0)
    pca2.fit(xy)
    mean = pca2.mean_
    # sklearn components_ are orthogonal directions in feature space (same units as input)
    v1 = pca2.components_[0]
    v2 = pca2.components_[1]
    for v, label, ls in ((v1, "PC1", "-"), (v2, "PC2", "--")):
        t = np.linspace(-span, span, 100)
        line_g = mean[0] + t * v[0]
        line_x = mean[1] + t * v[1]
        ax.plot(line_g, line_x, ls, color="0.35", lw=1.5, label=label)
    ax.legend(loc="upper left", fontsize=8)


def biplot_arrow_scale(scores: np.ndarray, loadings_2genes: np.ndarray) -> float:
    """Scale loading arrows to ~35% of the score range."""
    s_range = np.max(np.abs(scores), axis=0).max()
    l_range = np.max(np.abs(loadings_2genes)) + 1e-12
    return 0.35 * s_range / l_range


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA panels (a–f) for gene expression assignment.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--out-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--scree-pcs",
        type=int,
        default=100,
        help="Max number of PCs to show in scree plot (capped by rank)",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    require_files(data_dir)

    labels = load_class(data_dir / "class.tsv")
    annot = load_probe_annotation(data_dir / "columns.tsv.gz")
    expr = load_expression(data_dir / "filtered.tsv.gz")

    if expr.shape[0] != len(labels):
        raise ValueError(
            f"Sample count mismatch: expression has {expr.shape[0]} rows, "
            f"class.tsv has {len(labels)} lines."
        )

    g_gene, x_gene = GENES_TWO_GENE_SCATTER
    col_g = resolve_probe_column(expr, g_gene, annot)
    col_x = resolve_probe_column(expr, x_gene, annot)
    gata3 = expr[col_g].to_numpy(dtype=float)
    xbp1 = expr[col_x].to_numpy(dtype=float)
    X = expr.to_numpy(dtype=float)

    pos = labels == 1  # ER+
    neg = labels == 0  # ER-

    # --- Panel a: GATA3 (x) vs XBP1 (y), black / red (Nature-style) ---
    fig_a, ax_a = plt.subplots(figsize=(5.5, 5))
    ax_a.scatter(gata3[neg], xbp1[neg], c="black", s=28, edgecolors="none", alpha=0.85, label="ER−")
    ax_a.scatter(gata3[pos], xbp1[pos], c="red", s=28, edgecolors="none", alpha=0.85, label="ER+")
    ax_a.set_xlabel("GATA3")
    ax_a.set_ylabel("XBP1")
    ax_a.set_title("(a) Two-gene expression")
    ax_a.legend(loc="best", frameon=True)
    fig_a.tight_layout()
    fig_a.savefig(out_dir / "pca_panel_a_gata3_xbp1_scatter.png", dpi=200)
    plt.close(fig_a)

    # --- Panel b: same + PC1/PC2 in raw (GATA3, XBP1) space ---
    fig_b, ax_b = plt.subplots(figsize=(5.5, 5))
    ax_b.scatter(gata3[neg], xbp1[neg], c="black", s=28, edgecolors="none", alpha=0.85)
    ax_b.scatter(gata3[pos], xbp1[pos], c="red", s=28, edgecolors="none", alpha=0.85)
    draw_pc_lines_raw_space(ax_b, gata3, xbp1, span=np.ptp(gata3) + np.ptp(xbp1))
    ax_b.set_xlabel("GATA3")
    ax_b.set_ylabel("XBP1")
    ax_b.set_title("(b) Same data with PC axes (2-gene PCA)")
    fig_b.tight_layout()
    fig_b.savefig(out_dir / "pca_panel_b_gata3_xbp1_pc_axes.png", dpi=200)
    plt.close(fig_b)

    # Full-matrix PCA (genes standardized)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_samples, n_features = Xs.shape
    max_rank = min(n_samples, n_features) - 1
    n_scree = min(max(1, args.scree_pcs), max_rank)

    pca_scree = PCA(n_components=n_scree, random_state=0)
    pca_scree.fit(Xs)
    ratios_pct = pca_scree.explained_variance_ratio_ * 100.0

    pca2_full = PCA(n_components=2, random_state=0)
    scores_2d = pca2_full.fit_transform(Xs)

    pca1_full = PCA(n_components=1, random_state=0)
    pc1_scores = pca1_full.fit_transform(Xs).ravel()

    # --- Panel c: PC1 scores (full PCA), rows All / ER− / ER+ ---
    fig_c, ax_c = plt.subplots(figsize=(7, 4.2))
    row_y = {"All": 2.0, "ER−": 1.0, "ER+": 0.0}
    jitter_h = 0.12
    ax_c.scatter(pc1_scores, row_y["All"] + rng.uniform(-jitter_h, jitter_h, size=len(pc1_scores)), c=np.where(labels, "red", "black"), s=22, alpha=0.85)
    ax_c.scatter(pc1_scores[neg], row_y["ER−"] + rng.uniform(-jitter_h, jitter_h, size=neg.sum()), c="black", s=22, alpha=0.85)
    ax_c.scatter(pc1_scores[pos], row_y["ER+"] + rng.uniform(-jitter_h, jitter_h, size=pos.sum()), c="red", s=22, alpha=0.85)
    ax_c.set_xlabel("Projection onto PC1")
    ax_c.set_yticks([0.0, 1.0, 2.0])
    ax_c.set_yticklabels(["ER+", "ER−", "All"])
    ax_c.set_title("(c) Samples on PC1 (full-matrix PCA)")
    ax_c.set_ylim(-0.5, 2.6)
    fig_c.tight_layout()
    fig_c.savefig(out_dir / "pca_panel_c_pc1_projection_by_er.png", dpi=200)
    plt.close(fig_c)

    # --- Panel d: Scree ---
    fig_d, ax_d = plt.subplots(figsize=(7, 3.8))
    xs = np.arange(1, len(ratios_pct) + 1)
    ax_d.bar(xs, ratios_pct, color="0.45", width=0.9, edgecolor="0.2", linewidth=0.2)
    ax_d.set_xlabel("Principal component")
    ax_d.set_ylabel("Proportion of variance (%)")
    ax_d.set_title(f"(d) Scree plot (first {len(ratios_pct)} PCs)")
    ax_d.set_xlim(0.5, len(ratios_pct) + 0.5)
    fig_d.tight_layout()
    fig_d.savefig(out_dir / "pca_panel_d_scree_plot.png", dpi=200)
    plt.close(fig_d)

    # --- Panel e: Biplot PC1 vs PC2 + loading arrows for XBP1 and CCNB2 (or GATA3) ---
    sym2, col_b2 = resolve_biplot_second_gene(expr, annot)
    col_xbp = resolve_probe_column(expr, "XBP1", annot)
    idx_xbp = int(expr.columns.get_loc(col_xbp))
    idx_b2 = int(expr.columns.get_loc(col_b2))
    L = pca2_full.components_[:, [idx_xbp, idx_b2]]  # shape (2, 2): rows PC1/PC2, cols genes
    load_xy = L.T  # rows = genes, cols = (PC1, PC2) in score space
    scale = biplot_arrow_scale(scores_2d, load_xy)
    arrow_ends = load_xy * scale

    fig_e, ax_e = plt.subplots(figsize=(6, 5.5))
    ax_e.scatter(scores_2d[neg, 0], scores_2d[neg, 1], c="black", s=26, alpha=0.85, label="ER−")
    ax_e.scatter(scores_2d[pos, 0], scores_2d[pos, 1], c="red", s=26, alpha=0.85, label="ER+")
    ax_e.axhline(0, color="0.75", lw=0.8, zorder=0)
    ax_e.axvline(0, color="0.75", lw=0.8, zorder=0)
    for name, end in zip(("XBP1", sym2), arrow_ends):
        ax_e.annotate(
            "",
            xy=end,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="forestgreen", lw=1.8, shrinkA=0, shrinkB=0),
        )
        ax_e.scatter(end[0], end[1], c="limegreen", s=40, zorder=5, edgecolors="darkgreen", linewidths=0.5)
        ax_e.text(end[0] * 1.12, end[1] * 1.12, name, fontsize=11, fontstyle="italic", color="darkgreen")
    ax_e.set_xlabel("Projection onto PC1")
    ax_e.set_ylabel("Projection onto PC2")
    ax_e.set_title(f"(e) Biplot (loadings: XBP1, {sym2})")
    ax_e.legend(loc="best")
    fig_e.tight_layout()
    fig_e.savefig(out_dir / "pca_panel_e_biplot_pc1_pc2.png", dpi=200)
    plt.close(fig_e)

    # --- Panel f: same geometry, alternate colors (blue / orange) ---
    fig_f, ax_f = plt.subplots(figsize=(6, 5.5))
    ax_f.scatter(scores_2d[neg, 0], scores_2d[neg, 1], c="#1f77b4", s=28, alpha=0.88, label="ER−")
    ax_f.scatter(scores_2d[pos, 0], scores_2d[pos, 1], c="#ff7f0e", s=28, alpha=0.88, label="ER+")
    ax_f.axhline(0, color="0.75", lw=0.8, zorder=0)
    ax_f.axvline(0, color="0.75", lw=0.8, zorder=0)
    for name, end in zip(("XBP1", sym2), arrow_ends):
        ax_f.annotate(
            "",
            xy=end,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="forestgreen", lw=1.8, shrinkA=0, shrinkB=0),
        )
        ax_f.scatter(end[0], end[1], c="limegreen", s=40, zorder=5, edgecolors="darkgreen", linewidths=0.5)
        ax_f.text(end[0] * 1.12, end[1] * 1.12, name, fontsize=11, fontstyle="italic", color="darkgreen")
    ax_f.set_xlabel("Projection onto PC1")
    ax_f.set_ylabel("Projection onto PC2")
    ax_f.set_title("(f) Same biplot, alternate palette")
    ax_f.legend(loc="best")
    fig_f.tight_layout()
    fig_f.savefig(out_dir / "pca_panel_f_biplot_pc1_pc2_alt_palette.png", dpi=200)
    plt.close(fig_f)

    pca1_two_gene = PCA(n_components=1, random_state=0)
    pc1_two_gene_scores = pca1_two_gene.fit_transform(np.column_stack([gata3, xbp1])).ravel()

    # Table: scores + variance
    out_tbl = pd.DataFrame(
        {
            "sample_index": np.arange(len(labels)),
            "class_er_plus_1": labels,
            "gata3_probe": col_g,
            "xbp1_probe": col_x,
            "biplot_second_gene": sym2,
            "biplot_second_probe": col_b2,
            "pc1_full_matrix": pc1_scores,
            "pc2_full_matrix": scores_2d[:, 1],
            "pc1_two_gene_raw_pca": pc1_two_gene_scores,
        }
    )
    out_tbl.to_csv(out_dir / "table_pca_scores_and_projection.tsv", sep="\t", index=False)

    var_tbl = pd.DataFrame(
        {
            "principal_component": np.arange(1, len(ratios_pct) + 1),
            "explained_variance_ratio": pca_scree.explained_variance_ratio_,
            "explained_variance_pct": ratios_pct,
        }
    )
    var_tbl.to_csv(out_dir / "table_scree_explained_variance.tsv", sep="\t", index=False)

    written = [
        "pca_panel_a_gata3_xbp1_scatter.png",
        "pca_panel_b_gata3_xbp1_pc_axes.png",
        "pca_panel_c_pc1_projection_by_er.png",
        "pca_panel_d_scree_plot.png",
        "pca_panel_e_biplot_pc1_pc2.png",
        "pca_panel_f_biplot_pc1_pc2_alt_palette.png",
        "table_pca_scores_and_projection.tsv",
        "table_scree_explained_variance.tsv",
    ]
    print("Wrote:")
    for name in written:
        print(" ", out_dir / name)


if __name__ == "__main__":
    main()
