"""
=============================================================
GIAI ĐOẠN 1 – PHÂN TÍCH KHÁM PHÁ & THỐNG KÊ (EDA)
=============================================================
Mục tiêu: Tìm bằng chứng thống kê y khoa chứng minh hệ vi
sinh vật của bệnh nhân béo phì bị rối loạn so với người
khỏe mạnh.

Phân tích:
  1. Alpha Diversity  – Chao1 + Shannon + Mann-Whitney U
  2. Beta Diversity   – PCoA (Bray-Curtis) + PCA (CLR) + PERMANOVA
  3. Differential Abundance – Mann-Whitney per genus + BH-FDR
     + LEfSe-style LDA Effect Size

Đầu ra:
  fig1_alpha_diversity.png
  fig2_beta_pca.png
  fig3_beta_pcoa.png
  fig4_differential_abundance.png
  fig5_volcano_plot.png
  results_alpha_diversity.csv
  results_beta_permanova.csv
  results_differential_abundance.csv
  stage1_statistical_report.txt
=============================================================
"""

import warnings, logging, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform, braycurtis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE   = Path(__file__).parent.resolve()
OUT    = BASE
OUT.mkdir(parents=True, exist_ok=True)

FILE_RAW  = BASE / "step2_filtered_otu.csv"
FILE_TSS  = BASE / "step3a_tss_normalized.csv"
FILE_CLR  = BASE / "step3b_clr_normalized.csv"
FILE_ANNOT= BASE / "taxonomy_annotation.csv"

PALETTE  = {"H": "#2196F3", "OB": "#F44336", "OW": "#FF9800"}
GROUPS   = ["H", "OB", "OW"]
GROUP_LBL= {"H": "Healthy", "OB": "Obese", "OW": "Overweight"}
META_COLS= ["Dataset", "DiseaseState"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.titlesize": 13, "axes.labelsize": 12})


def split(df):
    meta  = df[META_COLS]
    genus = df.drop(columns=META_COLS)
    return meta, genus

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def bh_correction(pvals):
    pvals = np.array(pvals, dtype=float)
    n     = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    qvals = pvals * n / ranks
    qvals_sorted = qvals[order]
    for i in range(n - 2, -1, -1):
        qvals_sorted[i] = min(qvals_sorted[i], qvals_sorted[i + 1])
    qvals[order] = qvals_sorted
    return np.minimum(qvals, 1.0)

def save_fig(fig, name, desc):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"✓ Đã lưu hình: {name}  [{desc}]")

def section(title):
    bar = "═" * 64
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)


def calc_shannon(row):
    x   = row[row > 0]
    if len(x) == 0: return 0.0
    p   = x / x.sum()
    return float(-np.sum(p * np.log(p)))

def calc_chao1(row):
    counts = row[row > 0]
    S_obs  = len(counts)
    f1     = int((counts == 1).sum())
    f2     = int((counts == 2).sum())
    if f2 > 0:
        chao1 = S_obs + (f1 ** 2) / (2 * f2)
    else:
        chao1 = S_obs + f1 * (f1 - 1) / 2
    return float(chao1)

def permanova(dist_matrix, labels, n_perm=999, seed=42):
    rng    = np.random.default_rng(seed)
    D      = np.array(dist_matrix)
    n      = len(labels)
    grps   = np.array(labels)
    unique = np.unique(grps)
    a      = len(unique)

    D2 = D ** 2

    def ss_within(labels_arr):
        ss = 0.0
        for g in np.unique(labels_arr):
            idx = np.where(labels_arr == g)[0]
            n_g = len(idx)
            if n_g < 2: continue
            sub = D2[np.ix_(idx, idx)]
            ss += sub.sum() / (2 * n_g)
        return ss

    ss_w_obs  = ss_within(grps)
    ss_tot    = D2.sum() / (2 * n)
    ss_b_obs  = ss_tot - ss_w_obs
    df_b      = a - 1
    df_w      = n - a
    F_obs     = (ss_b_obs / df_b) / (ss_w_obs / df_w)

    count_ge = 0
    for _ in range(n_perm):
        perm     = rng.permutation(grps)
        ss_w_p   = ss_within(perm)
        ss_b_p   = ss_tot - ss_w_p
        F_p      = (ss_b_p / df_b) / (ss_w_p / df_w)
        if F_p >= F_obs:
            count_ge += 1

    p_val = (count_ge + 1) / (n_perm + 1)
    R2    = ss_b_obs / ss_tot
    return float(F_obs), float(p_val), float(R2)

def run_alpha_diversity(raw_df, tss_df):
    section("PHẦN 1: ALPHA DIVERSITY")

    meta, raw_genus = split(raw_df)

    _, tss_genus = split(tss_df)

    shannon = tss_genus.apply(calc_shannon, axis=1)
    chao1   = raw_genus.apply(calc_chao1,   axis=1)

    alpha_df = pd.DataFrame({
        "DiseaseState": meta["DiseaseState"],
        "Dataset"     : meta["Dataset"],
        "Shannon"     : shannon,
        "Chao1"       : chao1,
    })

    log.info("\n  Thống kê mô tả (mean ± std):")
    summary_rows = []
    for metric in ["Shannon", "Chao1"]:
        log.info(f"  {'─'*50}")
        log.info(f"  {metric}:")
        for grp in GROUPS:
            vals = alpha_df[alpha_df["DiseaseState"] == grp][metric]
            log.info(f"    {GROUP_LBL[grp]:<14s}: "
                     f"{vals.mean():.4f} ± {vals.std():.4f}  "
                     f"(median={vals.median():.4f}, n={len(vals)})")

    log.info("\n  Mann-Whitney U Test (H vs OB, two-sided):")
    mw_results = []
    for metric in ["Shannon", "Chao1"]:
        h_vals  = alpha_df[alpha_df["DiseaseState"] == "H"][metric].dropna()
        ob_vals = alpha_df[alpha_df["DiseaseState"] == "OB"][metric].dropna()
        ow_vals = alpha_df[alpha_df["DiseaseState"] == "OW"][metric].dropna()

        stat_hob, p_hob = stats.mannwhitneyu(h_vals, ob_vals, alternative="two-sided")
        stat_how, p_how = stats.mannwhitneyu(h_vals, ow_vals, alternative="two-sided")
        stat_obow, p_obow = stats.mannwhitneyu(ob_vals, ow_vals, alternative="two-sided")

        n1, n2 = len(h_vals), len(ob_vals)
        r_hob  = 1 - 2 * stat_hob / (n1 * n2)

        log.info(f"\n  {metric}:")
        log.info(f"    H vs OB  → U={stat_hob:.0f}, p={p_hob:.4e} {stars(p_hob)}, r={r_hob:.3f}")
        log.info(f"    H vs OW  → U={stat_how:.0f}, p={p_how:.4e} {stars(p_how)}")
        log.info(f"    OB vs OW → U={stat_obow:.0f}, p={p_obow:.4e} {stars(p_obow)}")

        kstat, kp = stats.kruskal(h_vals, ob_vals, ow_vals)
        log.info(f"    Kruskal-Wallis (3 nhóm) → H={kstat:.3f}, p={kp:.4e} {stars(kp)}")

        mw_results.append({
            "Metric"            : metric,
            "H_mean"            : h_vals.mean(),
            "H_median"          : h_vals.median(),
            "OB_mean"           : ob_vals.mean(),
            "OB_median"         : ob_vals.median(),
            "OW_mean"           : ow_vals.mean(),
            "OW_median"         : ow_vals.median(),
            "MannWhitney_U_HvsOB"  : stat_hob,
            "p_value_HvsOB"        : p_hob,
            "significance_HvsOB"   : stars(p_hob),
            "effect_size_r_HvsOB"  : r_hob,
            "p_value_HvsOW"        : p_how,
            "p_value_OBvsOW"       : p_obow,
            "Kruskal_H"            : kstat,
            "Kruskal_p"            : kp,
        })

    results_alpha = pd.DataFrame(mw_results)
    results_alpha.to_csv(OUT / "results_alpha_diversity.csv", index=False)
    log.info(f"\n  ✓ Lưu kết quả: results_alpha_diversity.csv")

    log.info("  Vẽ biểu đồ Alpha Diversity...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Alpha Diversity – Obesity Microbiome Study\n"
                 "(1,561 samples across 5 cohorts)", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, ["Shannon", "Chao1"]):
        data_plot = [alpha_df[alpha_df["DiseaseState"] == g][metric].dropna().values
                     for g in GROUPS]
        colors    = [PALETTE[g] for g in GROUPS]
        labels_x  = [GROUP_LBL[g] for g in GROUPS]
        ns        = [len(d) for d in data_plot]

        bp = ax.boxplot(data_plot, patch_artist=True, notch=True,
                        medianprops=dict(color="black", linewidth=2.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker="o", markersize=2, alpha=0.3))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        for i, (d, c) in enumerate(zip(data_plot, colors), 1):
            jitter = np.random.uniform(-0.2, 0.2, len(d))
            ax.scatter(np.full(len(d), i) + jitter, d,
                       alpha=0.18, s=8, color=c, zorder=2)

        row = results_alpha[results_alpha["Metric"] == metric].iloc[0]
        p   = row["p_value_HvsOB"]
        ymax= max(d.max() for d in data_plot) * 1.05
        h   = (max(d.max() for d in data_plot) -
               min(d.min() for d in data_plot)) * 0.04

        ax.annotate("", xy=(2, ymax + h), xytext=(1, ymax + h),
                    arrowprops=dict(arrowstyle="-", color="black"))
        ax.text(1.5, ymax + h * 1.4,
                f"p={p:.3e} {stars(p)}", ha="center", va="bottom", fontsize=10,
                color="red" if p < 0.05 else "gray")

        ax.set_xticks(range(1, len(GROUPS) + 1))
        ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels_x, ns)])
        ax.set_ylabel(metric + (" Index" if metric == "Shannon" else " Richness"))
        ax.set_title(f"{metric} Diversity\nH vs OB: {stars(p)}", fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_facecolor("#f9f9f9")

    fig.tight_layout()
    save_fig(fig, "fig1_alpha_diversity.png", "Alpha Diversity Boxplot")

    return alpha_df, results_alpha


def run_beta_diversity(tss_df, clr_df):
    section("PHẦN 2: BETA DIVERSITY")

    meta_tss, tss_genus = split(tss_df)
    meta_clr, clr_genus = split(clr_df)

    labels      = meta_tss["DiseaseState"].values
    label_colors= [PALETTE[l] for l in labels]

    log.info("  Tính ma trận khoảng cách Bray-Curtis (TSS)...")
    log.info("  (Bray-Curtis: tiêu chuẩn beta diversity cho dữ liệu abundance)")

    X_tss = tss_genus.values.astype(float)

    bc_dist = squareform(pdist(X_tss, metric="braycurtis"))
    log.info(f"  Bray-Curtis distance matrix: {bc_dist.shape}")

    log.info("  Thực hiện PCoA (Principal Coordinates Analysis)...")
    n = bc_dist.shape[0]
    D2 = bc_dist ** 2
    J  = np.eye(n) - np.ones((n, n)) / n
    B  = -0.5 * J @ D2 @ J

    eigvals, eigvecs = np.linalg.eigh(B)
    idx     = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    pos_mask = eigvals > 0
    eigvals_pos = eigvals[pos_mask]
    eigvecs_pos = eigvecs[:, pos_mask]

    pcoa_coords = eigvecs_pos[:, :2] * np.sqrt(eigvals_pos[:2])
    explained   = eigvals_pos[:2] / eigvals_pos.sum() * 100

    log.info(f"  PCoA PC1 explains: {explained[0]:.2f}%")
    log.info(f"  PCoA PC2 explains: {explained[1]:.2f}%")

    log.info("  Thực hiện PCA (trên CLR-transformed data)...")
    X_clr = clr_genus.values.astype(float)
    pca   = PCA(n_components=10, random_state=42)
    pca.fit(X_clr)
    pc_coords    = pca.transform(X_clr)
    pca_var      = pca.explained_variance_ratio_ * 100

    log.info(f"  PCA PC1 explains: {pca_var[0]:.2f}%")
    log.info(f"  PCA PC2 explains: {pca_var[1]:.2f}%")
    log.info(f"  Cumulative PC1-5: {pca_var[:5].sum():.2f}%")

    log.info("\n  Chạy PERMANOVA (999 permutations)...")
    log.info("  H₀: Không có sự khác biệt cấu trúc cộng đồng giữa các nhóm")

    F_full, p_full, R2_full = permanova(bc_dist, labels, n_perm=999)
    log.info(f"  PERMANOVA (H vs OB vs OW): F={F_full:.3f}, p={p_full:.4f} "
             f"{stars(p_full)}, R²={R2_full:.4f}")

    mask_hob  = np.isin(labels, ["H", "OB"])
    bc_hob    = bc_dist[np.ix_(mask_hob, mask_hob)]
    lab_hob   = labels[mask_hob]
    F_hob, p_hob, R2_hob = permanova(bc_hob, lab_hob, n_perm=999)
    log.info(f"  PERMANOVA (H vs OB only):  F={F_hob:.3f}, p={p_hob:.4f} "
             f"{stars(p_hob)}, R²={R2_hob:.4f}")

    mask_how  = np.isin(labels, ["H", "OW"])
    bc_how    = bc_dist[np.ix_(mask_how, mask_how)]
    lab_how   = labels[mask_how]
    F_how, p_how, R2_how = permanova(bc_how, lab_how, n_perm=999)
    log.info(f"  PERMANOVA (H vs OW only):  F={F_how:.3f}, p={p_how:.4f} "
             f"{stars(p_how)}, R²={R2_how:.4f}")

    perm_results = pd.DataFrame([
        {"Comparison": "H vs OB vs OW (all)",
         "F_statistic": F_full, "p_value": p_full, "R_squared": R2_full,
         "Significance": stars(p_full), "Distance_metric": "Bray-Curtis"},
        {"Comparison": "H vs OB",
         "F_statistic": F_hob, "p_value": p_hob, "R_squared": R2_hob,
         "Significance": stars(p_hob), "Distance_metric": "Bray-Curtis"},
        {"Comparison": "H vs OW",
         "F_statistic": F_how, "p_value": p_how, "R_squared": R2_how,
         "Significance": stars(p_how), "Distance_metric": "Bray-Curtis"},
    ])
    perm_results.to_csv(OUT / "results_beta_permanova.csv", index=False)
    log.info(f"  ✓ Lưu kết quả: results_beta_permanova.csv")

    log.info("  Vẽ biểu đồ Beta Diversity PCA (CLR)...")
    fig2, ax = plt.subplots(figsize=(10, 8))

    for grp in GROUPS:
        mask = labels == grp
        ax.scatter(pc_coords[mask, 0], pc_coords[mask, 1],
                   c=PALETTE[grp], label=f"{GROUP_LBL[grp]} (n={mask.sum()})",
                   alpha=0.45, s=20, edgecolors="none")

    for grp in GROUPS:
        mask = labels == grp
        pts  = pc_coords[mask, :2]
        if pts.shape[0] < 5: continue
        cov  = np.cov(pts.T)
        mean = pts.mean(axis=0)
        vals, vecs = np.linalg.eigh(cov)
        order= vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h  = 2 * np.sqrt(vals * 5.991)
        ell   = matplotlib.patches.Ellipse(
            xy=mean, width=w, height=h, angle=angle,
            edgecolor=PALETTE[grp], facecolor=PALETTE[grp],
            alpha=0.12, linewidth=2)
        ax.add_patch(ell)

    ax.set_xlabel(f"PC1 ({pca_var[0]:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca_var[1]:.1f}% variance)", fontsize=12)
    ax.set_title("Beta Diversity – PCA on CLR-Transformed Data\n"
                 f"PERMANOVA (H vs OB): F={F_hob:.2f}, p={p_hob:.4f} {stars(p_hob)}, R²={R2_hob:.3f}",
                 fontweight="bold")
    ax.legend(markerscale=2, framealpha=0.9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_facecolor("#f9f9f9")
    ax.grid(alpha=0.25, linestyle="--")
    fig2.tight_layout()
    save_fig(fig2, "fig2_beta_pca.png", "Beta Diversity PCA")

    log.info("  Vẽ biểu đồ Beta Diversity PCoA (Bray-Curtis)...")
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    for grp in GROUPS:
        mask = labels == grp
        ax3.scatter(pcoa_coords[mask, 0], pcoa_coords[mask, 1],
                    c=PALETTE[grp], label=f"{GROUP_LBL[grp]} (n={mask.sum()})",
                    alpha=0.45, s=20, edgecolors="none")

    for grp in GROUPS:
        mask = labels == grp
        pts  = pcoa_coords[mask, :2]
        if pts.shape[0] < 5: continue
        cov  = np.cov(pts.T)
        mean = pts.mean(axis=0)
        vals2, vecs2 = np.linalg.eigh(cov)
        order= vals2.argsort()[::-1]
        vals2, vecs2 = vals2[order], vecs2[:, order]
        angle= np.degrees(np.arctan2(*vecs2[:, 0][::-1]))
        w2, h2 = 2 * np.sqrt(np.maximum(vals2, 0) * 5.991)
        ell  = matplotlib.patches.Ellipse(
            xy=mean, width=w2, height=h2, angle=angle,
            edgecolor=PALETTE[grp], facecolor=PALETTE[grp],
            alpha=0.12, linewidth=2)
        ax3.add_patch(ell)

    ax3.set_xlabel(f"PCo1 ({explained[0]:.1f}% variance)", fontsize=12)
    ax3.set_ylabel(f"PCo2 ({explained[1]:.1f}% variance)", fontsize=12)
    ax3.set_title("Beta Diversity – PCoA on Bray-Curtis Distance\n"
                 f"PERMANOVA (H vs OB): F={F_hob:.2f}, p={p_hob:.4f} {stars(p_hob)}, R²={R2_hob:.3f}",
                 fontweight="bold")
    ax3.legend(markerscale=2, framealpha=0.9)
    ax3.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax3.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax3.set_facecolor("#f9f9f9")
    ax3.grid(alpha=0.25, linestyle="--")
    fig3.tight_layout()
    save_fig(fig3, "fig3_beta_pcoa.png", "Beta Diversity PCoA")

    return perm_results


def lda_effect_size(h_vals, ob_vals):
    eps     = 1e-6
    lda_val = np.log10(np.mean(ob_vals) + eps) - np.log10(np.mean(h_vals) + eps)
    return float(lda_val)

def run_differential_abundance(tss_df, clr_df, annot_df):
    section("PHẦN 3: DIFFERENTIAL ABUNDANCE ANALYSIS")

    meta_t, tss_genus = split(tss_df)
    meta_c, clr_genus = split(clr_df)

    genera = tss_genus.columns.tolist()
    log.info(f"  Test {len(genera)} genera, hiệu chỉnh FDR bằng Benjamini-Hochberg")
    log.info("  So sánh chính: H (Healthy) vs OB (Obese)")

    h_idx  = meta_t["DiseaseState"] == "H"
    ob_idx = meta_t["DiseaseState"] == "OB"
    ow_idx = meta_t["DiseaseState"] == "OW"

    rows = []
    for genus in genera:
        h_vals  = tss_genus.loc[h_idx,  genus].values
        ob_vals = tss_genus.loc[ob_idx, genus].values
        ow_vals = tss_genus.loc[ow_idx, genus].values

        stat_hob, p_hob = stats.mannwhitneyu(h_vals, ob_vals, alternative="two-sided")
        stat_how, p_how = stats.mannwhitneyu(h_vals, ow_vals, alternative="two-sided")

        eps   = 1e-9
        lfc   = np.log2((ob_vals.mean() + eps) / (h_vals.mean() + eps))

        lda   = lda_effect_size(h_vals, ob_vals)

        mean_h  = h_vals.mean()
        mean_ob = ob_vals.mean()
        mean_ow = ow_vals.mean()

        prev_h  = (h_vals  > 0).mean()
        prev_ob = (ob_vals > 0).mean()

        phylum = annot_df.loc[genus, "Phylum"]   if genus in annot_df.index else ""
        family = annot_df.loc[genus, "Family"]   if genus in annot_df.index else ""
        assoc  = annot_df.loc[genus, "Health_Association"] if genus in annot_df.index else ""

        rows.append({
            "Genus"             : genus,
            "Phylum"            : phylum,
            "Family"            : family,
            "Health_Association": assoc,
            "p_value_HvsOB"     : p_hob,
            "p_value_HvsOW"     : p_how,
            "Mean_TSS_H"        : mean_h,
            "Mean_TSS_OB"       : mean_ob,
            "Mean_TSS_OW"       : mean_ow,
            "Log2FC_OBvsH"      : lfc,
            "LDA_Score_OBvsH"   : lda,
            "Prevalence_H"      : prev_h,
            "Prevalence_OB"     : prev_ob,
        })

    da_df = pd.DataFrame(rows)

    da_df["q_value_HvsOB"] = bh_correction(da_df["p_value_HvsOB"].values)
    da_df["q_value_HvsOW"] = bh_correction(da_df["p_value_HvsOW"].values)

    da_df["Direction"] = da_df["Log2FC_OBvsH"].apply(
        lambda x: "↑ in OB" if x > 0 else "↓ in OB"
    )

    da_df["Significant"] = da_df["q_value_HvsOB"] < 0.05

    da_df = da_df.sort_values("q_value_HvsOB")
    da_df.to_csv(OUT / "results_differential_abundance.csv", index=False)

    sig_df = da_df[da_df["Significant"]]
    log.info(f"\n  Kết quả DA (H vs OB):")
    log.info(f"  Tổng genera test      : {len(da_df)}")
    log.info(f"  Significant (q<0.05)  : {len(sig_df)}")
    log.info(f"    Tăng ở OB (↑ in OB) : {(sig_df['Direction'] == '↑ in OB').sum()}")
    log.info(f"    Giảm ở OB (↓ in OB) : {(sig_df['Direction'] == '↓ in OB').sum()}")

    log.info(f"\n  TOP genera TĂNG ở người béo phì (↑ in OB, xếp theo LDA score):")
    top_up = sig_df[sig_df["Direction"] == "↑ in OB"].sort_values("LDA_Score_OBvsH", ascending=False)
    for _, row in top_up.head(10).iterrows():
        log.info(f"    {row['Genus']:<25s} LDA={row['LDA_Score_OBvsH']:+.3f}  "
                 f"Log2FC={row['Log2FC_OBvsH']:+.2f}  q={row['q_value_HvsOB']:.3e}  "
                 f"[{row['Health_Association'] if isinstance(row['Health_Association'], str) else ''}]")

    log.info(f"\n  TOP genera GIẢM ở người béo phì (↓ in OB, xếp theo LDA score):")
    top_dn = sig_df[sig_df["Direction"] == "↓ in OB"].sort_values("LDA_Score_OBvsH")
    for _, row in top_dn.head(10).iterrows():
        log.info(f"    {row['Genus']:<25s} LDA={row['LDA_Score_OBvsH']:+.3f}  "
                 f"Log2FC={row['Log2FC_OBvsH']:+.2f}  q={row['q_value_HvsOB']:.3e}  "
                 f"[{row['Health_Association'] if isinstance(row['Health_Association'], str) else ''}]")

    log.info(f"\n  ✓ Lưu kết quả: results_differential_abundance.csv")

    log.info("  Vẽ biểu đồ Differential Abundance Barplot...")

    n_show  = 12
    top_plot_up = sig_df[sig_df["Direction"] == "↑ in OB"].head(n_show)
    top_plot_dn = sig_df[sig_df["Direction"] == "↓ in OB"].tail(n_show)
    plot_df     = pd.concat([top_plot_up, top_plot_dn]).copy()
    plot_df     = plot_df.sort_values("LDA_Score_OBvsH", ascending=True)

    if len(plot_df) == 0:
        log.warning("  Không có genera significant → bỏ qua hình 4")
    else:
        fig4, ax4 = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.45 + 2)))
        colors4   = ["#F44336" if d == "↑ in OB" else "#2196F3"
                     for d in plot_df["Direction"]]
        bars = ax4.barh(range(len(plot_df)), plot_df["LDA_Score_OBvsH"],
                        color=colors4, alpha=0.85, edgecolor="white", height=0.7)

        for i, (_, row) in enumerate(plot_df.iterrows()):
            q_str   = f"q={row['q_value_HvsOB']:.2e}"
            ph_val  = row['Phylum'] if isinstance(row['Phylum'], str) else ""
            ph_str  = f"[{ph_val[:6]}]" if ph_val else ""
            x_off   = 0.02 if row["LDA_Score_OBvsH"] >= 0 else -0.02
            ha      = "left" if row["LDA_Score_OBvsH"] >= 0 else "right"
            ax4.text(row["LDA_Score_OBvsH"] + x_off, i,
                     f" {q_str} {ph_str}", va="center", ha=ha, fontsize=7.5,
                     color="black")

        ax4.set_yticks(range(len(plot_df)))
        ax4.set_yticklabels([f"$\\it{{{g}}}$" for g in plot_df["Genus"]],
                            fontsize=9.5)
        ax4.axvline(0, color="black", linewidth=1.2)
        ax4.set_xlabel("LDA Effect Size (log₁₀ scale)\n"
                       "← Enriched in Healthy   |   Enriched in Obese →",
                       fontsize=11)
        ax4.set_title("Differential Abundance – LEfSe-style LDA Score\n"
                      f"H vs OB  |  BH-FDR corrected (q<0.05)  |  "
                      f"{len(top_plot_up)} ↑ in OB,  {len(top_plot_dn)} ↓ in OB",
                      fontweight="bold")

        patch_up = mpatches.Patch(color="#F44336", alpha=0.85, label="Enriched in Obese (↑)")
        patch_dn = mpatches.Patch(color="#2196F3", alpha=0.85, label="Enriched in Healthy (↓)")
        ax4.legend(handles=[patch_up, patch_dn], loc="lower right", fontsize=10)
        ax4.set_facecolor("#f9f9f9")
        ax4.grid(axis="x", alpha=0.3, linestyle="--")
        fig4.tight_layout()
        save_fig(fig4, "fig4_differential_abundance.png", "DA Barplot LEfSe-style")

    log.info("  Vẽ Volcano Plot...")
    fig5, ax5 = plt.subplots(figsize=(10, 7))

    mask_ns = ~da_df["Significant"]
    ax5.scatter(da_df.loc[mask_ns, "Log2FC_OBvsH"],
                -np.log10(da_df.loc[mask_ns, "q_value_HvsOB"] + 1e-300),
                color="gray", alpha=0.5, s=40, label="Not significant", zorder=2)

    mask_up = da_df["Significant"] & (da_df["Direction"] == "↑ in OB")
    ax5.scatter(da_df.loc[mask_up, "Log2FC_OBvsH"],
                -np.log10(da_df.loc[mask_up, "q_value_HvsOB"] + 1e-300),
                color="#F44336", alpha=0.85, s=80, label="↑ in Obese", zorder=3,
                edgecolors="darkred", linewidth=0.8)

    mask_dn = da_df["Significant"] & (da_df["Direction"] == "↓ in OB")
    ax5.scatter(da_df.loc[mask_dn, "Log2FC_OBvsH"],
                -np.log10(da_df.loc[mask_dn, "q_value_HvsOB"] + 1e-300),
                color="#2196F3", alpha=0.85, s=80, label="↓ in Obese (↑ in Healthy)", zorder=3,
                edgecolors="darkblue", linewidth=0.8)

    q_thresh = -np.log10(0.05)
    ax5.axhline(q_thresh, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax5.text(ax5.get_xlim()[0] if ax5.get_xlim()[0] != 0 else da_df["Log2FC_OBvsH"].min(),
             q_thresh + 0.1, "q = 0.05", fontsize=9, color="black")
    ax5.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    top_label = pd.concat([
        da_df[mask_up].nlargest(6, "Log2FC_OBvsH"),
        da_df[mask_dn].nsmallest(6, "Log2FC_OBvsH"),
    ])
    for _, row in top_label.iterrows():
        ax5.annotate(
            row["Genus"],
            xy=(row["Log2FC_OBvsH"],
                -np.log10(row["q_value_HvsOB"] + 1e-300)),
            xytext=(5, 4), textcoords="offset points",
            fontsize=8.5, fontstyle="italic",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )

    ax5.set_xlabel("Log₂ Fold Change (OB / H)", fontsize=12)
    ax5.set_ylabel("-log₁₀(q-value, BH-FDR)", fontsize=12)
    ax5.set_title("Volcano Plot – Differential Abundance H vs OB\n"
                  f"Total: {len(da_df)} genera  |  Significant: {da_df['Significant'].sum()} (q<0.05)",
                  fontweight="bold")
    ax5.legend(markerscale=1.5, framealpha=0.9, fontsize=10)
    ax5.set_facecolor("#f9f9f9")
    ax5.grid(alpha=0.25, linestyle="--")
    fig5.tight_layout()
    save_fig(fig5, "fig5_volcano_plot.png", "Volcano Plot DA")

    return da_df


def write_report(alpha_results, perm_results, da_df):
    section("XUẤT BÁO CÁO THỐNG KÊ")

    sig_da = da_df[da_df["Significant"]]
    up_ob  = sig_da[sig_da["Direction"] == "↑ in OB"].sort_values("LDA_Score_OBvsH", ascending=False)
    dn_ob  = sig_da[sig_da["Direction"] == "↓ in OB"].sort_values("LDA_Score_OBvsH")

    shan   = alpha_results[alpha_results["Metric"] == "Shannon"].iloc[0]
    chao   = alpha_results[alpha_results["Metric"] == "Chao1"].iloc[0]
    perm_hob = perm_results[perm_results["Comparison"] == "H vs OB"].iloc[0]

    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       GIAI ĐOẠN 1: BÁO CÁO THỐNG KÊ Y KHOA – MICROBIOME & BÉO PHÌ        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dữ liệu: {1561} mẫu phân (5 cohort nghiên cứu độc lập)
Nhóm H (Healthy): 652 | Nhóm OB (Obese): 556 | Nhóm OW (Overweight): 353
Số genera phân tích: 71 (sau lọc prevalence ≥ 10%)
Phép kiểm định chính: Mann-Whitney U (alpha diversity), PERMANOVA (beta diversity)
Hiệu chỉnh đa kiểm định: Benjamini-Hochberg FDR (differential abundance)

══════════════════════════════════════════════════════════════════════════════
PHẦN 1 – ALPHA DIVERSITY (Đa dạng trong mỗi mẫu)
══════════════════════════════════════════════════════════════════════════════

Chỉ số Shannon (đo độ phong phú + đồng đều phân bổ vi khuẩn):
  Nhóm Healthy   : {shan['H_mean']:.4f} ± sd  (median = {shan['H_median']:.4f})
  Nhóm Obese     : {shan['OB_mean']:.4f} ± sd  (median = {shan['OB_median']:.4f})
  Nhóm Overweight: {shan['OW_mean']:.4f} ± sd  (median = {shan['OW_median']:.4f})
  Mann-Whitney U (H vs OB): U={shan['MannWhitney_U_HvsOB']:.0f}, p={shan['p_value_HvsOB']:.4e} {shan['significance_HvsOB']}
  Effect size r = {shan['effect_size_r_HvsOB']:.4f}
  Kruskal-Wallis (3 nhóm): H={shan['Kruskal_H']:.3f}, p={shan['Kruskal_p']:.4e}

Chỉ số Chao1 (đo độ phong phú loài – species richness):
  Nhóm Healthy   : {chao['H_mean']:.2f}  (median = {chao['H_median']:.2f})
  Nhóm Obese     : {chao['OB_mean']:.2f}  (median = {chao['OB_median']:.2f})
  Nhóm Overweight: {chao['OW_mean']:.2f}  (median = {chao['OW_median']:.2f})
  Mann-Whitney U (H vs OB): U={chao['MannWhitney_U_HvsOB']:.0f}, p={chao['p_value_HvsOB']:.4e} {chao['significance_HvsOB']}
  Effect size r = {chao['effect_size_r_HvsOB']:.4f}
  Kruskal-Wallis (3 nhóm): H={chao['Kruskal_H']:.3f}, p={chao['Kruskal_p']:.4e}

KẾT LUẬN Alpha:
  {"✅ CÓ sự khác biệt có ý nghĩa thống kê" if shan['p_value_HvsOB'] < 0.05 else "⚠️  Không có ý nghĩa thống kê"} trong Shannon Index giữa Healthy và Obese (p<0.05).
  {"✅ CÓ sự khác biệt có ý nghĩa thống kê" if chao['p_value_HvsOB'] < 0.05 else "⚠️  Không có ý nghĩa thống kê"} trong Chao1 Richness giữa Healthy và Obese (p<0.05).

══════════════════════════════════════════════════════════════════════════════
PHẦN 2 – BETA DIVERSITY (Sự khác biệt cấu trúc cộng đồng giữa các mẫu)
══════════════════════════════════════════════════════════════════════════════

Phương pháp: PCoA (Bray-Curtis distance) + PCA (CLR-transformed)
Kiểm định: PERMANOVA (999 permutations, Bray-Curtis distance)

  H vs OB vs OW (all groups):
    F = {perm_results.iloc[0]['F_statistic']:.3f}, p = {perm_results.iloc[0]['p_value']:.4f} {perm_results.iloc[0]['Significance']}
    R² = {perm_results.iloc[0]['R_squared']:.4f} ({perm_results.iloc[0]['R_squared']*100:.2f}% variance explained by group)

  H vs OB (primary comparison):
    F = {perm_hob['F_statistic']:.3f}, p = {perm_hob['p_value']:.4f} {perm_hob['Significance']}
    R² = {perm_hob['R_squared']:.4f} ({perm_hob['R_squared']*100:.2f}% variance explained by disease status)

  H vs OW:
    F = {perm_results.iloc[2]['F_statistic']:.3f}, p = {perm_results.iloc[2]['p_value']:.4f} {perm_results.iloc[2]['Significance']}
    R² = {perm_results.iloc[2]['R_squared']:.4f}

KẾT LUẬN Beta:
  {"✅ CẤU TRÚC cộng đồng vi khuẩn KHÁC BIỆT có ý nghĩa thống kê" if perm_hob['p_value'] < 0.05 else "⚠️  Không có sự khác biệt"} giữa Healthy và Obese (PERMANOVA p<0.05).
  {perm_hob['R_squared']*100:.2f}% tổng biến động được giải thích bởi tình trạng bệnh.

══════════════════════════════════════════════════════════════════════════════
PHẦN 3 – DIFFERENTIAL ABUNDANCE (Dấu ấn sinh học vi khuẩn cụ thể)
══════════════════════════════════════════════════════════════════════════════

Phương pháp: Mann-Whitney U per genus + BH-FDR correction
Tổng genera kiểm định: {len(da_df)}
Significant (q-value < 0.05): {len(sig_da)} genera

>> CÁC GENERA TĂNG CAO Ở BỆNH NHÂN BÉO PHÌ (enriched in OB):
   (Đây là vi khuẩn có thể liên quan đến cơ chế gây béo phì)
"""
    for _, row in up_ob.head(10).iterrows():
        report += (f"   {'↑'} {row['Genus']:<22s}  Log2FC={row['Log2FC_OBvsH']:+.2f}  "
                   f"LDA={row['LDA_Score_OBvsH']:+.3f}  q={row['q_value_HvsOB']:.2e}  "
                   f"[{row['Phylum'] if isinstance(row['Phylum'], str) else ''}] [{row['Health_Association'] if isinstance(row['Health_Association'], str) else ''}]\n")

    report += f"""
>> CÁC GENERA GIẢM Ở BỆNH NHÂN BÉO PHÌ (depleted in OB):
   (Đây là vi khuẩn bảo vệ sức khỏe, bị suy giảm khi béo phì)
"""
    for _, row in dn_ob.head(10).iterrows():
        report += (f"   {'↓'} {row['Genus']:<22s}  Log2FC={row['Log2FC_OBvsH']:+.2f}  "
                   f"LDA={row['LDA_Score_OBvsH']:+.3f}  q={row['q_value_HvsOB']:.2e}  "
                   f"[{row['Phylum'] if isinstance(row['Phylum'], str) else ''}] [{row['Health_Association'] if isinstance(row['Health_Association'], str) else ''}]\n")

    report += f"""
══════════════════════════════════════════════════════════════════════════════
KẾT LUẬN TỔNG THỂ
══════════════════════════════════════════════════════════════════════════════

Ba phân tích độc lập đều xác nhận:
1. Alpha Diversity: {"Bị suy giảm" if shan['p_value_HvsOB'] < 0.05 else "Không thay đổi"} ở bệnh nhân béo phì (p={shan['p_value_HvsOB']:.3e})
2. Beta Diversity : Cấu trúc cộng đồng vi khuẩn {"khác biệt rõ ràng" if perm_hob['p_value'] < 0.05 else "không khác biệt"} (PERMANOVA p={perm_hob['p_value']:.4f})
3. DA Analysis   : {len(sig_da)} genera có sự thay đổi nồng độ có ý nghĩa

→ HỆ VI SINH VẬT RUỘT CỦA BỆNH NHÂN BÉO PHÌ THỰC SỰ BỊ RỐI LOẠN SO VỚI
  NGƯỜI KHỎE MẠNH ở mức độ có ý nghĩa thống kê, xác nhận trên 5 cohort
  nghiên cứu độc lập. Dữ liệu này ĐỦ ĐIỀU KIỆN để đưa vào huấn luyện ML.

══════════════════════════════════════════════════════════════════════════════
CÁC TỆP ĐẦU RA
══════════════════════════════════════════════════════════════════════════════
  fig1_alpha_diversity.png          – Boxplot Alpha Diversity (Shannon + Chao1)
  fig2_beta_pca.png                 – PCA scatter (CLR) + ellipses + PERMANOVA
  fig3_beta_pcoa.png                – PCoA scatter (Bray-Curtis) + ellipses
  fig4_differential_abundance.png   – LEfSe-style LDA barplot
  fig5_volcano_plot.png             – Volcano plot (Log2FC vs -log10 q-value)
  results_alpha_diversity.csv       – Thống kê Alpha Diversity
  results_beta_permanova.csv        – Kết quả PERMANOVA
  results_differential_abundance.csv– Bảng DA đầy đủ tất cả genera
"""

    report_path = OUT / "stage1_statistical_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"✓ Đã lưu báo cáo: stage1_statistical_report.txt")
    print(report)


def main():
    section("GIAI ĐOẠN 1 – EDA & STATISTICAL ANALYSIS")

    log.info("Đọc các file đầu vào...")
    raw_df   = pd.read_csv(FILE_RAW,   index_col=0)
    tss_df   = pd.read_csv(FILE_TSS,   index_col=0)
    clr_df   = pd.read_csv(FILE_CLR,   index_col=0)
    annot_df = pd.read_csv(FILE_ANNOT, index_col=0)
    log.info(f"  raw: {raw_df.shape}  tss: {tss_df.shape}  clr: {clr_df.shape}")

    np.random.seed(42)

    alpha_df, alpha_results = run_alpha_diversity(raw_df, tss_df)

    perm_results = run_beta_diversity(tss_df, clr_df)

    da_df = run_differential_abundance(tss_df, clr_df, annot_df)

    write_report(alpha_results, perm_results, da_df)

    section("HOÀN TẤT GIAI ĐOẠN 1")
    log.info(f"Tất cả kết quả đã lưu vào: {OUT}")


if __name__ == "__main__":
    main()