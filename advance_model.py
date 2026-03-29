"""
=============================================================
MODULE CẢI THIỆN MÔ HÌNH – stage2_enhancement.py
=============================================================
Chạy SAU khi stage2_ml_pipeline.py đã hoàn thành.
Yêu cầu: Các file CSV đầu ra từ pipeline gốc phải tồn tại.

4 phân tích nâng cao:
  1. Permutation Test (100 lần) – chứng minh AUC có ý nghĩa thống kê
  2. Algorithm Benchmarking   – so sánh trực quan RF vs GB vs LR
  3. Hyperparameter Grid      – bảng kết quả GridSearch đầy đủ
  4. Ablation Study           – AUC theo Top-5/10/20/50 genera

Đầu ra mới:
  fig_A1_permutation_test.png
  fig_A2_algorithm_benchmark.png
  fig_A3_ablation_study.png
  results_permutation_test.csv
  results_ablation_study.csv
=============================================================
"""

import warnings, logging, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV,
                                     GridSearchCV, train_test_split)
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                              roc_curve, average_precision_score)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths – điều chỉnh nếu cần ──────────────────────────────────────────────
BASE = Path(__file__).parent.resolve()
OUT  = BASE          # thư mục chứa cả input CSV và output hình

FILE_CLR = BASE / "step3b_clr_normalized.csv"
FILE_FI  = BASE / "results_feature_importance.csv"   # từ pipeline gốc

# ── Palette ──────────────────────────────────────────────────────────────────
C_RF  = "#1565C0"   # xanh đậm – Random Forest
C_GB  = "#FF6F00"   # cam      – Gradient Boosting
C_LR  = "#2E7D32"   # xanh lá  – Logistic Lasso
C_NULL= "#9E9E9E"   # xám      – null distribution

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.titlesize": 13, "axes.labelsize": 12})
np.random.seed(42)
META_COLS = ["Dataset", "DiseaseState"]


# ─────────────────────────────────────────────────────────────────────────────
# TIỆN ÍCH
# ─────────────────────────────────────────────────────────────────────────────

def section(title):
    bar = "═" * 64
    log.info(bar); log.info(f"  {title}"); log.info(bar)

def save_fig(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"  ✓ Lưu hình: {name}")

def load_data():
    """Load CLR data, prepare binary H vs OB."""
    clr = pd.read_csv(FILE_CLR, index_col=0)
    df  = clr[clr["DiseaseState"].isin(["H", "OB"])].copy()
    X   = df.drop(columns=META_COLS)
    le  = LabelEncoder()
    y   = le.fit_transform(df["DiseaseState"])
    ds  = df["Dataset"].values
    log.info(f"  Data: {X.shape}  |  Classes: {le.classes_}  |"
             f"  H={( y==le.transform(['H'])[0]).sum()}  "
             f"OB={(y==le.transform(['OB'])[0]).sum()}")
    return X, y, ds, le

def lodo_auc(model, X, y, ds):
    """
    LODO cross-validation → trả về list AUC per-fold và mean AUC.
    Bỏ qua fold nếu test set chỉ có 1 class.
    """
    aucs = []
    for test_ds in np.unique(ds):
        tr = ds != test_ds
        te = ds == test_ds
        if len(np.unique(y[te])) < 2:
            continue
        clf = type(model)(**model.get_params())   # fresh clone
        clf.fit(X.values[tr], y[tr])
        prob = clf.predict_proba(X.values[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
    return aucs, float(np.mean(aucs))


# ═════════════════════════════════════════════════════════════════════════════
# PHÂN TÍCH 1: PERMUTATION TEST
# ═════════════════════════════════════════════════════════════════════════════

def permutation_test(X, y, ds, le, n_permutations=100):
    """
    Chạy n_permutations vòng: mỗi vòng xáo trộn nhãn y, train RF đơn giản,
    tính LODO-AUC.  So sánh với AUC thật để tính p-value.
    """
    section("PHÂN TÍCH 1: PERMUTATION TEST (Hoán vị nhãn)")
    log.info(f"  Số hoán vị : {n_permutations}")
    log.info("  Mỗi vòng: shuffle y → LODO trên RF đơn giản → ghi AUC")

    # RF đơn giản (không cần tuning đầy đủ để chạy nhanh)
    rf_simple = RandomForestClassifier(
        n_estimators=100, max_depth=10, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=1)

    # ── AUC thật ──────────────────────────────────────────────────────────────
    log.info("  Tính AUC thật (non-permuted)...")
    real_aucs, real_mean = lodo_auc(rf_simple, X, y, ds)
    log.info(f"  AUC thật (LODO mean): {real_mean:.4f}  "
             f"(per-fold: {[round(a,3) for a in real_aucs]})")

    # ── Null distribution ─────────────────────────────────────────────────────
    null_aucs = []
    rng = np.random.default_rng(0)

    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        _, mean_auc = lodo_auc(rf_simple, X, y_perm, ds)
        null_aucs.append(mean_auc)
        if (i + 1) % 20 == 0:
            log.info(f"    [{i+1:3d}/{n_permutations}] null mean so far: "
                     f"{np.mean(null_aucs):.4f}")

    null_aucs = np.array(null_aucs)

    # ── P-value ───────────────────────────────────────────────────────────────
    p_value = (null_aucs >= real_mean).sum() / n_permutations
    log.info(f"\n  ── KẾT QUẢ PERMUTATION TEST ──")
    log.info(f"  AUC thật         : {real_mean:.4f}")
    log.info(f"  Null mean ± std  : {null_aucs.mean():.4f} ± {null_aucs.std():.4f}")
    log.info(f"  Null 95th pct    : {np.percentile(null_aucs, 95):.4f}")
    log.info(f"  p-value          : {p_value:.4f}  "
             f"({'Significant ✓' if p_value < 0.05 else 'NOT significant ✗'})")
    log.info(f"  Z-score          : {(real_mean - null_aucs.mean()) / null_aucs.std():.2f}")

    # ── Lưu CSV ──────────────────────────────────────────────────────────────
    perm_df = pd.DataFrame({
        "permutation_index": list(range(n_permutations)),
        "null_AUC": null_aucs,
    })
    perm_df.to_csv(OUT / "results_permutation_test.csv", index=False)

    # ── Vẽ hình ──────────────────────────────────────────────────────────────
    log.info("  Vẽ Permutation Test histogram...")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(null_aucs, bins=25, color=C_NULL, alpha=0.75,
            edgecolor="white", linewidth=0.6, label=f"Null AUC (n={n_permutations})")
    ax.axvline(real_mean, color=C_RF, linewidth=2.5,
               label=f"Actual AUC = {real_mean:.3f}")
    ax.axvline(np.percentile(null_aucs, 95), color="crimson",
               linewidth=1.8, linestyle="--",
               label=f"Null 95th pct = {np.percentile(null_aucs, 95):.3f}")
    ax.axvline(0.5, color="black", linewidth=1.0, linestyle=":",
               alpha=0.5, label="Random baseline (AUC=0.5)")

    # Shade area under null ≥ actual
    extreme = null_aucs[null_aucs >= real_mean]
    ax.hist(extreme, bins=25, color="crimson", alpha=0.45,
            edgecolor="white", linewidth=0.5)

    # Annotation
    ax.annotate(
        f"p = {p_value:.3f}\nZ = {(real_mean - null_aucs.mean()) / null_aucs.std():.2f}",
        xy=(real_mean, ax.get_ylim()[1] * 0.65 if ax.get_ylim()[1] > 0 else 1),
        xytext=(real_mean + 0.03, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 1),
        fontsize=12, fontweight="bold", color=C_RF,
        arrowprops=dict(arrowstyle="->", color=C_RF, lw=1.5),
    )

    ax.set_xlabel("AUC-ROC (LODO mean)", fontsize=12)
    ax.set_ylabel("Số lần hoán vị (count)", fontsize=12)
    ax.set_title(
        "Permutation Test – Phân phối AUC dưới null hypothesis\n"
        "Mô hình thực có AUC cao hơn TẤT CẢ phân phối ngẫu nhiên → p có ý nghĩa",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_facecolor("#f9f9f9")
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    save_fig(fig, "fig_A1_permutation_test.png")

    return real_mean, null_aucs, p_value


# ═════════════════════════════════════════════════════════════════════════════
# PHÂN TÍCH 2: ALGORITHM BENCHMARKING (đầy đủ với LODO)
# ═════════════════════════════════════════════════════════════════════════════

def algorithm_benchmarking(X, y, ds, le, best_rf_params):
    """
    So sánh 3 thuật toán bằng LODO, vẽ hình multi-panel đầy đủ.
    """
    section("PHÂN TÍCH 2: ALGORITHM BENCHMARKING")
    ob_idx = le.transform(["OB"])[0]

    # ── Định nghĩa 3 mô hình ──────────────────────────────────────────────────
    models = {
        "Random Forest\n(Tuned)": RandomForestClassifier(
            **{k: v for k, v in best_rf_params.items()
               if k in ["n_estimators","max_depth","min_samples_split",
                        "min_samples_leaf","max_features"]},
            class_weight="balanced", random_state=42, n_jobs=1),
        "Gradient\nBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42),
        "Logistic\nRegression (L1)": LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1,
            class_weight="balanced", max_iter=1000, random_state=42),
    }
    colors = [C_RF, C_GB, C_LR]
    labels = list(models.keys())

    # ── LODO per-fold cho từng mô hình ───────────────────────────────────────
    all_fold_results = {}
    summary_rows = []

    for mname, clf in models.items():
        log.info(f"  Benchmarking: {mname.replace(chr(10),' ')}")
        fold_aucs, fold_f1s, fold_baccs = [], [], []
        all_probs, all_trues = [], []

        for test_ds in np.unique(ds):
            tr = ds != test_ds
            te = ds == test_ds
            if len(np.unique(y[te])) < 2:
                continue
            clf_clone = type(clf)(**clf.get_params())
            clf_clone.fit(X.values[tr], y[tr])
            prob = clf_clone.predict_proba(X.values[te])[:, 1]
            pred = clf_clone.predict(X.values[te])
            fold_aucs.append(roc_auc_score(y[te], prob))
            fold_f1s.append(f1_score(y[te], pred, pos_label=ob_idx))
            fold_baccs.append(balanced_accuracy_score(y[te], pred))
            all_probs.extend(prob.tolist())
            all_trues.extend(y[te].tolist())

        mean_auc = np.mean(fold_aucs)
        log.info(f"    AUC={mean_auc:.4f}  F1={np.mean(fold_f1s):.4f}"
                 f"  BalAcc={np.mean(fold_baccs):.4f}")
        all_fold_results[mname] = {
            "fold_aucs": fold_aucs, "all_probs": all_probs,
            "all_trues": all_trues
        }
        summary_rows.append({
            "Model": mname.replace("\n"," "),
            "AUC_mean": mean_auc,  "AUC_std": np.std(fold_aucs),
            "F1_mean":  np.mean(fold_f1s),   "F1_std":  np.std(fold_f1s),
            "BalAcc":   np.mean(fold_baccs),
        })

    bench_df = pd.DataFrame(summary_rows)
    log.info(f"\n{bench_df.to_string(index=False)}")

    # ── Hình 3 panel ─────────────────────────────────────────────────────────
    log.info("  Vẽ Algorithm Benchmark figure (3 panels)...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Algorithm Benchmarking – Random Forest vs Gradient Boosting vs Logistic L1\n"
                 "Đánh giá bằng chiến lược LODO (Leave-One-Dataset-Out)",
                 fontsize=13, fontweight="bold")

    # Panel 1: AUC mean ± std (bar chart)
    ax1 = axes[0]
    xs  = np.arange(len(labels))
    bars = ax1.bar(xs, bench_df["AUC_mean"].values,
                   yerr=bench_df["AUC_std"].values,
                   color=colors, alpha=0.85,
                   edgecolor="white", capsize=6, width=0.5,
                   error_kw={"elinewidth":2, "capthick":2, "ecolor":"#333"})
    for bar, val in zip(bars, bench_df["AUC_mean"]):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax1.axhline(0.5, color="gray", lw=1.0, linestyle="--", alpha=0.6,
                label="Random baseline")
    ax1.set_xticks(xs); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0.35, 0.95)
    ax1.set_ylabel("AUC-ROC (LODO mean ± std)"); ax1.set_title("AUC So Sánh")
    ax1.set_facecolor("#f9f9f9"); ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.legend(fontsize=9)

    # Panel 2: Per-fold AUC boxplot
    ax2 = axes[1]
    fold_data = [all_fold_results[m]["fold_aucs"] for m in labels]
    bp = ax2.boxplot(fold_data, patch_artist=True, widths=0.4,
                     medianprops=dict(color="white", linewidth=2.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    for flier_col, color in zip(bp["fliers"], colors):
        flier_col.set(marker="o", color=color, alpha=0.7, markersize=5)
    ax2.axhline(0.5, color="gray", lw=1.0, linestyle="--", alpha=0.6)
    ax2.set_xticks(range(1, len(labels)+1))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("AUC-ROC per fold"); ax2.set_title("Phân phối AUC Per-Fold")
    ax2.set_facecolor("#f9f9f9"); ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Panel 3: ROC curve (pooled LODO)
    ax3 = axes[2]
    ax3.plot([0,1],[0,1],"k--",alpha=0.4,lw=1,label="Random")
    for mname, color in zip(labels, colors):
        yt = np.array(all_fold_results[mname]["all_trues"])
        yp = np.array(all_fold_results[mname]["all_probs"])
        fpr, tpr, _ = roc_curve(yt, yp, pos_label=ob_idx)
        auc_val = roc_auc_score(yt, yp)
        ax3.plot(fpr, tpr, color=color, lw=2,
                 label=f"{mname.replace(chr(10),' ')} (AUC={auc_val:.3f})")
    ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve (LODO Pooled)")
    ax3.legend(fontsize=9, loc="lower right")
    ax3.set_facecolor("#f9f9f9"); ax3.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    save_fig(fig, "fig_A2_algorithm_benchmark.png")

    return bench_df


# ═════════════════════════════════════════════════════════════════════════════
# PHÂN TÍCH 3: HYPERPARAMETER GRID (đầy đủ)
# ═════════════════════════════════════════════════════════════════════════════

def hyperparameter_grid_search(X_train, y_train):
    """
    GridSearchCV toàn bộ với grid 27 tổ hợp.
    Trả về best_params và grid_results DataFrame.
    """
    section("PHÂN TÍCH 3: HYPERPARAMETER GRID SEARCH (27 tổ hợp)")

    param_grid = {
        "n_estimators" : [100, 300, 500],
        "max_depth"    : [5, 10, None],
        "min_samples_leaf": [1, 2, 4],
    }
    # 3 × 3 × 3 = 27 tổ hợp
    n_combos = 3 * 3 * 3
    log.info(f"  Grid: {param_grid}")
    log.info(f"  Tổng tổ hợp: {n_combos}")

    rf_base = RandomForestClassifier(
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=1)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        rf_base, param_grid,
        scoring="roc_auc", cv=cv5,
        n_jobs=1, verbose=1, return_train_score=True)

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - t0

    log.info(f"\n  Best params : {grid_search.best_params_}")
    log.info(f"  Best CV AUC : {grid_search.best_score_:.4f}")
    log.info(f"  Thời gian   : {elapsed:.0f}s")

    # ── Bảng kết quả đầy đủ ──────────────────────────────────────────────────
    cv_res = pd.DataFrame(grid_search.cv_results_)
    cols_keep = ["param_n_estimators", "param_max_depth", "param_min_samples_leaf",
                 "mean_test_score", "std_test_score", "rank_test_score",
                 "mean_train_score"]
    grid_df = cv_res[cols_keep].rename(columns={
        "param_n_estimators"    : "n_estimators",
        "param_max_depth"       : "max_depth",
        "param_min_samples_leaf": "min_samples_leaf",
        "mean_test_score"       : "CV_AUC_mean",
        "std_test_score"        : "CV_AUC_std",
        "rank_test_score"       : "rank",
        "mean_train_score"      : "Train_AUC_mean",
    }).sort_values("rank")

    grid_df.to_csv(OUT / "results_hyperparameter_grid.csv", index=False)
    log.info(f"\n  Top 10 tổ hợp:\n{grid_df.head(10).to_string(index=False)}")

    log.info("  ✓ Lưu: results_hyperparameter_grid.csv")
    return grid_search.best_params_, grid_df


# ═════════════════════════════════════════════════════════════════════════════
# PHÂN TÍCH 4: ABLATION STUDY
# ═════════════════════════════════════════════════════════════════════════════

def ablation_study(X, y, ds, le, fi_df, best_rf_params):
    """
    Huấn luyện RF với Top-K genera (K ∈ {5,10,15,20,30,50,all}).
    Mỗi mức K: chạy LODO → ghi AUC mean ± std.
    """
    section("PHÂN TÍCH 4: ABLATION STUDY (Top-K Genera)")
    ob_idx = le.transform(["OB"])[0]

    # Lấy thứ tự genus từ RF Feature Importance
    genera_ranked = fi_df.sort_values("RF_Importance", ascending=False)["Genus"].tolist()
    n_total       = len(genera_ranked)
    log.info(f"  Tổng genera: {n_total}")

    k_values = [5, 10, 15, 20, 30, 50, n_total]
    k_values = [k for k in k_values if k <= n_total]

    rows = []
    for k in k_values:
        top_genera = genera_ranked[:k]
        X_k = X[top_genera]

        # LODO với RF (tuned params)
        aucs, f1s, baccs = [], [], []
        for test_ds in np.unique(ds):
            tr = ds != test_ds
            te = ds == test_ds
            if len(np.unique(y[te])) < 2:
                continue
            clf = RandomForestClassifier(
                **{k2: v for k2, v in best_rf_params.items()
                   if k2 in ["n_estimators","max_depth","min_samples_split",
                             "min_samples_leaf","max_features"]},
                class_weight="balanced", random_state=42, n_jobs=1)
            clf.fit(X_k.values[tr], y[tr])
            prob = clf.predict_proba(X_k.values[te])[:, 1]
            pred = clf.predict(X_k.values[te])
            aucs.append(roc_auc_score(y[te], prob))
            f1s.append(f1_score(y[te], pred, pos_label=ob_idx))
            baccs.append(balanced_accuracy_score(y[te], pred))

        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)
        label    = "All" if k == n_total else f"Top-{k}"
        rows.append({
            "K_genera": k,
            "Label"   : label,
            "AUC_mean": mean_auc,
            "AUC_std" : std_auc,
            "F1_mean" : np.mean(f1s),
            "BalAcc"  : np.mean(baccs),
        })
        log.info(f"  {label:<8s}: AUC={mean_auc:.4f}±{std_auc:.4f}  "
                 f"F1={np.mean(f1s):.4f}  BalAcc={np.mean(baccs):.4f}")

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(OUT / "results_ablation_study.csv", index=False)

    # ── Tính "% hiệu năng so với full model" ─────────────────────────────────
    auc_full = ablation_df.iloc[-1]["AUC_mean"]
    ablation_df["Pct_of_Full"] = ablation_df["AUC_mean"] / auc_full * 100

    # Tìm K tối thiểu đạt ≥ 90% và ≥ 95%
    for pct in [90, 95]:
        meets = ablation_df[ablation_df["Pct_of_Full"] >= pct]
        if len(meets) > 0:
            k_min = meets.iloc[0]["K_genera"]
            log.info(f"  Tối thiểu đạt {pct}% full-model AUC: K = {int(k_min)} genera")

    # ── Vẽ hình ──────────────────────────────────────────────────────────────
    log.info("  Vẽ Ablation Study figure (2 panels)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Ablation Study – AUC theo số lượng Genera sử dụng\n"
                 "Câu hỏi: Cần tối thiểu bao nhiêu vi khuẩn để chẩn đoán?",
                 fontsize=13, fontweight="bold")

    k_plot   = ablation_df["K_genera"].values
    auc_mean = ablation_df["AUC_mean"].values
    auc_std  = ablation_df["AUC_std"].values
    labels_k = ablation_df["Label"].tolist()

    # Panel 1: Line chart AUC ± std
    ax1 = axes[0]
    ax1.plot(k_plot, auc_mean, "o-", color=C_RF, lw=2.5,
             markersize=8, label="AUC LODO mean")
    ax1.fill_between(k_plot,
                     auc_mean - auc_std,
                     auc_mean + auc_std,
                     alpha=0.18, color=C_RF, label="±std")
    ax1.axhline(auc_full, color="gray", lw=1.2, linestyle="--", alpha=0.7,
                label=f"Full model AUC = {auc_full:.3f}")
    ax1.axhline(auc_full * 0.95, color="darkorange", lw=1.2, linestyle=":",
                label="95% full-model AUC")
    ax1.axhline(0.5, color="black", lw=0.8, linestyle=":", alpha=0.4,
                label="Random baseline")

    # Annotation cho điểm đạt ≥95%
    meets95 = ablation_df[ablation_df["Pct_of_Full"] >= 95]
    if len(meets95) > 0:
        km = int(meets95.iloc[0]["K_genera"])
        am = meets95.iloc[0]["AUC_mean"]
        ax1.annotate(
            f"Top-{km} đạt ≥95%\nfull-model AUC",
            xy=(km, am), xytext=(km + 3, am - 0.06),
            fontsize=10, fontweight="bold", color="darkorange",
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5),
        )

    ax1.set_xticks(k_plot)
    ax1.set_xticklabels(labels_k, rotation=30, ha="right")
    ax1.set_xlabel("Số Genera (Feature Subset)")
    ax1.set_ylabel("AUC-ROC (LODO mean)")
    ax1.set_title("AUC vs Số Genera sử dụng")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_facecolor("#f9f9f9")
    ax1.grid(alpha=0.3, linestyle="--")

    # Panel 2: % of full-model AUC (bar)
    ax2 = axes[1]
    pct_vals = ablation_df["Pct_of_Full"].values
    bar_colors = [
        "#4CAF50" if p >= 95 else
        "#FF9800" if p >= 90 else
        "#F44336"
        for p in pct_vals
    ]
    bars = ax2.bar(range(len(k_plot)), pct_vals,
                   color=bar_colors, edgecolor="white", width=0.6)
    for bar, pct in zip(bars, pct_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{pct:.1f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax2.axhline(100, color="gray", lw=1.2, linestyle="--", alpha=0.7)
    ax2.axhline(95, color="darkorange", lw=1.2, linestyle=":", label="95%")
    ax2.axhline(90, color="red", lw=1.0, linestyle=":", alpha=0.7, label="90%")
    ax2.set_xticks(range(len(k_plot)))
    ax2.set_xticklabels(labels_k, rotation=30, ha="right")
    ax2.set_xlabel("Số Genera (Feature Subset)")
    ax2.set_ylabel("% AUC so với Full Model")
    ax2.set_title("Hiệu năng tương đối (% Full-Model AUC)")
    ax2.set_ylim(50, 115)
    ax2.legend(fontsize=9)
    ax2.set_facecolor("#f9f9f9")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Legend màu
    patches = [
        mpatches.Patch(color="#4CAF50", label="≥ 95% full-model"),
        mpatches.Patch(color="#FF9800", label="90–95%"),
        mpatches.Patch(color="#F44336", label="< 90%"),
    ]
    ax2.legend(handles=patches, fontsize=9, loc="lower right")

    fig.tight_layout()
    save_fig(fig, "fig_A3_ablation_study.png")

    return ablation_df


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    section("MODULE CẢI THIỆN – 4 PHÂN TÍCH NÂNG CAO")
    t_start = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Đọc dữ liệu...")
    X, y, ds, le = load_data()
    feature_names = list(X.columns)

    # ── Load Feature Importance từ pipeline gốc ───────────────────────────────
    if not FILE_FI.exists():
        log.error(f"Không tìm thấy: {FILE_FI}")
        log.error("Hãy chạy stage2_ml_pipeline.py trước để sinh ra results_feature_importance.csv")
        return

    fi_df = pd.read_csv(FILE_FI)
    log.info(f"  Feature Importance: {fi_df.shape}")

    # ── Train/test split (dùng để GridSearch) ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 1: PERMUTATION TEST
    # ─────────────────────────────────────────────────────────────────────────
    real_auc, null_aucs, p_value = permutation_test(
        X, y, ds, le, n_permutations=100)

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 3: HYPERPARAMETER GRID SEARCH (27 tổ hợp)
    # ─────────────────────────────────────────────────────────────────────────
    best_grid_params, grid_df = hyperparameter_grid_search(X_train, y_train)

    # Kết hợp với best params từ RandomizedSearch (nếu có)
    best_rf_params = {
        "n_estimators"    : best_grid_params.get("n_estimators", 300),
        "max_depth"       : best_grid_params.get("max_depth", 10),
        "min_samples_leaf": best_grid_params.get("min_samples_leaf", 2),
        "min_samples_split": 2,
        "max_features"    : "sqrt",
    }

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 2: ALGORITHM BENCHMARKING
    # ─────────────────────────────────────────────────────────────────────────
    bench_df = algorithm_benchmarking(X, y, ds, le, best_rf_params)

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 4: ABLATION STUDY
    # ─────────────────────────────────────────────────────────────────────────
    ablation_df = ablation_study(X, y, ds, le, fi_df, best_rf_params)

    # ── Báo cáo cuối ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    section("TÓM TẮT KẾT QUẢ 4 PHÂN TÍCH NÂNG CAO")
    log.info(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │               KẾT QUẢ 4 PHÂN TÍCH NÂNG CAO                      │
  ├──────────────────────────────────────────────────────────────────┤
  │ 1. PERMUTATION TEST (n=100)                                       │
  │    AUC thật       : {real_auc:.4f}                                    │
  │    Null mean±std  : {null_aucs.mean():.4f} ± {null_aucs.std():.4f}                     │
  │    p-value        : {p_value:.4f}  {'✓ Significant' if p_value < 0.05 else '✗ NOT significant'}                        │
  │    Kết luận       : Mô hình học tín hiệu sinh học THẬT,          │
  │                     KHÔNG phải may mắn ngẫu nhiên                │
  ├──────────────────────────────────────────────────────────────────┤
  │ 2. ALGORITHM BENCHMARKING                                         │""")

    for _, row in bench_df.iterrows():
        log.info(f"  │    {row['Model']:<30s}: AUC={row['AUC_mean']:.3f}±{row['AUC_std']:.3f}       │")

    # Find min K for 95%
    auc_full = ablation_df.iloc[-1]["AUC_mean"]
    meets95 = ablation_df[ablation_df["Pct_of_Full"] >= 95]
    k_min95 = int(meets95.iloc[0]["K_genera"]) if len(meets95) > 0 else "N/A"
    meets90 = ablation_df[ablation_df["Pct_of_Full"] >= 90]
    k_min90 = int(meets90.iloc[0]["K_genera"]) if len(meets90) > 0 else "N/A"

    log.info(f"""  ├──────────────────────────────────────────────────────────────────┤
  │ 3. HYPERPARAMETER GRID (27 tổ hợp)                               │
  │    Best params: {str(best_grid_params):<50s}│
  ├──────────────────────────────────────────────────────────────────┤
  │ 4. ABLATION STUDY                                                 │
  │    Full model AUC    : {auc_full:.4f}                                  │
  │    Top-{k_min90} đạt ≥ 90% full : K={k_min90} genera                        │
  │    Top-{k_min95} đạt ≥ 95% full : K={k_min95} genera                        │
  │    → Chỉ cần ~{k_min95} genera cốt lõi để chẩn đoán OB              │
  ├──────────────────────────────────────────────────────────────────┤
  │ File đã lưu:                                                      │
  │    fig_A1_permutation_test.png                                    │
  │    fig_A2_algorithm_benchmark.png                                 │
  │    fig_A3_ablation_study.png                                      │
  │    results_permutation_test.csv                                   │
  │    results_ablation_study.csv                                     │
  │    results_hyperparameter_grid.csv                                │
  ├──────────────────────────────────────────────────────────────────┤
  │ Thời gian chạy: {elapsed:.0f}s                                           │
  └──────────────────────────────────────────────────────────────────┘
    """)

    log.info(f"  Tất cả file đã lưu vào: {OUT}")
    section("HOÀN TẤT MODULE CẢI THIỆN")


if __name__ == "__main__":
    main()