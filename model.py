"""
=============================================================
GIAI ĐOẠN 2 & 3 – MACHINE LEARNING PIPELINE
=============================================================
Mục tiêu: Xây dựng và đánh giá mô hình AI chẩn đoán béo phì
từ dữ liệu microbiome đường ruột (CLR-transformed).

Chiến lược:
  A. Binary classification: H vs OB
  B. Multiclass: H vs OW vs OB

Bước 1 : Chuẩn bị dữ liệu & kiểm tra class imbalance
Bước 2 : LODO Cross-Validation (Leave-One-Dataset-Out)
          − Random Forest   (primary model)
          − Gradient Boosting (XGBoost substitute)
          − Logistic Regression Lasso (interpretable baseline)
Bước 3 : Hyperparameter tuning (RandomizedSearchCV + LODO)
Bước 4 : Final model training & evaluation
          − Confusion Matrix, ROC-AUC, PR-AUC, Classification report
Bước 5 : Feature Importance (RF built-in + Permutation)
Bước 6 : SHAP-style analysis (TreeExplainer approximation)
          − Beeswarm summary plot
          − Dependency plots top biomarkers
Bước 7 : Biomarker convergence (ML vs Statistics)

Đầu ra:
  fig6_lodo_roc_curves.png
  fig7_confusion_matrix.png
  fig8_feature_importance.png
  fig9_shap_summary.png
  fig10_shap_dependence.png
  fig11_multiclass_roc.png
  results_lodo_performance.csv
  results_feature_importance.csv
  results_shap_values.csv
=============================================================
"""

import warnings, logging, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV,
                                     cross_val_predict)
from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report,
                              roc_curve, average_precision_score,
                              precision_recall_curve, f1_score,
                              balanced_accuracy_score)
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE   = Path(__file__).parent.resolve()
OUT    = BASE
OUT.mkdir(parents=True, exist_ok=True)

FILE_CLR   = OUT / "step3b_clr_normalized.csv"
FILE_ANNOT = OUT / "taxonomy_annotation.csv"
FILE_DA    = OUT / "results_differential_abundance.csv"

PAL = {"H": "#2196F3", "OB": "#F44336", "OW": "#FF9800"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.titlesize": 13, "axes.labelsize": 12})
np.random.seed(42)

META_COLS = ["Dataset", "DiseaseState"]


def section(title):
    bar = "═" * 64
    log.info(bar); log.info(f"  {title}"); log.info(bar)

def save_fig(fig, name, desc):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"  ✓ Hình: {name}  [{desc}]")

def shap_tree_approx(model, X, n_repeats=10, sample_size=300, random_state=42):
    rng   = np.random.default_rng(random_state)
    n     = min(sample_size, len(X))
    idx   = rng.choice(len(X), size=n, replace=False)
    X_sub = X.iloc[idx].values
    proba = model.predict_proba(X_sub)[:, 1]

    shap_matrix = np.zeros((n, X.shape[1]))
    for j in range(X.shape[1]):
        baseline   = X_sub.copy()
        baseline[:, j] = rng.permutation(baseline[:, j])
        proba_perm = model.predict_proba(baseline)[:, 1]
        shap_matrix[:, j] = proba - proba_perm

    return shap_matrix, idx


def prepare_data(clr_df, mode="binary"):
    if mode == "binary":
        df = clr_df[clr_df["DiseaseState"].isin(["H", "OB"])].copy()
    else:
        df = clr_df.copy()

    X = df.drop(columns=META_COLS)
    y = df["DiseaseState"]
    datasets = df["Dataset"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    log.info(f"  Mode={mode}  |  X: {X.shape}  |  Classes: {le.classes_}")
    log.info(f"  Class distribution: {dict(zip(le.classes_, np.bincount(y_enc)))}")

    return X, y_enc, y, datasets, le


def lodo_cross_validation(X, y_enc, y_str, datasets, le, models_dict):
    section("BƯỚC 2: LEAVE-ONE-DATASET-OUT (LODO) CROSS-VALIDATION")

    unique_ds = datasets.unique()
    log.info(f"  Datasets: {list(unique_ds)}")
    log.info(f"  Số fold : {len(unique_ds)}")

    results = {name: [] for name in models_dict}
    all_preds = {name: {"y_true": [], "y_prob": [], "y_pred": [],
                        "fold_id": []} for name in models_dict}

    for test_ds in unique_ds:
        train_mask = (datasets != test_ds).values
        test_mask  = (datasets == test_ds).values

        X_train = X[train_mask]; y_train = y_enc[train_mask]
        X_test  = X[test_mask];  y_test  = y_enc[test_mask]

        n_test_classes = len(np.unique(y_test))
        if n_test_classes < 2:
            log.warning(f"  [{test_ds}] Chỉ có 1 class trong test → bỏ qua fold này")
            continue

        log.info(f"\n  Fold TEST = {test_ds}  "
                 f"(train n={train_mask.sum()}, test n={test_mask.sum()})")

        for model_name, clf in models_dict.items():
            t0 = time.time()
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = clf.predict(X_test)

            auc   = roc_auc_score(y_test, y_prob)
            f1    = f1_score(y_test, y_pred, average="binary",
                             pos_label=le.transform(["OB"])[0])
            bacc  = balanced_accuracy_score(y_test, y_pred)
            prauc = average_precision_score(y_test, y_prob)
            elapsed = time.time() - t0

            metrics = {"dataset": test_ds, "AUC": auc, "F1": f1,
                       "BalancedAcc": bacc, "PR_AUC": prauc,
                       "n_train": train_mask.sum(), "n_test": test_mask.sum()}
            results[model_name].append(metrics)

            all_preds[model_name]["y_true"].extend(y_test.tolist())
            all_preds[model_name]["y_prob"].extend(y_prob.tolist())
            all_preds[model_name]["y_pred"].extend(y_pred.tolist())
            all_preds[model_name]["fold_id"].extend([test_ds] * len(y_test))

            log.info(f"    [{model_name:<22s}] AUC={auc:.4f}  F1={f1:.4f}  "
                     f"BalAcc={bacc:.4f}  PR-AUC={prauc:.4f}  ({elapsed:.1f}s)")

    log.info("\n  ── TỔNG HỢP LODO ──")
    summary_rows = []
    for model_name, fold_list in results.items():
        if not fold_list: continue
        df_folds = pd.DataFrame(fold_list)
        row = {
            "Model"       : model_name,
            "AUC_mean"    : df_folds["AUC"].mean(),
            "AUC_std"     : df_folds["AUC"].std(),
            "F1_mean"     : df_folds["F1"].mean(),
            "F1_std"      : df_folds["F1"].std(),
            "BalAcc_mean" : df_folds["BalancedAcc"].mean(),
            "PR_AUC_mean" : df_folds["PR_AUC"].mean(),
        }
        summary_rows.append(row)
        log.info(f"  {model_name:<22s} AUC={row['AUC_mean']:.4f}±{row['AUC_std']:.4f}  "
                 f"F1={row['F1_mean']:.4f}±{row['F1_std']:.4f}  "
                 f"BalAcc={row['BalAcc_mean']:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / "results_lodo_performance.csv", index=False)
    log.info("  ✓ Lưu: results_lodo_performance.csv")

    return results, all_preds, summary_df


def tune_random_forest(X_train, y_train):
    section("BƯỚC 3: HYPERPARAMETER TUNING (RandomForest)")

    param_dist = {
        "n_estimators"      : [100, 200, 300, 500],
        "max_depth"         : [None, 5, 10, 15, 20],
        "min_samples_split" : [2, 5, 10],
        "min_samples_leaf"  : [1, 2, 4],
        "max_features"      : ["sqrt", "log2", 0.3, 0.5],
    }

    rf_base = RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1)

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        rf_base, param_dist,
        n_iter=30,
        scoring="roc_auc",
        cv=inner_cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    log.info(f"  Best params : {search.best_params_}")
    log.info(f"  Best CV AUC : {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


def final_evaluation(model, X_test, y_test, le, all_preds_rf, results_rf):
    section("BƯỚC 4: FINAL MODEL EVALUATION")

    y_prob  = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)
    ob_idx  = le.transform(["OB"])[0]

    auc     = roc_auc_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred, average="binary", pos_label=ob_idx)
    bacc    = balanced_accuracy_score(y_test, y_pred)
    prauc   = average_precision_score(y_test, y_prob)

    log.info(f"  Holdout Test AUC        : {auc:.4f}")
    log.info(f"  Holdout Test F1 (OB)    : {f1:.4f}")
    log.info(f"  Holdout Balanced Acc    : {bacc:.4f}")
    log.info(f"  Holdout PR-AUC          : {prauc:.4f}")
    log.info("\n" + classification_report(y_test, y_pred,
                                          target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    log.info(f"  Confusion Matrix:\n{cm}")

    log.info("  Vẽ LODO ROC curves...")
    fig6, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig6.suptitle("LODO Cross-Validation – ROC Curves per Fold\n"
                  "Train: 4 cohorts → Test: 1 unseen cohort",
                  fontsize=13, fontweight="bold")

    model_colors = {"RandomForest": "#F44336",
                    "GradientBoosting": "#FF9800",
                    "LogisticReg_Lasso": "#2196F3"}
    model_labels = {"RandomForest": "Random Forest",
                    "GradientBoosting": "Gradient Boosting",
                    "LogisticReg_Lasso": "Logistic Reg. (Lasso)"}

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    for mname, preds_dict in all_preds_rf.items():
        yt = np.array(preds_dict["y_true"])
        yp = np.array(preds_dict["y_prob"])
        fpr, tpr, _ = roc_curve(yt, yp, pos_label=ob_idx)
        auc_val = roc_auc_score(yt, yp)
        color = model_colors.get(mname, "gray")
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{model_labels.get(mname, mname)} (AUC={auc_val:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC – All Models (LODO pooled)")
    ax.legend(fontsize=9, loc="lower right"); ax.set_facecolor("#f9f9f9")
    ax.grid(alpha=0.3, linestyle="--")

    ax2 = axes[1]
    rf_folds = results_rf.get("RandomForest", [])
    if rf_folds:
        fold_names = [r["dataset"].replace("ob_", "") for r in rf_folds]
        fold_aucs  = [r["AUC"] for r in rf_folds]
        colors_bar = ["#EF9A9A" if a < 0.7 else "#F44336" if a < 0.85
                      else "#B71C1C" for a in fold_aucs]
        bars = ax2.bar(range(len(fold_names)), fold_aucs, color=colors_bar,
                       edgecolor="white", width=0.6)
        ax2.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5)
        ax2.axhline(0.7, color="orange", linestyle="--", lw=1, alpha=0.7,
                    label="AUC = 0.70")
        ax2.axhline(0.8, color="#F44336", linestyle="--", lw=1, alpha=0.7,
                    label="AUC = 0.80")
        mean_auc = np.mean(fold_aucs)
        ax2.axhline(mean_auc, color="#B71C1C", linestyle="-", lw=2,
                    label=f"Mean={mean_auc:.3f}")
        for bar, auc_val in zip(bars, fold_aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{auc_val:.3f}", ha="center", va="bottom", fontsize=9.5,
                     fontweight="bold")
        ax2.set_xticks(range(len(fold_names)))
        ax2.set_xticklabels(fold_names, rotation=30, ha="right")
        ax2.set_ylim(0.4, 1.05)
        ax2.set_ylabel("AUC-ROC"); ax2.set_title("Random Forest – Per-fold AUC")
        ax2.legend(fontsize=9); ax2.set_facecolor("#f9f9f9")
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

    ax3 = axes[2]
    ax3.plot([0, 1], [ob_idx / len(le.classes_)] * 2, "k--",
             alpha=0.4, lw=1, label="Random baseline")
    for mname, preds_dict in all_preds_rf.items():
        yt = np.array(preds_dict["y_true"])
        yp = np.array(preds_dict["y_prob"])
        precision, recall, _ = precision_recall_curve(yt, yp, pos_label=ob_idx)
        ap = average_precision_score(yt, yp)
        color = model_colors.get(mname, "gray")
        ax3.plot(recall, precision, color=color, lw=2,
                 label=f"{model_labels.get(mname, mname)} (AP={ap:.3f})")
    ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
    ax3.set_title("Precision-Recall Curve (LODO pooled)")
    ax3.legend(fontsize=9, loc="lower left"); ax3.set_facecolor("#f9f9f9")
    ax3.grid(alpha=0.3, linestyle="--")

    fig6.tight_layout()
    save_fig(fig6, "fig6_lodo_roc_curves.png", "LODO ROC + PR Curves")

    log.info("  Vẽ Confusion Matrix...")
    rf_preds = all_preds_rf["RandomForest"]
    yt_all   = np.array(rf_preds["y_true"])
    yp_all   = np.array(rf_preds["y_pred"])
    cm_lodo  = confusion_matrix(yt_all, yp_all)

    fig7, axes7 = plt.subplots(1, 2, figsize=(13, 5))
    fig7.suptitle("Random Forest – Confusion Matrix (LODO)", fontweight="bold")

    sns.heatmap(cm_lodo, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, ax=axes7[0], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes7[0].set_xlabel("Predicted"); axes7[0].set_ylabel("True")
    axes7[0].set_title("Count")

    cm_norm = cm_lodo.astype(float) / cm_lodo.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, ax=axes7[1], cbar=False,
                annot_kws={"size": 13})
    axes7[1].set_xlabel("Predicted"); axes7[1].set_ylabel("True")
    axes7[1].set_title("Recall (row-normalized)")

    yt_bin = yt_all; ob_le = le.transform(["OB"])[0]
    h_le   = le.transform(["H"])[0]
    tp = cm_lodo[ob_le, ob_le]
    fn = cm_lodo[ob_le, :].sum() - tp
    fp = cm_lodo[:, ob_le].sum() - tp
    tn = cm_lodo[h_le, h_le]
    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv   = tp / (tp + fp) if (tp + fp) > 0 else 0
    auc_v = roc_auc_score(yt_all, np.array(rf_preds["y_prob"]))

    stats_text = (f"Sensitivity (Recall OB): {sens:.3f}\n"
                  f"Specificity:             {spec:.3f}\n"
                  f"Precision (PPV):         {ppv:.3f}\n"
                  f"AUC-ROC (LODO):          {auc_v:.3f}")
    fig7.text(0.5, -0.04, stats_text, ha="center", fontsize=11,
              bbox=dict(boxstyle="round,pad=0.4", fc="#f0f7ff", ec="#2196F3"))

    fig7.tight_layout()
    save_fig(fig7, "fig7_confusion_matrix.png", "Confusion Matrix LODO")

    return auc, f1, bacc


def feature_importance_analysis(model, X_test, y_test, feature_names,
                                  annot_df, da_df):
    section("BƯỚC 5: FEATURE IMPORTANCE ANALYSIS")

    fi = model.feature_importances_
    fi_df = pd.DataFrame({
        "Genus"         : feature_names,
        "RF_Importance" : fi,
    }).sort_values("RF_Importance", ascending=False)

    log.info("  Tính Permutation Importance (30 repeats)...")
    perm_imp = permutation_importance(
        model, X_test, y_test, n_repeats=30,
        random_state=42, scoring="roc_auc", n_jobs=-1)
    fi_df["Perm_Importance"]     = [perm_imp.importances_mean[
        feature_names.index(g)] for g in fi_df["Genus"]]
    fi_df["Perm_Importance_std"] = [perm_imp.importances_std[
        feature_names.index(g)] for g in fi_df["Genus"]]

    if annot_df is not None:
        fi_df = fi_df.merge(
            annot_df[["Phylum", "Family", "Health_Association"]],
            left_on="Genus", right_index=True, how="left")
    if da_df is not None:
        fi_df = fi_df.merge(
            da_df[["Genus", "Log2FC_OBvsH", "q_value_HvsOB",
                   "LDA_Score_OBvsH", "Direction"]],
            on="Genus", how="left")

    fi_df["RF_Rank"]   = fi_df["RF_Importance"].rank(ascending=False).astype(int)
    fi_df["Perm_Rank"] = fi_df["Perm_Importance"].rank(ascending=False).astype(int)

    fi_df.to_csv(OUT / "results_feature_importance.csv", index=False)

    top20 = fi_df.head(20)
    log.info(f"\n  Top 20 genera (RF importance):")
    for _, r in top20.iterrows():
        phyl = r.get("Phylum", "")[:8] if pd.notna(r.get("Phylum")) else "—"
        lfc  = f"{r['Log2FC_OBvsH']:+.2f}" if pd.notna(r.get("Log2FC_OBvsH")) else "—"
        log.info(f"    {r['Genus']:<28s} RF={r['RF_Importance']:.4f}  "
                 f"Perm={r['Perm_Importance']:.4f}  [{phyl}]  Log2FC={lfc}")

    log.info("  Vẽ Feature Importance plot...")
    n_show = 25
    top_show = fi_df.head(n_show).copy()

    phylum_colors = {
        "Firmicutes":     "#E53935",
        "Bacteroidetes":  "#1E88E5",
        "Actinobacteria": "#8E24AA",
        "Proteobacteria": "#43A047",
        "Verrucomicrobia":"#00ACC1",
        "Euryarchaeota":  "#FB8C00",
        "Lentisphaerae":  "#6D4C41",
    }
    bar_colors = [phylum_colors.get(str(p) if pd.notna(p) else "", "#78909C")
                  for p in top_show["Phylum"]]

    fig8, (ax_main, ax_perm) = plt.subplots(1, 2, figsize=(16, 10))
    fig8.suptitle(f"Feature Importance – Top {n_show} Genera\n"
                  "Random Forest (Built-in) vs Permutation Importance",
                  fontweight="bold", fontsize=13)

    bars = ax_main.barh(range(n_show), top_show["RF_Importance"][::-1].values,
                        color=list(reversed(bar_colors)),
                        edgecolor="white", height=0.7)
    ytick_labels = []
    for _, r in top_show[::-1].iterrows():
        lfc_str = f"  [FC:{r['Log2FC_OBvsH']:+.1f}]" if pd.notna(
            r.get("Log2FC_OBvsH")) else ""
        ytick_labels.append(f"{r['Genus']}{lfc_str}")
    ax_main.set_yticks(range(n_show))
    ax_main.set_yticklabels(ytick_labels, fontsize=8.5, fontstyle="italic")
    ax_main.set_xlabel("Mean Decrease in Impurity (MDI)")
    ax_main.set_title("RF Built-in Importance")
    ax_main.set_facecolor("#f9f9f9")
    ax_main.grid(axis="x", alpha=0.3, linestyle="--")

    used_phyla = top_show["Phylum"].dropna().unique()
    patches = [mpatches.Patch(color=phylum_colors.get(p, "#78909C"), label=p)
               for p in used_phyla if p in phylum_colors]
    ax_main.legend(handles=patches, title="Phylum", fontsize=8,
                   loc="lower right", framealpha=0.9)

    perm_sorted = fi_df.sort_values("Perm_Importance", ascending=False).head(n_show)
    perm_colors = [phylum_colors.get(str(p) if pd.notna(p) else "", "#78909C")
                   for p in perm_sorted["Phylum"][::-1]]
    ax_perm.barh(range(n_show),
                 perm_sorted["Perm_Importance"][::-1].values,
                 xerr=perm_sorted["Perm_Importance_std"][::-1].values,
                 color=perm_colors, edgecolor="white", height=0.7,
                 ecolor="gray", capsize=2)
    ytick_labels2 = []
    for _, r in perm_sorted[::-1].iterrows():
        lfc_str = f"  [FC:{r['Log2FC_OBvsH']:+.1f}]" if pd.notna(
            r.get("Log2FC_OBvsH")) else ""
        ytick_labels2.append(f"{r['Genus']}{lfc_str}")
    ax_perm.set_yticks(range(n_show))
    ax_perm.set_yticklabels(ytick_labels2, fontsize=8.5, fontstyle="italic")
    ax_perm.axvline(0, color="black", lw=0.8, linestyle="--")
    ax_perm.set_xlabel("Mean AUC decrease (permuted − original)")
    ax_perm.set_title("Permutation Importance (±std, 30 repeats)")
    ax_perm.set_facecolor("#f9f9f9")
    ax_perm.grid(axis="x", alpha=0.3, linestyle="--")

    fig8.tight_layout()
    save_fig(fig8, "fig8_feature_importance.png", "Feature Importance")

    return fi_df


def shap_analysis(model, X_train, X_test, y_test, le, feature_names, fi_df):
    section("BƯỚC 6: SHAP-STYLE EXPLAINABILITY ANALYSIS")

    log.info("  Tính SHAP approximation (permutation-based)...")
    shap_matrix, sample_idx = shap_tree_approx(
        model, X_test, n_repeats=15, sample_size=400, random_state=42)

    X_sub     = X_test.iloc[sample_idx]
    y_sub     = y_test[sample_idx]
    ob_le_idx = le.transform(["OB"])[0]

    shap_df = pd.DataFrame(shap_matrix, columns=feature_names)
    shap_df["true_label"] = y_sub
    shap_df.to_csv(OUT / "results_shap_values.csv", index=False)

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:20]
    top_names = [feature_names[i] for i in top_idx]
    top_shap  = shap_matrix[:, top_idx]
    top_vals  = X_sub.values[:, top_idx]

    log.info(f"  Top 10 genera by |mean SHAP|:")
    for i, nm in enumerate(top_names[:10]):
        ms = mean_abs_shap[top_idx[i]]
        mean_shap_dir = shap_matrix[:, top_idx[i]].mean()
        log.info(f"    {nm:<28s} |SHAP|={ms:.5f}  mean={mean_shap_dir:+.5f}")

    log.info("  Vẽ SHAP Summary (beeswarm) plot...")
    fig9, ax9 = plt.subplots(figsize=(11, 9))

    n_top = 18
    y_pos = np.arange(n_top)

    for i in range(n_top):
        shap_vals = top_shap[:, i]
        feat_vals = top_vals[:, i]

        vmin, vmax = feat_vals.min(), feat_vals.max()
        if vmax > vmin:
            feat_norm = (feat_vals - vmin) / (vmax - vmin)
        else:
            feat_norm = np.zeros_like(feat_vals)

        jitter = np.random.uniform(-0.3, 0.3, len(shap_vals))

        scatter = ax9.scatter(
            shap_vals,
            y_pos[i] + jitter,
            c=feat_norm,
            cmap="RdBu_r",
            alpha=0.45, s=12,
            vmin=0, vmax=1)

    ax9.set_yticks(y_pos)
    ax9.set_yticklabels([f"$\\it{{{n}}}$" for n in top_names[:n_top]],
                        fontsize=9.5)
    ax9.axvline(0, color="black", lw=1.2, linestyle="--")
    ax9.set_xlabel("SHAP value  (impact on P(Obese))", fontsize=12)
    ax9.set_title("SHAP Summary Plot – Top 18 Genera\n"
                  "← Pushes toward Healthy    |    Pushes toward Obese →",
                  fontweight="bold")

    cbar = fig9.colorbar(scatter, ax=ax9, pad=0.01)
    cbar.set_label("Feature value\n(low → high CLR)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    ax9.set_facecolor("#f9f9f9")
    ax9.grid(axis="x", alpha=0.3, linestyle="--")
    fig9.tight_layout()
    save_fig(fig9, "fig9_shap_summary.png", "SHAP Summary Beeswarm")

    log.info("  Vẽ SHAP Dependence plots (top 6 genera)...")
    fig10, axes10 = plt.subplots(2, 3, figsize=(15, 9))
    fig10.suptitle("SHAP Dependence Plots – Top 6 Genera\n"
                   "x-axis: CLR-transformed abundance  |  "
                   "y-axis: SHAP impact on P(Obese)",
                   fontweight="bold")

    axes10_flat = axes10.flatten()
    for ax_i, i in enumerate(range(6)):
        ax = axes10_flat[ax_i]
        gname  = top_names[i]
        sv     = top_shap[:, i]
        fv     = top_vals[:, i]
        colors = [PAL["OB"] if l == ob_le_idx else PAL["H"] for l in y_sub]

        ax.scatter(fv, sv, c=colors, alpha=0.45, s=18, edgecolors="none")
        try:
            z = np.polyfit(fv, sv, 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(fv.min(), fv.max(), 100)
            ax.plot(x_line, p_line(x_line), "k-", lw=1.8, alpha=0.7)
        except Exception:
            pass

        ax.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_xlabel(f"CLR({gname})", fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(f"$\\it{{{gname}}}$", fontsize=11, fontweight="bold")
        ax.set_facecolor("#f9f9f9")
        ax.grid(alpha=0.25, linestyle="--")

    patch_ob = mpatches.Patch(color=PAL["OB"], label="Obese")
    patch_h  = mpatches.Patch(color=PAL["H"],  label="Healthy")
    fig10.legend(handles=[patch_ob, patch_h], loc="lower right",
                 fontsize=10, framealpha=0.9)
    fig10.tight_layout()
    save_fig(fig10, "fig10_shap_dependence.png", "SHAP Dependence Plots")

    return top_names


def multiclass_analysis(clr_df, best_rf_params):
    section("BƯỚC 7: MULTICLASS ANALYSIS (H vs OW vs OB)")

    X_mc, y_mc, y_str_mc, ds_mc, le_mc = prepare_data(clr_df, mode="multi")

    clf_mc = RandomForestClassifier(
        **{k: v for k, v in best_rf_params.items()
           if k in ["n_estimators","max_depth","min_samples_split",
                    "min_samples_leaf","max_features"]},
        class_weight="balanced", random_state=42, n_jobs=-1)

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_mc = cross_val_predict(clf_mc, X_mc, y_mc, cv=cv5,
                                   method="predict_proba")
    y_pred_mc = y_prob_mc.argmax(axis=1)

    y_bin_mc = label_binarize(y_mc, classes=range(len(le_mc.classes_)))
    auc_macro = roc_auc_score(y_bin_mc, y_prob_mc, multi_class="ovr",
                               average="macro")
    auc_weighted = roc_auc_score(y_bin_mc, y_prob_mc, multi_class="ovr",
                                  average="weighted")

    log.info(f"  Multiclass AUC (macro)   : {auc_macro:.4f}")
    log.info(f"  Multiclass AUC (weighted): {auc_weighted:.4f}")
    log.info("\n" + classification_report(y_mc, y_pred_mc,
                                          target_names=le_mc.classes_))

    fig11, (ax_roc, ax_cm) = plt.subplots(1, 2, figsize=(14, 6))
    fig11.suptitle("Multiclass Classification: H vs OW vs OB\n"
                   "5-fold Stratified CV – Random Forest",
                   fontweight="bold")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    for i, cls in enumerate(le_mc.classes_):
        fpr, tpr, _ = roc_curve(y_bin_mc[:, i], y_prob_mc[:, i])
        auc_cls = roc_auc_score(y_bin_mc[:, i], y_prob_mc[:, i])
        ax_roc.plot(fpr, tpr, color=PAL.get(cls, "gray"), lw=2,
                    label=f"{cls} (AUC={auc_cls:.3f})")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"One-vs-Rest ROC\nmacro-AUC={auc_macro:.3f}")
    ax_roc.legend(fontsize=10, loc="lower right")
    ax_roc.set_facecolor("#f9f9f9")
    ax_roc.grid(alpha=0.3, linestyle="--")

    cm_mc = confusion_matrix(y_mc, y_pred_mc)
    cm_mc_norm = cm_mc.astype(float) / cm_mc.sum(axis=1, keepdims=True)
    sns.heatmap(cm_mc_norm, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=le_mc.classes_, yticklabels=le_mc.classes_,
                linewidths=0.5, ax=ax_cm, cbar=False,
                annot_kws={"size": 13})
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    ax_cm.set_title("Confusion Matrix (recall normalized)")

    fig11.tight_layout()
    save_fig(fig11, "fig11_multiclass_roc.png", "Multiclass ROC + CM")

    return auc_macro


def convergence_summary(fi_df, top_shap_names, da_df):
    section("TỔNG HỢP: Hội tụ ML × Thống kê")

    top_rf   = set(fi_df.head(20)["Genus"])
    top_shap_set = set(top_shap_names[:18])
    sig_da   = set(da_df[da_df["q_value_HvsOB"] < 0.05]["Genus"])

    converged = top_rf & top_shap_set & sig_da
    rf_da     = top_rf & sig_da - converged
    shap_da   = top_shap_set & sig_da - converged

    log.info(f"  Genera trong Top-20 RF:          {len(top_rf)}")
    log.info(f"  Genera trong Top-18 SHAP:        {len(top_shap_set)}")
    log.info(f"  Genera significant DA (q<0.05):  {len(sig_da)}")
    log.info(f"  HỘI TỤ 3 phương pháp (RF ∩ SHAP ∩ DA): {len(converged)}")
    log.info(f"    → {sorted(converged)}")
    log.info(f"  Hội tụ RF ∩ DA (không có SHAP): {sorted(rf_da)}")

    all_genera = sorted(top_rf | top_shap_set)
    rows = []
    for g in all_genera:
        in_rf   = g in top_rf
        in_shap = g in top_shap_set
        in_da   = g in sig_da
        rf_rank = fi_df[fi_df["Genus"] == g]["RF_Rank"].values
        da_row  = da_df[da_df["Genus"] == g]
        rows.append({
            "Genus"        : g,
            "In_RF_top20"  : in_rf,
            "In_SHAP_top18": in_shap,
            "In_DA_sig"    : in_da,
            "Convergence"  : sum([in_rf, in_shap, in_da]),
            "RF_Rank"      : int(rf_rank[0]) if len(rf_rank) > 0 else 999,
            "Log2FC"       : da_row["Log2FC_OBvsH"].values[0] if len(da_row) > 0 else np.nan,
            "q_value"      : da_row["q_value_HvsOB"].values[0] if len(da_row) > 0 else np.nan,
            "Direction"    : da_row["Direction"].values[0] if len(da_row) > 0 else "—",
        })
    conv_df = pd.DataFrame(rows).sort_values(
        ["Convergence", "RF_Rank"], ascending=[False, True])
    conv_df.to_csv(OUT / "results_biomarker_convergence.csv", index=False)
    log.info("  ✓ Lưu: results_biomarker_convergence.csv")

    return conv_df, converged


def main():
    section("GIAI ĐOẠN 2 & 3 – MACHINE LEARNING PIPELINE")
    t_start = time.time()

    log.info("Đọc dữ liệu đầu vào...")
    clr_df   = pd.read_csv(FILE_CLR,   index_col=0)
    annot_df = pd.read_csv(FILE_ANNOT, index_col=0)
    da_df    = pd.read_csv(FILE_DA)
    log.info(f"  CLR: {clr_df.shape}  |  annot: {annot_df.shape}  |  DA: {da_df.shape}")

    section("BƯỚC 1: CHUẨN BỊ DỮ LIỆU")
    X, y_enc, y_str, datasets, le = prepare_data(clr_df, mode="binary")
    feature_names = list(X.columns)

    ob_le  = le.transform(["OB"])[0]
    h_le   = le.transform(["H"])[0]
    n_ob   = (y_enc == ob_le).sum()
    n_h    = (y_enc == h_le).sum()
    ratio  = max(n_h, n_ob) / min(n_h, n_ob)
    log.info(f"  H={n_h}, OB={n_ob}, imbalance ratio={ratio:.2f}")
    log.info(f"  class_weight='balanced' sẽ được dùng cho tất cả models")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, ds_train, ds_test = train_test_split(
        X, y_enc, datasets, test_size=0.2, random_state=42, stratify=y_enc)
    log.info(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    best_rf, best_params = tune_random_forest(X_train, y_train)
    log.info(f"  Best RF params: {best_params}")

    models_dict = {
        "RandomForest": RandomForestClassifier(
            **{k: v for k, v in best_params.items()
               if k in ["n_estimators","max_depth","min_samples_split",
                        "min_samples_leaf","max_features"]},
            class_weight="balanced", random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42),
        "LogisticReg_Lasso": LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1,
            class_weight="balanced", max_iter=1000, random_state=42),
    }

    results, all_preds, summary_df = lodo_cross_validation(
        X, y_enc, y_str, datasets, le, models_dict)

    best_rf.fit(X_train, y_train)
    auc_final, f1_final, bacc_final = final_evaluation(
        best_rf, X_test, y_test, le, all_preds, results)

    fi_df = feature_importance_analysis(
        best_rf, X_test, y_test, feature_names, annot_df, da_df)

    top_shap_names = shap_analysis(
        best_rf, X_train, X_test, y_test, le, feature_names, fi_df)

    auc_mc = multiclass_analysis(clr_df, best_params)
    conv_df, converged = convergence_summary(fi_df, top_shap_names, da_df)

    elapsed = time.time() - t_start
    section("KẾT QUẢ TỔNG HỢP")
    rf_lodo = summary_df[summary_df["Model"] == "RandomForest"]
    gb_lodo = summary_df[summary_df["Model"] == "GradientBoosting"]
    lr_lodo = summary_df[summary_df["Model"] == "LogisticReg_Lasso"]

    log.info(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │              PERFORMANCE SUMMARY (H vs OB)                  │
  ├────────────────────────┬────────────┬────────────┬──────────┤
  │ Model                  │  LODO AUC  │  LODO F1   │ Holdout  │
  ├────────────────────────┼────────────┼────────────┼──────────┤
  │ Random Forest          │ {rf_lodo["AUC_mean"].values[0]:.3f}±{rf_lodo["AUC_std"].values[0]:.3f} │ {rf_lodo["F1_mean"].values[0]:.3f}±{rf_lodo["F1_std"].values[0]:.3f} │  {auc_final:.3f}   │
  │ Gradient Boosting      │ {gb_lodo["AUC_mean"].values[0]:.3f}±{gb_lodo["AUC_std"].values[0]:.3f} │ {gb_lodo["F1_mean"].values[0]:.3f}±{gb_lodo["F1_std"].values[0]:.3f} │    —     │
  │ Logistic Reg. (Lasso)  │ {lr_lodo["AUC_mean"].values[0]:.3f}±{lr_lodo["AUC_std"].values[0]:.3f} │ {lr_lodo["F1_mean"].values[0]:.3f}±{lr_lodo["F1_std"].values[0]:.3f} │    —     │
  ├────────────────────────┼────────────┼────────────┼──────────┤
  │ Multiclass H/OW/OB     │    —       │    —       │  {auc_mc:.3f}   │
  └────────────────────────┴────────────┴────────────┴──────────┘

  Biomarkers hội tụ 3 phương pháp (RF + SHAP + DA):
  → {sorted(converged)}

  Thời gian chạy: {elapsed:.0f}s
    """)

    log.info("  Tất cả kết quả đã lưu vào: " + str(OUT))
    section("HOÀN TẤT GIAI ĐOẠN 2 & 3")


if __name__ == "__main__":
    main()