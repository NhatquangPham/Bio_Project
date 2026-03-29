"""
=============================================================
GIAI ĐOẠN 0 – DATA ENGINEERING PIPELINE
=============================================================
Input : merged_clean_microbiome.csv  (raw counts, 1561 x 133)
Output:
  step1_otu_with_labels.csv      → đã có sẵn (skip, ghi lại cho đồng bộ)
  step2_filtered_otu.csv         → lọc nhiễu prevalence < 10%
  step3a_tss_normalized.csv      → chuẩn hóa TSS (tần suất tương đối)
  step3b_clr_normalized.csv      → chuẩn hóa CLR (tiêu chuẩn vàng)
  taxonomy_annotation.csv        → bảng phân loại học chi tiết
=============================================================
"""

import io
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.resolve()
INPUT_CSV  = BASE_DIR / "merged_clean_microbiome.csv"
OUTPUT_DIR = BASE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREVALENCE_THRESHOLD = 0.10
MIN_COUNT_THRESHOLD  = 1
CLR_PSEUDOCOUNT      = 0.5


def split_meta_genus(df: pd.DataFrame):
    meta_cols  = ["Dataset", "DiseaseState"]
    genus_cols = [c for c in df.columns if c not in meta_cols]
    return df[meta_cols].copy(), df[genus_cols].copy()


def save_with_meta(genus_df: pd.DataFrame,
                   meta_df:  pd.DataFrame,
                   path:     Path,
                   desc:     str):
    out = pd.concat([meta_df, genus_df], axis=1)
    out.to_csv(path)
    log.info(f"✓ Đã lưu {desc}: {path.name}  [{out.shape[0]} rows × {out.shape[1]} cols]")
    return out


def print_section(title: str):
    bar = "═" * 60
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)


def step1_verify_and_export(df: pd.DataFrame) -> pd.DataFrame:
    print_section("BƯỚC 1: Xác nhận cấu trúc & xuất step1_otu_with_labels.csv")

    meta_df, genus_df = split_meta_genus(df)

    assert "DiseaseState" in meta_df.columns, "Thiếu cột DiseaseState!"
    assert genus_df.shape[1] > 0, "Không có cột genus!"
    n_samples = len(df)
    n_genera  = genus_df.shape[1]

    log.info(f"  Samples      : {n_samples}")
    log.info(f"  Genera       : {n_genera}")
    log.info(f"  DiseaseState : {meta_df['DiseaseState'].value_counts().to_dict()}")
    log.info(f"  Datasets     : {meta_df['Dataset'].value_counts().to_dict()}")

    sparsity = (genus_df == 0).sum().sum() / genus_df.size * 100
    log.info(f"  Sparsity (% zeros) TRƯỚC lọc: {sparsity:.1f}%")

    out_path = OUTPUT_DIR / "step1_otu_with_labels.csv"
    save_with_meta(genus_df, meta_df, out_path, "step1_otu_with_labels")

    return df


def step2_filter_noise(df: pd.DataFrame) -> pd.DataFrame:
    print_section("BƯỚC 2: Lọc nhiễu sinh học → step2_filtered_otu.csv")

    meta_df, genus_df = split_meta_genus(df)
    n_samples = len(genus_df)

    log.info(f"  Số genera TRƯỚC lọc : {genus_df.shape[1]}")
    log.info(f"  Ngưỡng prevalence    : ≥ {PREVALENCE_THRESHOLD*100:.0f}% samples")

    prevalence   = (genus_df > MIN_COUNT_THRESHOLD).mean(axis=0)
    keep_prev    = prevalence[prevalence >= PREVALENCE_THRESHOLD].index
    removed_prev = prevalence[prevalence <  PREVALENCE_THRESHOLD].index

    log.info(f"  Genera bị loại (prevalence <10%): {len(removed_prev)}")
    if len(removed_prev) > 0:
        log.info(f"    → {sorted(removed_prev.tolist())}")

    genus_filtered = genus_df[keep_prev]
    all_zero_cols  = genus_filtered.columns[genus_filtered.sum() == 0]
    if len(all_zero_cols) > 0:
        genus_filtered = genus_filtered.drop(columns=all_zero_cols)
        log.info(f"  Genus toàn số 0 bị xóa thêm: {list(all_zero_cols)}")

    log.info(f"  Số genera SAU lọc   : {genus_filtered.shape[1]}")

    sparsity_after = (genus_filtered == 0).sum().sum() / genus_filtered.size * 100
    log.info(f"  Sparsity SAU lọc    : {sparsity_after:.1f}%")

    prev_after = (genus_filtered > 0).mean()
    log.info(f"  Prevalence min/mean/max: "
             f"{prev_after.min():.3f} / {prev_after.mean():.3f} / {prev_after.max():.3f}")

    out_path = OUTPUT_DIR / "step2_filtered_otu.csv"
    out_df   = save_with_meta(genus_filtered, meta_df, out_path, "step2_filtered_otu")

    return out_df


def step3a_tss_normalize(df: pd.DataFrame) -> pd.DataFrame:
    print_section("BƯỚC 3a: Chuẩn hóa TSS → step3a_tss_normalized.csv")

    meta_df, genus_df = split_meta_genus(df)

    row_totals = genus_df.sum(axis=1)

    zero_samples = row_totals[row_totals == 0]
    if len(zero_samples) > 0:
        log.warning(f"  {len(zero_samples)} mẫu có tổng đếm = 0, "
                    f"sẽ điền NaN → thay bằng 0: {zero_samples.index.tolist()}")

    tss_df = genus_df.div(row_totals, axis=0)
    tss_df = tss_df.fillna(0)

    row_sums = tss_df.sum(axis=1)
    valid_rows = row_totals > 0
    max_deviation = (row_sums[valid_rows] - 1.0).abs().max()
    log.info(f"  Kiểm tra TSS: max deviation từ 1.0 = {max_deviation:.2e}  ✓")
    log.info(f"  Giá trị min  : {tss_df.values.min():.6f}")
    log.info(f"  Giá trị max  : {tss_df.values.max():.6f}")
    log.info(f"  Giá trị mean : {tss_df.values[tss_df.values > 0].mean():.6f}")

    mean_abund = tss_df.mean().sort_values(ascending=False)
    log.info(f"  Top 5 genus phong phú nhất (mean relative abundance):")
    for g, v in mean_abund.head(5).items():
        log.info(f"    {g:<30s}: {v:.4f} ({v*100:.2f}%)")

    out_path = OUTPUT_DIR / "step3a_tss_normalized.csv"
    out_df   = save_with_meta(tss_df, meta_df, out_path, "step3a_tss_normalized")

    return out_df


def step3b_clr_normalize(df: pd.DataFrame) -> pd.DataFrame:
    print_section("BƯỚC 3b: Chuẩn hóa CLR → step3b_clr_normalized.csv")

    meta_df, genus_df = split_meta_genus(df)

    log.info(f"  Pseudocount được thêm vào: {CLR_PSEUDOCOUNT}")
    log.info(f"  (Tránh log(0) cho các ô bằng 0)")

    X = genus_df.values.astype(float) + CLR_PSEUDOCOUNT

    log_X = np.log(X)

    gm_log = log_X.mean(axis=1, keepdims=True)

    clr_X = log_X - gm_log

    clr_df = pd.DataFrame(clr_X,
                          index=genus_df.index,
                          columns=genus_df.columns)

    row_sums = clr_df.sum(axis=1)
    max_dev  = row_sums.abs().max()
    log.info(f"  Kiểm tra CLR: max |row_sum| = {max_dev:.2e}  "
             f"{'✓ (≈0, đúng tính chất CLR)' if max_dev < 1e-8 else '⚠ kiểm tra lại'}")

    log.info(f"  Giá trị min  : {clr_df.values.min():.4f}")
    log.info(f"  Giá trị max  : {clr_df.values.max():.4f}")
    log.info(f"  Giá trị mean : {clr_df.values.mean():.6f}  (≈ 0 theo định nghĩa CLR)")
    log.info(f"  Std overall  : {clr_df.values.std():.4f}")

    temp = pd.concat([meta_df[["DiseaseState"]], clr_df], axis=1)
    log.info("  CLR mean theo DiseaseState (5 genera đầu):")
    for ds in ["H", "OB", "OW"]:
        subset = temp[temp["DiseaseState"] == ds][genus_df.columns[:5]]
        log.info(f"    {ds}: {subset.mean().round(3).to_dict()}")

    out_path = OUTPUT_DIR / "step3b_clr_normalized.csv"
    out_df   = save_with_meta(clr_df, meta_df, out_path, "step3b_clr_normalized")

    return out_df


def build_taxonomy_annotation(df: pd.DataFrame) -> pd.DataFrame:
    print_section("TAXONOMY ANNOTATION → taxonomy_annotation.csv")

    meta_df, genus_df = split_meta_genus(df)
    genus_list = genus_df.columns.tolist()

    core_genera_path = BASE_DIR / "file-S3_core_genera.txt"

    lineage_map    = {}
    health_map     = {}

    if core_genera_path.exists():
        with open(core_genera_path) as fh:
            next(fh)
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                taxon = parts[0].strip()
                label = parts[1].strip() if len(parts) > 1 else ""
                if not taxon:
                    continue
                g_parts = [p for p in taxon.split(";") if p.startswith("g__")]
                if not g_parts:
                    continue
                genus_name = g_parts[-1].replace("g__", "").strip()
                if genus_name:
                    lineage_map[genus_name] = taxon
                    health_map[genus_name]  = label
        log.info(f"  Đọc được taxonomy lineage cho {len(lineage_map)} genera")
    else:
        log.warning(f"  Không tìm thấy {core_genera_path}, bỏ qua taxonomy lineage")

    rows = []
    for genus in sorted(genus_list):
        lineage = lineage_map.get(genus, "")

        tax_levels = {
            "Kingdom" : "",
            "Phylum"  : "",
            "Class"   : "",
            "Order"   : "",
            "Family"  : "",
            "Genus"   : genus,
        }
        prefix_map = {
            "k__": "Kingdom",
            "p__": "Phylum",
            "c__": "Class",
            "o__": "Order",
            "f__": "Family",
            "g__": "Genus",
        }
        if lineage:
            for part in lineage.split(";"):
                part = part.strip()
                for pfx, level in prefix_map.items():
                    if part.startswith(pfx):
                        tax_levels[level] = part[len(pfx):].strip()

        col_data    = genus_df[genus]
        prevalence  = (col_data > 0).mean()
        mean_count  = col_data.mean()
        median_count= col_data.median()
        max_count   = col_data.max()
        std_count   = col_data.std()
        nonzero_mean= col_data[col_data > 0].mean() if (col_data > 0).any() else 0.0

        h_mean  = genus_df.loc[meta_df["DiseaseState"] == "H",  genus].mean()
        ob_mean = genus_df.loc[meta_df["DiseaseState"] == "OB", genus].mean()
        ow_mean = genus_df.loc[meta_df["DiseaseState"] == "OW", genus].mean()

        rows.append({
            "Genus"              : genus,
            "Kingdom"            : tax_levels["Kingdom"],
            "Phylum"             : tax_levels["Phylum"],
            "Class"              : tax_levels["Class"],
            "Order"              : tax_levels["Order"],
            "Family"             : tax_levels["Family"],
            "Health_Association" : health_map.get(genus, "unclassified"),
            "Full_Lineage"       : lineage,
            "Prevalence"         : round(prevalence, 4),
            "Mean_Count"         : round(mean_count, 2),
            "Median_Count"       : round(median_count, 2),
            "Max_Count"          : int(max_count),
            "Std_Count"          : round(std_count, 2),
            "Mean_Count_NonZero" : round(nonzero_mean, 2),
            "Mean_Count_H"       : round(h_mean, 2),
            "Mean_Count_OB"      : round(ob_mean, 2),
            "Mean_Count_OW"      : round(ow_mean, 2),
        })

    annot_df = pd.DataFrame(rows).set_index("Genus")

    annot_df = annot_df.sort_values(["Phylum", "Genus"])

    log.info(f"  Tổng genera được annotate: {len(annot_df)}")
    log.info("  Phân phối theo Phylum:")
    for phylum, cnt in annot_df["Phylum"].value_counts().items():
        log.info(f"    {phylum:<40s}: {cnt} genera")
    log.info("  Phân phối Health_Association:")
    for label, cnt in annot_df["Health_Association"].value_counts().items():
        log.info(f"    {str(label):<20s}: {cnt} genera")

    out_path = OUTPUT_DIR / "taxonomy_annotation.csv"
    annot_df.to_csv(out_path)
    log.info(f"✓ Đã lưu taxonomy_annotation: {out_path.name}  [{annot_df.shape[0]} genera]")

    return annot_df


def main():
    print_section("GIAI ĐOẠN 0 – DATA ENGINEERING PIPELINE")
    log.info(f"  Input  : {INPUT_CSV}")
    log.info(f"  Output : {OUTPUT_DIR}")

    log.info("Đọc merged_clean_microbiome.csv ...")
    raw_df = pd.read_csv(INPUT_CSV, index_col=0)
    log.info(f"  Shape đọc vào: {raw_df.shape}")

    step1_df = step1_verify_and_export(raw_df)

    step2_df = step2_filter_noise(step1_df)

    step3a_df = step3a_tss_normalize(step2_df)

    step3b_df = step3b_clr_normalize(step2_df)

    annot_df = build_taxonomy_annotation(step2_df)

    print_section("TÓM TẮT ĐẦU RA")

    _, g1 = split_meta_genus(step1_df)
    _, g2 = split_meta_genus(step2_df)

    summary = {
        "step1_otu_with_labels.csv"  : f"{len(step1_df)} samples × {g1.shape[1]} genera  [raw counts + labels]",
        "step2_filtered_otu.csv"     : f"{len(step2_df)} samples × {g2.shape[1]} genera  [sau lọc prevalence ≥10%]",
        "step3a_tss_normalized.csv"  : f"{len(step3a_df)} samples × {g2.shape[1]} genera  [tần suất tương đối 0–1]",
        "step3b_clr_normalized.csv"  : f"{len(step3b_df)} samples × {g2.shape[1]} genera  [CLR-transformed, dùng cho ML]",
        "taxonomy_annotation.csv"    : f"{len(annot_df)} genera × {annot_df.shape[1]} annotation fields",
    }

    log.info("")
    for fname, desc in summary.items():
        log.info(f"  {fname:<38s} → {desc}")

    log.info("")
    log.info("  File khuyến nghị dùng để huấn luyện ML: step3b_clr_normalized.csv")
    log.info("  File khuyến nghị dùng cho phân tích trực quan: step3a_tss_normalized.csv")
    log.info("")
    print_section("HOÀN TẤT GIAI ĐOẠN 0")


if __name__ == "__main__":
    main()