"""
=============================================================
Microbiome Data Processing Pipeline
=============================================================
Mục tiêu: Đọc 5 file .tar.gz, chuẩn hóa về cấp độ Genus,
          lọc bỏ non-core genera, ghép thành 1 file CSV duy nhất.
=============================================================
"""

import os
import io
import tarfile
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Đường dẫn ────────────────────────────────────────────────────────────────
# Tất cả file (.tar.gz, core_genera.txt, script) đặt cùng 1 thư mục.
# Script tự động tìm thư mục chứa chính nó → không cần sửa đường dẫn thủ công.
BASE_DIR      = Path(__file__).parent.resolve()
UPLOADS_DIR   = BASE_DIR          # các file .tar.gz nằm cùng thư mục với script
OUTPUT_DIR    = BASE_DIR          # output CSV cũng xuất ra cùng thư mục
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_GENERA_FILE = UPLOADS_DIR / "file-S3.core_genera.txt"
OUTPUT_CSV       = OUTPUT_DIR  / "merged_clean_microbiome.csv"

# ── Danh sách 5 dataset ───────────────────────────────────────────────────────
# Mỗi entry: (tên_file.tar.gz, dataset_id, cách lấy DiseaseState)
DATASETS = [
    {
        "file"   : "ob_gordon_2008_v2_results.tar.gz",
        "name"   : "ob_gordon",
        "meta_col": "DiseaseState",          # cột trong metadata
    },
    {
        "file"   : "ob_ross_results.tar.gz",
        "name"   : "ob_ross",
        "meta_col": "DiseaseState",
    },
    {
        "file"   : "ob_zupancic_results.tar.gz",
        "name"   : "ob_zupancic",
        "meta_col": "DiseaseState",
    },
    {
        "file"   : "ob_goodrich_results.tar.gz",
        "name"   : "ob_goodrich",
        "meta_col": "DiseaseState",
        # chỉ lấy n_sample == 0 (baseline / lần thu thập đầu tiên)
        "filter" : ("n_sample", 0),
    },
    {
        "file"   : "ob_escobar_results.tar.gz",
        "name"   : "ob_escobar",
        # Escobar không có metadata.txt → dùng quy ước tên mẫu:
        #   A* = H (Healthy),  B* = OW (Overweight),  C* = OB (Obese)
        "meta_col": None,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 0 – Đọc danh sách Core Genera
# ─────────────────────────────────────────────────────────────────────────────
def load_core_genera(path: Path) -> set:
    """
    Trả về tập hợp các tên genus (chuỗi sau 'g__') có trong file core_genera.
    """
    genera = set()
    with open(path) as fh:
        next(fh)                                  # bỏ dòng header
        for line in fh:
            taxon = line.split("\t")[0].strip()
            if not taxon:
                continue
            # lấy phần genus: phần tử cuối cùng chứa 'g__'
            parts = [p for p in taxon.split(";") if p.startswith("g__")]
            if parts:
                genus_name = parts[-1].replace("g__", "").strip()
                if genus_name:
                    genera.add(genus_name)
    log.info(f"Core genera loaded: {len(genera)} genera")
    return genera


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1+2 – Mở tar.gz, rút trích OTU table và metadata vào RAM
# ─────────────────────────────────────────────────────────────────────────────
def read_tar_contents(tar_path: Path):
    """
    Trả về (otu_df, meta_df) - cả hai là pandas DataFrame, đọc thẳng vào RAM.
    otu_df : index = full taxonomy string, columns = sample IDs, values = counts
    meta_df: index = sample IDs, columns = metadata fields (None nếu không có)
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}

        # --- OTU table (rdp_assigned) ---
        otu_member = next(
            (m for name, m in members.items() if "rdp_assigned" in name), None
        )
        if otu_member is None:
            raise FileNotFoundError(f"Không tìm thấy OTU table trong {tar_path}")

        raw_otu = tar.extractfile(otu_member).read().decode("utf-8", errors="replace")
        otu_df  = pd.read_csv(io.StringIO(raw_otu), sep="\t", index_col=0)

        # --- Metadata (nếu có) ---
        meta_member = next(
            (m for name, m in members.items() if "metadata.txt" in name), None
        )
        meta_df = None
        if meta_member:
            raw_meta = tar.extractfile(meta_member).read().decode("utf-8", errors="replace")
            meta_df  = pd.read_csv(io.StringIO(raw_meta), sep="\t", index_col=0)

    return otu_df, meta_df


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3 – Chuẩn hóa cục bộ: OTU → Genus, gán nhãn DiseaseState
# ─────────────────────────────────────────────────────────────────────────────
def extract_genus(taxonomy_str: str) -> str:
    """
    'k__Bacteria;...;g__Faecalibacterium;s__;d__denovo7709'  →  'Faecalibacterium'
    Trả về '' nếu không tìm thấy genus hợp lệ.
    """
    for part in taxonomy_str.split(";"):
        part = part.strip()
        if part.startswith("g__"):
            name = part[3:].strip()
            # bỏ qua trường rỗng hoặc chỉ là phần suffix denovo
            if name and not name.startswith("d__"):
                return name
    return ""


def escobar_disease_state(sample_id: str) -> str:
    """
    Escobar: A* → H,  B* → OW,  C* → OB
    """
    prefix = sample_id[0].upper()
    mapping = {"A": "H", "B": "OW", "C": "OB"}
    return mapping.get(prefix, "Unknown")


def process_dataset(ds_cfg: dict, core_genera: set) -> pd.DataFrame:
    """
    Xử lý một dataset theo pipeline 5 bước, trả về DataFrame đã chuẩn hóa.
    Rows = samples, Columns = genus names + DiseaseState + Dataset.
    Chỉ giữ lại các genus có trong core_genera.
    """
    ds_name = ds_cfg["name"]
    tar_path = UPLOADS_DIR / ds_cfg["file"]
    log.info(f"[{ds_name}] Đang đọc {ds_cfg['file']} ...")

    # ── Bước 1+2: Rút trích vào RAM ──────────────────────────────────────────
    otu_df, meta_df = read_tar_contents(tar_path)
    log.info(f"[{ds_name}] OTU table: {otu_df.shape[0]} OTUs × {otu_df.shape[1]} samples")

    # ── Bước 3a: Chuẩn hóa taxonomy → Genus ─────────────────────────────────
    otu_df.index = otu_df.index.map(extract_genus)

    # Loại bỏ các OTU không xác định được genus
    otu_df = otu_df[otu_df.index != ""]
    otu_df = otu_df[~otu_df.index.isna()]

    # Gom nhóm (sum) các OTU cùng genus
    otu_df = otu_df.groupby(level=0).sum()
    log.info(f"[{ds_name}] Sau gom nhóm genus: {otu_df.shape[0]} genera")

    # ── Bước 3b: Chuyển vị – samples thành rows ──────────────────────────────
    sample_df = otu_df.T                          # rows=samples, cols=genera
    sample_df.index.name = "SampleID"

    # ── Bước 3c: Lọc chỉ giữ Core Genera ─────────────────────────────────────
    keep_cols = [c for c in sample_df.columns if c in core_genera]
    missing   = len(core_genera) - len(keep_cols)
    sample_df = sample_df[keep_cols]
    log.info(
        f"[{ds_name}] Core genera hiện diện: {len(keep_cols)}/{len(core_genera)} "
        f"({missing} không có trong dataset)"
    )

    # ── Bước 3d: Lọc sample theo điều kiện riêng (nếu có) ────────────────────
    if ds_cfg.get("filter") and meta_df is not None:
        fcol, fval = ds_cfg["filter"]
        if fcol in meta_df.columns:
            valid_samples = meta_df[meta_df[fcol] == fval].index
            sample_df = sample_df[sample_df.index.isin(valid_samples)]
            log.info(
                f"[{ds_name}] Sau lọc '{fcol}=={fval}': {len(sample_df)} samples"
            )

    # ── Bước 3e: Gán nhãn DiseaseState ───────────────────────────────────────
    if ds_cfg["meta_col"] is None:
        # Escobar: suy ra từ tên mẫu
        sample_df["DiseaseState"] = sample_df.index.map(escobar_disease_state)
    elif meta_df is not None and ds_cfg["meta_col"] in meta_df.columns:
        # Align index: lấy theo mẫu đang có
        disease_map = meta_df[ds_cfg["meta_col"]]
        sample_df["DiseaseState"] = sample_df.index.map(disease_map)
    else:
        sample_df["DiseaseState"] = np.nan
        log.warning(f"[{ds_name}] Không tìm thấy cột DiseaseState!")

    # Loại bỏ mẫu không có nhãn bệnh
    before = len(sample_df)
    sample_df = sample_df.dropna(subset=["DiseaseState"])
    after  = len(sample_df)
    if before != after:
        log.info(f"[{ds_name}] Bỏ {before - after} mẫu không có DiseaseState")

    # ── Bước 3f: Đánh dấu nguồn gốc dataset ─────────────────────────────────
    sample_df["Dataset"] = ds_name

    log.info(
        f"[{ds_name}] ✓ {len(sample_df)} samples | "
        f"DiseaseState: {sample_df['DiseaseState'].value_counts().to_dict()}"
    )
    return sample_df


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4+5 – Nối và xuất
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("MICROBIOME MERGE PIPELINE – BẮT ĐẦU")
    log.info("=" * 60)

    # Bước 0: Đọc core genera
    core_genera = load_core_genera(CORE_GENERA_FILE)

    # Bước 1-3: Xử lý từng dataset
    frames = []
    for ds_cfg in DATASETS:
        df = process_dataset(ds_cfg, core_genera)
        frames.append(df)

    # Bước 4: Nối tất cả lại (pd.concat tự động căn cột, NaN nếu không có)
    log.info("Đang ghép nối tất cả datasets ...")
    merged = pd.concat(frames, axis=0, sort=True)
    merged.index.name = "SampleID"

    # Đặt thứ tự cột: metadata trước, rồi genus theo alphabet
    meta_cols  = ["Dataset", "DiseaseState"]
    genus_cols = sorted([c for c in merged.columns if c not in meta_cols])
    merged     = merged[meta_cols + genus_cols]

    # Điền NaN (genus không có trong một dataset) = 0
    merged[genus_cols] = merged[genus_cols].fillna(0).astype(int)

    log.info(f"Super DataFrame: {merged.shape[0]} samples × {merged.shape[1]} cột")
    log.info(f"  - Genus columns : {len(genus_cols)}")
    log.info(f"  - DiseaseState  : {merged['DiseaseState'].value_counts().to_dict()}")
    log.info(f"  - Dataset       : {merged['Dataset'].value_counts().to_dict()}")

    # Bước 5: Xuất ra file CSV duy nhất
    merged.to_csv(OUTPUT_CSV)
    log.info(f"✓ Đã lưu: {OUTPUT_CSV}")
    log.info("=" * 60)
    log.info("PIPELINE HOÀN TẤT")
    log.info("=" * 60)

    # In thống kê tóm tắt
    print("\n" + "="*60)
    print("THỐNG KÊ TỔNG HỢP")
    print("="*60)
    print(f"Tổng số samples  : {len(merged)}")
    print(f"Tổng số features : {len(genus_cols)} genus")
    print(f"\nPhân phối nhãn (DiseaseState):")
    print(merged["DiseaseState"].value_counts().to_string())
    print(f"\nPhân phối theo Dataset:")
    print(merged.groupby(["Dataset", "DiseaseState"]).size().to_string())
    print(f"\nOutput: {OUTPUT_CSV}")
    return merged


if __name__ == "__main__":
    merged_df = main()