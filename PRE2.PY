import pandas as pd
import numpy as np

# 1. TẢI DỮ LIỆU BƯỚC 1 (Đã chuyển vị và ghép nhãn)
print("1. Đang tải merged_clean_microbiome.csv...")
df_step1 = pd.read_csv('merged_clean_microbiome.csv', index_col=0)

# Tách riêng phần Metadata (thông tin bệnh án) và Feature (đếm vi khuẩn)
metadata_cols = ['Dataset_Source', 'DiseaseState', 'Target']
# Lọc lấy các cột metadata có tồn tại trong df
existing_meta_cols = [col for col in metadata_cols if col in df_step1.columns]
df_meta = df_step1[existing_meta_cols]

# Ma trận đếm vi khuẩn (X)
df_otu = df_step1.drop(columns=existing_meta_cols)
# Chuyển đổi an toàn sang số học
df_otu = df_otu.apply(pd.to_numeric, errors='coerce').fillna(0)

# =========================================================
# BƯỚC 2: LỌC NHIỄU ĐỘ THƯA THỚT (SPARSITY REDUCTION)
# =========================================================
print("2. Đang tiến hành lọc vi khuẩn thưa thớt (dưới 10% số mẫu)...")
total_samples = len(df_otu)
threshold = 0.10 * total_samples # Ngưỡng 10%

# Đếm số lượng mẫu (hàng) mà vi khuẩn đó > 0
prevalence = (df_otu > 0).sum(axis=0)

# Giữ lại các cột (vi khuẩn) vượt qua ngưỡng
cols_to_keep = prevalence[prevalence >= threshold].index
df_otu_filtered = df_otu[cols_to_keep]

print(f" -> Ban đầu có: {df_otu.shape[1]} vi khuẩn")
print(f" -> Sau khi lọc giữ lại: {df_otu_filtered.shape[1]} vi khuẩn phổ biến")

# Ghép metadata lại và lưu file step 2
step2_df = pd.concat([df_meta, df_otu_filtered], axis=1)
step2_df.to_csv('step2_filtered_otu.csv')
print(" -> [HOÀN TẤT] Đã lưu file: step2_filtered_otu.csv")

# =========================================================
# BƯỚC 3A: CHUẨN HÓA TSS (TẦN SUẤT TƯƠNG ĐỐI 0-1)
# =========================================================
print("3. Đang chuẩn hóa TSS (Relative Abundance)...")
# Lấy số đếm của từng ô chia cho tổng số đếm của hàng đó
row_sums = df_otu_filtered.sum(axis=1)
df_tss = df_otu_filtered.div(row_sums, axis=0)
# Điền 0 nếu chia cho 0 (hàng không có vi khuẩn nào)
df_tss = df_tss.fillna(0)

step3a_df = pd.concat([df_meta, df_tss], axis=1)
step3a_df.to_csv('step3a_tss_normalized.csv')
print(" -> [HOÀN TẤT] Đã lưu file: step3a_tss_normalized.csv")

# =========================================================
# BƯỚC 3B: CHUẨN HÓA CLR (CENTERED LOG-RATIO CHO ML)
# =========================================================
print("4. Đang chuẩn hóa CLR (Dành cho AI/Machine Learning)...")
# CLR yêu cầu không được có số 0 (vì log(0) = vô cực). Cần cộng Pseudo-count = 1
df_pseudo = df_otu_filtered + 1

# Tính giá trị trung bình nhân (Geometric mean) của mỗi hàng
geom_mean = np.exp(np.mean(np.log(df_pseudo), axis=1))

# Phép biến đổi: log( Giá_trị / Trung_bình_nhân )
df_clr = np.log(df_pseudo.div(geom_mean, axis=0))

step3b_df = pd.concat([df_meta, df_clr], axis=1)
step3b_df.to_csv('step3b_clr_normalized.csv')
print(" -> [HOÀN TẤT] Đã lưu file: step3b_clr_normalized.csv")

print("\n🎉 CHÚC MỪNG! BẠN ĐÃ HOÀN THÀNH 100% GIAI ĐOẠN 0.")