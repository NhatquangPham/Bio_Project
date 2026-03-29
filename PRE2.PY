import pandas as pd
import numpy as np

print("1. Đang tải merged_clean_microbiome.csv...")
df_step1 = pd.read_csv('merged_clean_microbiome.csv', index_col=0)

metadata_cols = ['Dataset_Source', 'DiseaseState', 'Target']
existing_meta_cols = [col for col in metadata_cols if col in df_step1.columns]
df_meta = df_step1[existing_meta_cols]

df_otu = df_step1.drop(columns=existing_meta_cols)
df_otu = df_otu.apply(pd.to_numeric, errors='coerce').fillna(0)

print("2. Đang tiến hành lọc vi khuẩn thưa thớt (dưới 10% số mẫu)...")
total_samples = len(df_otu)
threshold = 0.10 * total_samples

prevalence = (df_otu > 0).sum(axis=0)

cols_to_keep = prevalence[prevalence >= threshold].index
df_otu_filtered = df_otu[cols_to_keep]

print(f" -> Ban đầu có: {df_otu.shape[1]} vi khuẩn")
print(f" -> Sau khi lọc giữ lại: {df_otu_filtered.shape[1]} vi khuẩn phổ biến")

step2_df = pd.concat([df_meta, df_otu_filtered], axis=1)
step2_df.to_csv('step2_filtered_otu.csv')
print(" -> [HOÀN TẤT] Đã lưu file: step2_filtered_otu.csv")

print("3. Đang chuẩn hóa TSS (Relative Abundance)...")
row_sums = df_otu_filtered.sum(axis=1)
df_tss = df_otu_filtered.div(row_sums, axis=0)
df_tss = df_tss.fillna(0)

step3a_df = pd.concat([df_meta, df_tss], axis=1)
step3a_df.to_csv('step3a_tss_normalized.csv')
print(" -> [HOÀN TẤT] Đã lưu file: step3a_tss_normalized.csv")

print("4. Đang chuẩn hóa CLR (Dành cho AI/Machine Learning)...")
df_pseudo = df_otu_filtered + 1

geom_mean = np.exp(np.mean(np.log(df_pseudo), axis=1))

df_clr = np.log(df_pseudo.div(geom_mean, axis=0))

step3b_df = pd.concat([df_meta, df_clr], axis=1)
step3b_df.to_csv('step3b_clr_normalized.csv')
print(" -> [HOÀN TẤT] Đã lưu file: step3b_clr_normalized.csv")

print("\n🎉 CHÚC MỪNG! BẠN ĐÃ HOÀN THÀNH 100% GIAI ĐOẠN 0.")