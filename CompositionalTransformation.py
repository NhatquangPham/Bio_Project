import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# BƯỚC 1: ĐỌC VÀ KIỂM TRA DỮ LIỆU
# ==========================================
print("1. Đang tải dữ liệu microbiome...")
df = pd.read_csv('merged_clean_microbiome.csv', index_col=0)

# In ra các nhãn bệnh lý hiện có để kiểm tra xem có bị lộn xộn giữa các dataset không
print("Các nhãn tình trạng bệnh hiện có:", df['DiseaseState'].unique())

# Giả sử chúng ta gom nhóm: 'H' (Khỏe mạnh) -> 0, 'OB' (Béo phì) -> 1
# Lưu ý: Bạn có thể cần điều chỉnh dictionary này dựa trên kết quả in ra ở trên
label_map = {'H': 0, 'OB': 1, 'Control': 0, 'Obese': 1} 
df['Target'] = df['DiseaseState'].map(label_map)

# Loại bỏ những hàng không có nhãn bệnh lý (nếu có)
df = df.dropna(subset=['Target'])

# ==========================================
# BƯỚC 2: TÁCH MA TRẬN VI KHUẨN VÀ BIẾN ĐỔI CLR
# ==========================================
print("2. Đang thực hiện biến đổi CLR...")

# Tách riêng phần dữ liệu đếm vi khuẩn (bỏ qua các cột thông tin bệnh án)
metadata_cols = ['Dataset', 'DiseaseState', 'Target']
X_raw = df.drop(columns=metadata_cols, errors='ignore')

# Chuyển đổi tất cả sang dạng số (đề phòng có lỗi ký tự)
X_raw = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

# Thêm Pseudo-count = 1 vào toàn bộ ma trận để tránh lỗi Toán học (Log của 0 không xác định)
X_pseudo = X_raw + 1

# Hàm tính Biến đổi CLR (Centered Log-Ratio) - Khử sai số do máy giải trình tự
def clr_transform(matrix):
    # Tính trung bình nhân (Geometric mean) cho từng mẫu (từng hàng)
    geom_mean = np.exp(np.mean(np.log(matrix), axis=1))
    # Lấy giá trị vi khuẩn chia cho trung bình nhân, sau đó tính Log
    return np.log(matrix.div(geom_mean, axis=0))

X_clr = clr_transform(X_pseudo)
print(" -> Chuyển đổi CLR thành công! Kích thước ma trận:", X_clr.shape)

# ==========================================
# BƯỚC 3: PHÂN TÍCH THÀNH PHẦN CHÍNH (PCA)
# ==========================================
print("3. Đang vẽ biểu đồ PCA để kiểm tra sự phân cụm...")

# Giảm từ hàng ngàn chiều (vi khuẩn) xuống còn 2 chiều (2 Components) để vẽ trục tọa độ Oxyz
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clr)

# Trực quan hóa bằng biểu đồ phân tán (Scatter Plot)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_pca[:, 0], 
    y=X_pca[:, 1], 
    hue=df['DiseaseState'],  # Tô màu theo tình trạng bệnh
    style=df['Dataset'],     # Hình dáng điểm (vuông/tròn) theo nguồn dataset
    palette='Set1', 
    s=100, 
    alpha=0.8
)

# Hiển thị tỷ lệ biến thiên mà mỗi trục giải thích được
plt.title('Biểu đồ PCA: Sự khác biệt Hệ vi sinh vật (Béo phì vs Khỏe mạnh)', fontsize=14, fontweight='bold')
plt.xlabel(f'Thành phần chính 1 (PC1) - {pca.explained_variance_ratio_[0]:.2%} độ lệch')
plt.ylabel(f'Thành phần chính 2 (PC2) - {pca.explained_variance_ratio_[1]:.2%} độ lệch')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)

# Lưu biểu đồ ra file
plt.savefig('pca_microbiome_plot.png', dpi=300)
print("Hoàn tất! Biểu đồ đã được lưu thành 'pca_microbiome_plot.png'.")