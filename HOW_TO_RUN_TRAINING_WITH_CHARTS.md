# 📊 HƯỚNG DẪN CHẠY TRAINING VỚI BIỂU ĐỒ

## 🎯 Tổng quan

Có 2 models chính cần train với **2 phiên bản visualization**:

1. **p_freelancer_accept** - Dự đoán freelancer có chấp nhận lời mời không
2. **p_match** - Dự đoán cặp job-freelancer có thành công không

---

## 🚀 CÁCH CHẠY (KHUYẾN NGHỊ - BIỂU ĐỒ RIÊNG BIỆT)

### ⭐ Option A: Separate Charts (Không bị đè chữ)

#### 1. Training p_freelancer_accept:

```bash
python -m app.workers.train_p_freelancer_accept_separate_charts
```

**Output:**
- ✅ Classification report trên terminal
- ✅ **4 files ảnh riêng biệt** trong folder `separate_charts/`:
  - `01_confusion_matrix.png` - Ma trận nhầm lẫn
  - `02_performance_metrics.png` - Các chỉ số đánh giá
  - `03_feature_importance.png` - Tầm quan trọng features
  - `04_dataset_overview.png` - Tổng quan dataset
- ✅ Model file: `models/logreg_p_freelancer_accept.pkl`

#### 2. Training p_match:

```bash
python -m app.workers.train_p_match_separate_charts
```

**Output:**
- ✅ Classification report trên terminal
- ✅ **4 files ảnh riêng biệt** trong folder `p_match_separate_charts/`:
  - `01_confusion_matrix.png` - Ma trận nhầm lẫn
  - `02_performance_metrics.png` - Các chỉ số đánh giá (bao gồm AUC)
  - `03_feature_importance.png` - Tầm quan trọng features
  - `04_dataset_overview.png` - Tổng quan dataset
- ✅ Model file: `app/models/p_match_logreg.joblib`

### 🔄 Option B: Combined Chart (Có thể bị đè chữ)

#### 1. Training p_freelancer_accept:

```bash
python -m app.workers.train_p_freelancer_accept_visual
```

**Output:**
- ✅ Classification report trên terminal
- ✅ File ảnh: `visualization_results/p_freelancer_accept_training_results.png`
- ✅ Model file: `models/logreg_p_freelancer_accept.pkl`

#### 2. Training p_match:

```bash
python -m app.workers.train_p_match_visual
```

**Output:**
- ✅ Classification report trên terminal  
- ✅ File ảnh: `visualization_results/p_match_training_results.png`
- ✅ Model file: `models/p_match_logreg.pkl`

---

## 📊 BIỂU ĐỒ ĐƯỢC TẠO

Mỗi file ảnh bao gồm **4 phần chính**:

### 1. 📊 Confusion Matrix
- Ma trận nhầm lẫn với số lượng dự đoán đúng/sai
- **Chữ to, rõ ràng** cho presentation
- Accuracy score được highlight

### 2. 📈 Performance Metrics  
- 4 chỉ số: Accuracy, Precision, Recall, F1-Score
- **Biểu đồ cột màu sắc** dễ nhìn
- **Giá trị số to** trên mỗi cột

### 3. 📋 Dataset Information
- Thông tin tổng quan về dữ liệu
- Số lượng samples, phân bố labels
- Cấu hình model

### 4. 🔍 Top Feature Importance
- **12 features quan trọng nhất**
- Thanh ngang với **chữ to**
- Màu xanh = tăng xác suất, đỏ = giảm xác suất
- **Không bị đè chữ**

---

## 🎨 ĐẶC ĐIỂM BIỂU ĐỒ

### ✅ Tối ưu cho Presentation:
- **Font size lớn** (14-28pt) - giảng viên dễ nhìn
- **Layout rộng rãi** - không bị đè chữ
- **Màu sắc chuyên nghiệp** - phù hợp luận văn
- **Độ phân giải cao** (300 DPI) - in ấn đẹp

### ✅ Thông tin đầy đủ:
- Tất cả metrics quan trọng
- Feature importance với direction (+/-)
- Dataset statistics
- Model configuration

### ✅ Tên file rõ ràng:
- `p_freelancer_accept_training_results.png`
- `p_match_training_results.png`

---

## 📁 CẤU TRÚC FILE OUTPUT

### Option A: Separate Charts (Khuyến nghị)
```
lvtn_ml/
├── separate_charts/                              ← p_freelancer_accept charts
│   ├── 01_confusion_matrix.png
│   ├── 02_performance_metrics.png
│   ├── 03_feature_importance.png
│   └── 04_dataset_overview.png
├── p_match_separate_charts/                      ← p_match charts
│   ├── 01_confusion_matrix.png
│   ├── 02_performance_metrics.png
│   ├── 03_feature_importance.png
│   └── 04_dataset_overview.png
├── models/
│   ├── logreg_p_freelancer_accept.pkl            ← Model 1
│   └── p_match_logreg.joblib                     ← Model 2
└── dataset_p_freelancer_accept.csv               ← Dataset CSV
```

### Option B: Combined Charts
```
lvtn_ml/
├── visualization_results/
│   ├── p_freelancer_accept_training_results.png  ← Biểu đồ model 1
│   └── p_match_training_results.png              ← Biểu đồ model 2
├── models/
│   ├── logreg_p_freelancer_accept.pkl            ← Model 1
│   └── p_match_logreg.pkl                        ← Model 2
└── dataset_p_freelancer_accept.csv               ← Dataset CSV
```

---

## 💡 TIPS SỬ DỤNG

### Cho Luận văn:
1. Chạy cả 2 commands
2. Copy 2 file PNG vào Word/PowerPoint
3. Resize theo nhu cầu (chất lượng vẫn sắc nét)
4. Có thể crop từng phần nếu cần

### Cho Presentation:
1. File PNG có thể dùng trực tiếp
2. Chữ đủ to để chiếu projector
3. Màu sắc rõ ràng trên màn hình

### Cho Development:
1. Chạy để kiểm tra model performance
2. Xem feature importance để hiểu model
3. So sánh kết quả giữa các lần train

---

## 🔧 TROUBLESHOOTING

### Lỗi "No module named matplotlib":
```bash
pip install matplotlib pandas
```

### Lỗi import train_p_match:
- Đảm bảo file `train_p_match.py` tồn tại
- Hoặc tạo file đó dựa trên `train_p_freelancer_accept.py`

### Lỗi "Dataset is empty":
- Kiểm tra database connection
- Đảm bảo có dữ liệu trong bảng job_invitation

### Biểu đồ không hiển thị:
- Trên server: Biểu đồ vẫn được save file PNG
- Trên Windows: Sẽ popup window hiển thị

---

## 📊 SAMPLE OUTPUT

### Terminal Output:
```
🚀 TRAINING p_freelancer_accept MODEL WITH VISUALIZATION
============================================================

📊 Building dataset...
✅ Dataset loaded: 398 samples

🤖 Training model...

📊 CLASSIFICATION REPORT:
==================================================
              precision    recall  f1-score   support

           0       0.98      0.87      0.93        71
           1       0.47      0.89      0.62         9

    accuracy                           0.88        80
   macro avg       0.73      0.88      0.77        80
weighted avg       0.93      0.88      0.89        80

🎨 Creating presentation-ready visualization...

📊 Visualization saved to: visualization_results/p_freelancer_accept_training_results.png
📁 Full path: C:\...\lvtn_ml\visualization_results\p_freelancer_accept_training_results.png

🔍 TOP 5 MOST IMPORTANT FEATURES:
----------------------------------------
1. Job Stats Offers                    ↑ Increases Accept Rate
2. Job Experience Level Num            ↑ Increases Accept Rate  
3. Skill Overlap Count                 ↑ Increases Accept Rate
4. Skill Overlap Ratio                 ↑ Increases Accept Rate
5. Similarity Score                    ↑ Increases Accept Rate

💾 Model saved to: models/logreg_p_freelancer_accept.pkl

✅ TRAINING COMPLETED SUCCESSFULLY!
📊 Check the visualization file for presentation-ready charts.
```

---

## 🎯 KẾT LUẬN

### 🌟 KHUYẾN NGHỊ: Separate Charts (Option A)
```bash
python -m app.workers.train_p_freelancer_accept_separate_charts
python -m app.workers.train_p_match_separate_charts
```

### 🔄 Backup: Combined Charts (Option B)
```bash
python -m app.workers.train_p_freelancer_accept_visual
python -m app.workers.train_p_match_visual
```

Bạn sẽ có:
- ✅ **2 models trained** sẵn sàng production
- ✅ **8 biểu đồ riêng biệt** (4 cho mỗi model) hoặc **2 biểu đồ tổng hợp**
- ✅ **Chữ to, rõ ràng** - giảng viên dễ nhìn
- ✅ **KHÔNG BỊ ĐÈ CHỮ** - layout chuyên nghiệp với separate charts
- ✅ **Tên file rõ ràng** - dễ quản lý
- ✅ **Linh hoạt sử dụng** - có thể dùng từng chart riêng hoặc tổng hợp

**Perfect cho defense luận văn!** 🎓

### 💡 Lời khuyên:
- **Dùng Option A** nếu cần charts rõ ràng, không đè chữ
- **Dùng Option B** nếu muốn overview nhanh trong 1 file
- **Có thể chạy cả 2** để có nhiều lựa chọn presentation