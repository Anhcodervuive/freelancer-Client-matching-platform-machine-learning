# 📊 HƯỚNG DẪN VISUALIZATION CHO TRAINING

## 🎯 Tổng quan

Có 3 cách để chạy training với visualization khác nhau:

1. **Original** - Chỉ text output (như cũ)
2. **Simple Viz** - 1 biểu đồ tổng hợp 
3. **Full Viz** - 7 biểu đồ chi tiết

---

## 🚀 Cách sử dụng

### 1. Cài đặt thư viện

```bash
pip install matplotlib seaborn pandas
```

### 2. Chạy training

#### Option A: Original (chỉ text)
```bash
python -m app.workers.train_p_freelancer_accept
```
**Output:** Chỉ classification report trên terminal

#### Option B: Simple Visualization (khuyến nghị)
```bash
python -m app.workers.train_with_simple_viz
```
**Output:** 
- Classification report trên terminal
- 1 file ảnh tổng hợp: `simple_viz/training_results.png`

#### Option C: Full Visualization (chi tiết)
```bash
python -m app.workers.train_p_freelancer_accept_with_viz
```
**Output:**
- Classification report trên terminal  
- 7 files ảnh chi tiết trong folder `training_visualizations/`

---

## 📊 Các biểu đồ được tạo

### Simple Viz (1 ảnh)
- **training_results.png**: Tổng hợp 4 biểu đồ chính
  - Confusion Matrix
  - Top 10 Feature Importance  
  - Performance Metrics
  - Summary Text

### Full Viz (7 ảnh)
1. **01_dataset_overview.png**: Tổng quan dataset
   - Label distribution (pie chart)
   - Status breakdown (bar chart)
   - Key features distribution
   - Dataset statistics

2. **02_feature_analysis.png**: Phân tích features
   - Box plots của 6 features quan trọng nhất
   - So sánh giữa Accepted vs Declined
   - Mean values cho mỗi group

3. **03_confusion_matrix.png**: Ma trận nhầm lẫn
   - Heatmap với số lượng và phần trăm
   - Metrics: Accuracy, Precision, Recall, F1

4. **04_performance_curves.png**: Đường cong hiệu suất
   - ROC Curve với AUC score
   - Precision-Recall Curve với AUC score

5. **05_feature_importance.png**: Tầm quan trọng features
   - Absolute importance (thanh ngang)
   - Coefficient values với direction (+/-)
   - Top features được highlight

6. **06_learning_curves.png**: Đường cong học
   - Training vs Validation scores
   - Theo training set size
   - Phát hiện overfitting/underfitting

7. **07_summary_report.png**: Báo cáo tổng kết
   - Thông tin dataset
   - Cấu hình model
   - Performance metrics
   - Top 5 features quan trọng nhất
   - Key insights

---

## 🎨 Ví dụ output

### Simple Viz Output:
```
🚀 Training with Simple Visualization...
📊 Dataset: 398 samples
🤖 Training model...

📊 Classification Report:
              precision    recall  f1-score   support
           0       0.98      0.87      0.93        71
           1       0.47      0.89      0.62         9
    accuracy                           0.88        80

🎨 Creating visualization...
📊 Visualization saved to: simple_viz/training_results.png
💾 Model saved to: models/logreg_p_freelancer_accept.pkl
✅ Training completed!
```

### Full Viz Output:
```
🚀 Starting Enhanced Training with Visualizations...
📁 Output directory: /path/to/training_visualizations

📊 Building dataset...
💾 Saved 398 rows to dataset_p_freelancer_accept.csv

📈 Creating dataset overview...
🔍 Analyzing features...
🤖 Training model...

📊 Classification Report:
              precision    recall  f1-score   support
           0       0.98      0.87      0.93        71
           1       0.47      0.89      0.62         9

🎨 Creating visualizations...
   📊 Confusion Matrix...
   📈 Performance Curves...
   🔍 Feature Importance...
   📈 Learning Curves...
   📋 Summary Report...

💾 Model saved to: models/logreg_p_freelancer_accept.pkl

✅ Training completed! Check visualizations in: training_visualizations/
📁 Generated files:
   • 01_dataset_overview.png
   • 02_feature_analysis.png
   • 03_confusion_matrix.png
   • 04_performance_curves.png
   • 05_feature_importance.png
   • 06_learning_curves.png
   • 07_summary_report.png
```

---

## 💡 Khuyến nghị sử dụng

### Cho Development:
- Dùng **Simple Viz** để kiểm tra nhanh
- Dùng **Full Viz** khi cần phân tích sâu

### Cho Presentation/Luận văn:
- Dùng **Full Viz** để có đầy đủ biểu đồ
- Các file PNG có độ phân giải cao (300 DPI)
- Có thể copy trực tiếp vào Word/PowerPoint

### Cho Production:
- Dùng **Original** để tối ưu tốc độ
- Không cần visualization trong production

---

## 🔧 Tùy chỉnh

### Thay đổi output directory:
```python
OUTPUT_DIR = Path("my_custom_folder")
```

### Thay đổi figure size:
```python
plt.rcParams['figure.figsize'] = (16, 10)  # Width, Height
```

### Thay đổi DPI (độ phân giải):
```python
plt.savefig("output.png", dpi=600)  # Higher quality
```

### Thay đổi color scheme:
```python
sns.set_palette("Set2")  # Different color palette
```

---

## 🐛 Troubleshooting

### Lỗi "No module named matplotlib":
```bash
pip install matplotlib seaborn pandas
```

### Lỗi "cannot connect to X server" (Linux server):
```python
import matplotlib
matplotlib.use('Agg')  # Add before importing pyplot
```

### Lỗi font rendering:
```python
plt.rcParams['font.family'] = 'DejaVu Sans'
```

### Memory issues với dataset lớn:
- Dùng Simple Viz thay vì Full Viz
- Giảm DPI xuống 150-200
- Giảm figure size

---

## 📝 Notes

- Tất cả visualizations đều tự động save file PNG
- Files được đặt tên theo thứ tự để dễ sắp xếp
- Có thể chạy nhiều lần, files cũ sẽ bị ghi đè
- Visualization không ảnh hưởng đến model training
- Model được save giống hệt như bản original

---

## 🎯 Kết luận

Với 3 options này, bạn có thể:
- ✅ Giữ nguyên workflow cũ (Original)
- ✅ Thêm visualization đơn giản (Simple)  
- ✅ Có đầy đủ biểu đồ cho luận văn (Full)

Chọn option phù hợp với nhu cầu của bạn! 🚀