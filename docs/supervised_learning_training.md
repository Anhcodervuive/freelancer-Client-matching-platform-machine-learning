# HỌC CÓ GIÁM SÁT: LẤY NHÃN VÀ HUẤN LUYỆN MÔ HÌNH

## 1. Tổng quan

Hệ thống sử dụng **Supervised Learning** (Học có giám sát) để dự đoán hai xác suất quan trọng:

1. **p_freelancer_accept**: Xác suất freelancer chấp nhận lời mời (invitation)
2. **p_match**: Xác suất job và freelancer match thành công (dẫn đến hợp đồng)

Cả hai mô hình đều sử dụng **Logistic Regression** với 20 features được thiết kế thủ công.

---

## 2. Mô hình p_freelancer_accept

### 2.1. Mục tiêu

Dự đoán xác suất freelancer sẽ **ACCEPT** một lời mời (invitation) từ client cho một job cụ thể.

**Câu hỏi nghiên cứu:**
> "Nếu gửi invitation cho freelancer X về job Y, khả năng họ chấp nhận là bao nhiêu?"

**Ứng dụng thực tế:**
- Gợi ý client nên mời freelancer nào (có khả năng accept cao)
- Tối ưu hóa tỷ lệ chuyển đổi từ invitation → acceptance
- Giảm lãng phí thời gian với freelancer ít quan tâm


### 2.2. Cách lấy nhãn (Label Extraction)

#### 2.2.1. Nguồn dữ liệu

Nhãn được lấy từ bảng **job_invitation** - lưu trữ lịch sử các lời mời đã gửi.

**Cấu trúc bảng job_invitation:**
```sql
CREATE TABLE job_invitation (
    id VARCHAR PRIMARY KEY,
    job_id VARCHAR,
    freelancer_id VARCHAR,
    status ENUM('SENT', 'ACCEPTED', 'DECLINED', 'EXPIRED', 'WITHDRAWN'),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

#### 2.2.2. Quy tắc gán nhãn

**Label = 1 (Positive - Chấp nhận):**
```sql
SELECT * FROM job_invitation 
WHERE status = 'ACCEPTED'
```

**Điều kiện:** Freelancer đã click "Accept Invitation" trong hệ thống.

**Label = 0 (Negative - Từ chối):**
```sql
SELECT * FROM job_invitation 
WHERE status IN ('DECLINED', 'EXPIRED')
```

**Điều kiện:**
- `DECLINED`: Freelancer chủ động từ chối
- `EXPIRED`: Invitation hết hạn mà freelancer không phản hồi (coi như từ chối ngầm)


**Loại bỏ (Không dùng):**
```sql
-- Không lấy các trường hợp này
WHERE status = 'SENT'      -- Chưa có phản hồi (chưa biết kết quả)
   OR status = 'WITHDRAWN' -- Client rút lại lời mời
```

**Lý do:** Các trường hợp này chưa có kết quả cuối cùng, không thể gán nhãn chính xác.

#### 2.2.3. Ví dụ minh họa

**Dữ liệu thực tế:**

| job_id | freelancer_id | status | Label | Giải thích |
|--------|---------------|--------|-------|------------|
| job_001 | fr_123 | ACCEPTED | 1 | Freelancer chấp nhận |
| job_001 | fr_456 | DECLINED | 0 | Freelancer từ chối |
| job_002 | fr_789 | EXPIRED | 0 | Hết hạn, không phản hồi |
| job_002 | fr_234 | SENT | ❌ Bỏ qua | Chưa có kết quả |
| job_003 | fr_567 | WITHDRAWN | ❌ Bỏ qua | Client rút lại |

**Kết quả:** Từ 5 invitation, chỉ lấy 3 samples (1 positive, 2 negative) để train.


### 2.3. Xây dựng Dataset

#### 2.3.1. SQL Query lấy dữ liệu

```sql
SELECT
    ji.job_id,
    ji.freelancer_id,
    ji.status,
    
    -- Core similarity features
    mf.similarity_score,
    mf.budget_gap,
    mf.timezone_gap_hours,
    mf.level_gap,
    
    -- Job features
    mf.job_experience_level_num,
    mf.job_required_skill_count,
    mf.job_screening_question_count,
    mf.job_stats_applies,
    mf.job_stats_offers,
    mf.job_stats_accepts,
    
    -- Freelancer features
    mf.freelancer_skill_count,
    mf.freelancer_stats_applies,
    mf.freelancer_stats_offers,
    mf.freelancer_stats_accepts,
    mf.freelancer_invite_accept_rate,
    
    -- Pairwise features
    mf.skill_overlap_count,
    mf.skill_overlap_ratio,
    mf.has_past_collaboration,
    mf.past_collaboration_count,
    mf.has_viewed_job
    
FROM job_invitation ji
JOIN match_feature mf
  ON mf.job_id = ji.job_id
 AND mf.freelancer_id = ji.freelancer_id
WHERE ji.status IN ('ACCEPTED', 'DECLINED', 'EXPIRED')
```


#### 2.3.2. Xử lý dữ liệu

**Bước 1: Gán nhãn**
```python
def assign_label(status):
    if status == 'ACCEPTED':
        return 1  # Positive
    elif status in ['DECLINED', 'EXPIRED']:
        return 0  # Negative
    else:
        return None  # Bỏ qua
```

**Bước 2: Xử lý giá trị thiếu (Missing Values)**
```python
def handle_missing(value, default=0.0):
    return float(value) if value is not None else default
```

**Bước 3: Chuyển đổi boolean**
```python
def convert_boolean(value):
    return 1 if value else 0
```

**Bước 4: Tạo feature matrix X và label vector y**
```python
X = [
    [similarity_score, budget_gap, timezone_gap_hours, ...],  # Sample 1
    [0.85, 5000.0, 0.0, ...],                                 # Sample 2
    ...
]

y = [1, 0, 1, 0, ...]  # Labels tương ứng
```


### 2.4. Huấn luyện mô hình

#### 2.4.1. Kiến trúc mô hình

Sử dụng **Logistic Regression** với pipeline xử lý:

```
Input Features (20D) → StandardScaler → Logistic Regression → Probability Output
```

**Lý do chọn Logistic Regression:**
- ✅ Phù hợp với bài toán phân loại nhị phân (accept/decline)
- ✅ Output là xác suất [0, 1] - dễ diễn giải
- ✅ Nhanh, ổn định với dataset vừa và nhỏ
- ✅ Ít overfitting hơn so với mô hình phức tạp
- ✅ Có thể phân tích feature importance

#### 2.4.2. Cấu hình mô hình

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
])
```

**Giải thích tham số:**
- `max_iter=1000`: Số vòng lặp tối đa để tối ưu hóa
- `class_weight='balanced'`: Tự động cân bằng class (xử lý imbalanced data)
- `random_state=42`: Đảm bảo kết quả reproducible


#### 2.4.3. Quy trình huấn luyện

**Bước 1: Chia tập dữ liệu**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 80% train, 20% test
    random_state=42,
    stratify=y          # Giữ tỷ lệ class giống nhau
)
```

**Phân bổ dữ liệu:**
```
Total samples: 836
├── Training set: 668 samples (80%)
│   ├── Positive (ACCEPTED): ~334 samples
│   └── Negative (DECLINED/EXPIRED): ~334 samples
└── Test set: 168 samples (20%)
    ├── Positive: ~84 samples
    └── Negative: ~84 samples
```

**Bước 2: Chuẩn hóa features (StandardScaler)**

Công thức chuẩn hóa Z-score:
```
         x - μ
z = ─────────
          σ
```

Trong đó:
- `x`: giá trị gốc
- `μ`: trung bình của feature
- `σ`: độ lệch chuẩn

**Ví dụ:**
```python
# Feature: similarity_score
Original: [0.85, 0.72, 0.91, 0.68, ...]
Mean (μ): 0.78
Std (σ): 0.12

Scaled: [(0.85-0.78)/0.12, (0.72-0.78)/0.12, ...]
      = [0.58, -0.50, 1.08, -0.83, ...]
```


**Bước 3: Huấn luyện**
```python
model.fit(X_train, y_train)
```

**Quá trình tối ưu hóa:**

Logistic Regression tối ưu hóa hàm mất mát (loss function):

```
         n
L(w) = - Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
        i=1
```

Trong đó:
- `yᵢ`: nhãn thực tế (0 hoặc 1)
- `ŷᵢ`: xác suất dự đoán
- `w`: trọng số của mô hình

**Hàm sigmoid:**
```
              1
ŷ = ─────────────────
     1 + e^(-w·x)
```

**Bước 4: Đánh giá trên tập test**
```python
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```


#### 2.4.4. Kết quả đánh giá

**Ví dụ Classification Report:**
```
              precision    recall  f1-score   support

           0       0.82      0.79      0.80        84
           1       0.80      0.83      0.81        84

    accuracy                           0.81       168
   macro avg       0.81      0.81      0.81       168
weighted avg       0.81      0.81      0.81       168
```

**Giải thích các chỉ số:**

1. **Precision (Độ chính xác):**
```
                True Positive
Precision = ─────────────────────────
             True Positive + False Positive
```
Trong 100 lần dự đoán "ACCEPT", có bao nhiêu lần đúng?

2. **Recall (Độ phủ):**
```
            True Positive
Recall = ─────────────────────────
          True Positive + False Negative
```
Trong 100 trường hợp thực tế "ACCEPT", model phát hiện được bao nhiêu?

3. **F1-Score (Điểm cân bằng):**
```
            2 × Precision × Recall
F1-Score = ─────────────────────────
            Precision + Recall
```
Trung bình điều hòa của Precision và Recall.


#### 2.4.5. Lưu mô hình

```python
import joblib

model_path = "models/logreg_p_freelancer_accept.pkl"
joblib.dump(model, model_path)
```

**Cấu trúc file lưu:**
- Pipeline hoàn chỉnh (StandardScaler + LogisticRegression)
- Tất cả tham số đã học (weights, biases)
- Thống kê chuẩn hóa (mean, std của mỗi feature)

---

## 3. Mô hình p_match

### 3.1. Mục tiêu

Dự đoán xác suất một cặp (job, freelancer) sẽ **match thành công** - tức là dẫn đến hợp đồng làm việc thực tế.

**Câu hỏi nghiên cứu:**
> "Cặp job X và freelancer Y có khả năng làm việc cùng nhau (ký hợp đồng) là bao nhiêu?"

**Ứng dụng thực tế:**
- Xếp hạng freelancer phù hợp nhất cho mỗi job
- Gợi ý job phù hợp nhất cho mỗi freelancer
- Tối ưu hóa matching algorithm tổng thể


### 3.2. Cách lấy nhãn (Label Extraction)

#### 3.2.1. Nguồn dữ liệu

Nhãn được lấy từ **nhiều bảng** để xác định match thành công:

```
job_invitation → job_proposal → job_offer → contract
```

**Luồng dữ liệu:**
1. Client gửi **invitation** cho freelancer
2. Freelancer gửi **proposal** (đề xuất)
3. Client gửi **offer** (đề nghị hợp đồng)
4. Freelancer accept offer → tạo **contract**

#### 3.2.2. Quy tắc gán nhãn

**Label = 1 (Positive - Match thành công):**

Một trong các điều kiện sau:

**Cách 1: Có contract hoàn thành**
```sql
SELECT DISTINCT job_id, freelancer_id
FROM contract
WHERE status IN ('ACTIVE', 'COMPLETED')
```

**Cách 2: Có offer được chấp nhận**
```sql
SELECT DISTINCT job_id, freelancer_id
FROM job_offer
WHERE status = 'ACCEPTED'
```

**Cách 3: Có proposal được hire**
```sql
SELECT DISTINCT job_id, freelancer_id
FROM job_proposal
WHERE status = 'HIRED'
```


**Label = 0 (Negative - Match thất bại):**

Các trường hợp có tương tác nhưng không dẫn đến hợp đồng:

**Từ job_proposal:**
```sql
SELECT DISTINCT job_id, freelancer_id
FROM job_proposal
WHERE status IN ('DECLINED', 'WITHDRAWN')
  AND NOT EXISTS (
      SELECT 1 FROM contract c
      WHERE c.job_id = job_proposal.job_id
        AND c.freelancer_id = job_proposal.freelancer_id
  )
```

**Từ job_offer:**
```sql
SELECT DISTINCT job_id, freelancer_id
FROM job_offer
WHERE status IN ('DECLINED', 'EXPIRED')
  AND NOT EXISTS (
      SELECT 1 FROM contract c
      WHERE c.job_id = job_offer.job_id
        AND c.freelancer_id = job_offer.freelancer_id
  )
```

**Từ job_invitation:**
```sql
SELECT DISTINCT job_id, freelancer_id
FROM job_invitation
WHERE status = 'ACCEPTED'
  AND DATEDIFF(NOW(), updated_at) > 30  -- Đã 30 ngày
  AND NOT EXISTS (
      SELECT 1 FROM contract c
      WHERE c.job_id = job_invitation.job_id
        AND c.freelancer_id = job_invitation.freelancer_id
  )
```


**Loại bỏ (Không dùng):**
```sql
-- Các trường hợp đang pending, chưa có kết quả cuối cùng
WHERE proposal.status = 'SUBMITTED'
   OR offer.status = 'SENT'
   OR invitation.status = 'SENT'
```

#### 3.2.3. Ví dụ minh họa

**Kịch bản 1: Match thành công (Label = 1)**
```
job_001 + fr_123:
  ├─ invitation: ACCEPTED
  ├─ proposal: SUBMITTED
  ├─ offer: ACCEPTED
  └─ contract: ACTIVE ✅
  
→ Label = 1
```

**Kịch bản 2: Match thất bại (Label = 0)**
```
job_002 + fr_456:
  ├─ invitation: ACCEPTED
  ├─ proposal: SUBMITTED
  └─ offer: DECLINED ❌
  
→ Label = 0
```

**Kịch bản 3: Bỏ qua**
```
job_003 + fr_789:
  ├─ invitation: ACCEPTED
  └─ proposal: SUBMITTED (đang chờ)
  
→ Bỏ qua (chưa có kết quả)
```


### 3.3. Xây dựng Dataset

#### 3.3.1. SQL Query tổng hợp

```sql
-- Lấy tất cả cặp (job, freelancer) có tương tác
WITH all_interactions AS (
    -- Từ contracts (positive)
    SELECT DISTINCT job_id, freelancer_id, 1 as label
    FROM contract
    WHERE status IN ('ACTIVE', 'COMPLETED')
    
    UNION
    
    -- Từ offers accepted (positive)
    SELECT DISTINCT job_id, freelancer_id, 1 as label
    FROM job_offer
    WHERE status = 'ACCEPTED'
    
    UNION
    
    -- Từ proposals declined (negative)
    SELECT DISTINCT job_id, freelancer_id, 0 as label
    FROM job_proposal
    WHERE status IN ('DECLINED', 'WITHDRAWN')
    
    UNION
    
    -- Từ offers declined (negative)
    SELECT DISTINCT job_id, freelancer_id, 0 as label
    FROM job_offer
    WHERE status IN ('DECLINED', 'EXPIRED')
)

SELECT 
    ai.job_id,
    ai.freelancer_id,
    ai.label,
    mf.*  -- Tất cả features từ match_feature
FROM all_interactions ai
JOIN match_feature mf
  ON mf.job_id = ai.job_id
 AND mf.freelancer_id = ai.freelancer_id
```


### 3.4. Huấn luyện mô hình

Quy trình huấn luyện **giống hệt** với p_freelancer_accept:

1. **Chia dữ liệu:** 80% train, 20% test
2. **Chuẩn hóa:** StandardScaler
3. **Mô hình:** Logistic Regression
4. **Đánh giá:** Precision, Recall, F1-Score

**Điểm khác biệt:**
- **Nhãn khác:** p_match dựa trên contract, p_freelancer_accept dựa trên invitation
- **Ý nghĩa khác:** p_match đo "khả năng làm việc thành công", p_freelancer_accept đo "khả năng chấp nhận lời mời"

---

## 4. So sánh hai mô hình

| Tiêu chí | p_freelancer_accept | p_match |
|----------|---------------------|---------|
| **Mục tiêu** | Dự đoán freelancer accept invitation | Dự đoán match thành công (contract) |
| **Nguồn nhãn** | job_invitation.status | contract, job_offer, job_proposal |
| **Positive label** | status = 'ACCEPTED' | Có contract hoặc offer accepted |
| **Negative label** | status IN ('DECLINED', 'EXPIRED') | Có tương tác nhưng không có contract |
| **Thời điểm dự đoán** | Khi gửi invitation | Bất kỳ lúc nào trong quá trình |
| **Use case** | Gợi ý client nên mời ai | Xếp hạng freelancer phù hợp |
| **Features quan trọng** | freelancer_invite_accept_rate, skill_overlap_ratio | similarity_score, has_past_collaboration |


---

## 5. Quy trình tổng thể

### 5.1. Sơ đồ luồng dữ liệu

```
┌─────────────────────────────────────────────────────────────┐
│                    HISTORICAL DATA                          │
│  (job_invitation, job_proposal, job_offer, contract)        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   LABEL EXTRACTION                          │
│  • p_freelancer_accept: invitation.status                   │
│  • p_match: contract existence                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   JOIN WITH FEATURES                        │
│  • Lấy 20 features từ match_feature                         │
│  • Xử lý missing values                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   TRAIN/TEST SPLIT                          │
│  • 80% training, 20% testing                                │
│  • Stratified sampling                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   STANDARDIZATION                           │
│  • StandardScaler: z = (x - μ) / σ                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                LOGISTIC REGRESSION TRAINING                 │
│  • Optimize: L(w) = -Σ[y log(ŷ) + (1-y)log(1-ŷ)]           │
│  • Output: ŷ = 1 / (1 + e^(-w·x))                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      EVALUATION                             │
│  • Precision, Recall, F1-Score                              │
│  • Confusion Matrix                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      SAVE MODEL                             │
│  • joblib.dump(model, "model.pkl")                          │
└─────────────────────────────────────────────────────────────┘
```


### 5.2. Code implementation tổng hợp

```python
# 1. Load và xử lý dữ liệu
async def build_dataset():
    # Lấy dữ liệu từ database
    data = await fetch_data_from_db()
    
    # Gán nhãn
    data['label'] = data['status'].apply(assign_label)
    
    # Loại bỏ samples không có nhãn
    data = data[data['label'].notna()]
    
    # Tách features và labels
    feature_cols = [
        'similarity_score', 'budget_gap', 'timezone_gap_hours',
        'level_gap', 'job_experience_level_num', ...
    ]
    X = data[feature_cols].values
    y = data['label'].values
    
    return X, y

# 2. Chia dữ liệu
X, y = await build_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Xây dựng pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ))
])

# 4. Huấn luyện
model.fit(X_train, y_train)

# 5. Đánh giá
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Lưu model
joblib.dump(model, 'model.pkl')
```


---

## 6. Sử dụng mô hình đã train

### 6.1. Load mô hình

```python
import joblib
import numpy as np

# Load model
model = joblib.load('logreg_p_freelancer_accept.pkl')
```

### 6.2. Dự đoán cho cặp mới

```python
def predict_p_freelancer_accept(features):
    """
    features: list of 20 values theo đúng thứ tự
    [
        similarity_score,
        budget_gap,
        timezone_gap_hours,
        level_gap,
        job_experience_level_num,
        job_required_skill_count,
        job_screening_question_count,
        job_stats_applies,
        job_stats_offers,
        job_stats_accepts,
        freelancer_skill_count,
        freelancer_stats_applies,
        freelancer_stats_offers,
        freelancer_stats_accepts,
        freelancer_invite_accept_rate,
        skill_overlap_count,
        skill_overlap_ratio,
        has_past_collaboration,
        past_collaboration_count,
        has_viewed_job
    ]
    """
    # Reshape thành 2D array
    X = np.array(features).reshape(1, -1)
    
    # Dự đoán xác suất
    proba = model.predict_proba(X)[0, 1]
    
    return float(proba)
```


### 6.3. Ví dụ thực tế

```python
# Ví dụ: Dự đoán xác suất freelancer accept invitation

features = [
    0.85,    # similarity_score (cao - phù hợp)
    5000.0,  # budget_gap
    0.0,     # timezone_gap_hours
    0.0,     # level_gap (cùng level)
    2.0,     # job_experience_level_num (INTERMEDIATE)
    5.0,     # job_required_skill_count
    2.0,     # job_screening_question_count
    10.0,    # job_stats_applies
    3.0,     # job_stats_offers
    2.0,     # job_stats_accepts
    8.0,     # freelancer_skill_count
    25.0,    # freelancer_stats_applies
    15.0,    # freelancer_stats_offers
    12.0,    # freelancer_stats_accepts
    0.80,    # freelancer_invite_accept_rate (80% - cao)
    4.0,     # skill_overlap_count
    0.80,    # skill_overlap_ratio (80% - cao)
    0,       # has_past_collaboration (chưa từng làm việc)
    0.0,     # past_collaboration_count
    1        # has_viewed_job (đã xem job)
]

probability = predict_p_freelancer_accept(features)
print(f"Xác suất freelancer accept: {probability:.2%}")
# Output: Xác suất freelancer accept: 87.34%
```

**Giải thích kết quả:**
- Xác suất 87.34% → Rất cao
- Lý do: similarity_score cao (0.85), skill_overlap_ratio cao (0.80), freelancer có lịch sử accept tốt (80%)
- **Khuyến nghị:** Nên gửi invitation cho freelancer này


---

## 7. Ưu điểm và hạn chế

### 7.1. Ưu điểm

**1. Dựa trên dữ liệu thực tế**
- Nhãn được lấy từ hành vi người dùng thực tế
- Phản ánh chính xác patterns trong hệ thống

**2. Diễn giải được (Interpretable)**
- Logistic Regression cho phép phân tích feature importance
- Có thể giải thích tại sao một dự đoán được đưa ra

**3. Nhanh và hiệu quả**
- Training nhanh (< 1 phút với 1000 samples)
- Inference cực nhanh (< 1ms per prediction)
- Phù hợp cho production

**4. Ổn định**
- Ít overfitting
- Hoạt động tốt với dataset vừa và nhỏ
- Không cần GPU

### 7.2. Hạn chế

**1. Phụ thuộc vào chất lượng features**
- Cần feature engineering thủ công
- Features kém → Model kém

**2. Giả định tuyến tính**
- Logistic Regression giả định quan hệ tuyến tính
- Không capture được non-linear patterns phức tạp

**3. Cần dữ liệu lịch sử**
- Cold start problem: User/Job mới không có lịch sử
- Cần thời gian thu thập dữ liệu

**4. Class imbalance**
- Thường có nhiều negative hơn positive
- Cần xử lý bằng class_weight='balanced'


---

## 8. Kết luận

### 8.1. Tóm tắt quy trình

Hệ thống sử dụng **Feature-Based Supervised Learning** với hai mô hình:

**p_freelancer_accept:**
- **Input:** 20 features từ match_feature
- **Label:** invitation.status (ACCEPTED vs DECLINED/EXPIRED)
- **Output:** Xác suất freelancer chấp nhận lời mời
- **Use case:** Gợi ý client nên mời freelancer nào

**p_match:**
- **Input:** 20 features từ match_feature
- **Label:** Có contract thành công hay không
- **Output:** Xác suất match thành công
- **Use case:** Xếp hạng freelancer phù hợp cho job

### 8.2. Công thức toán học tổng hợp

**1. Chuẩn hóa features:**
```
         xᵢ - μᵢ
zᵢ = ─────────
          σᵢ
```

**2. Linear combination:**
```
         20
score = Σ (wᵢ × zᵢ) + b
        i=1
```

**3. Sigmoid activation:**
```
              1
P(y=1) = ─────────────
          1 + e^(-score)
```

**4. Loss function (Binary Cross-Entropy):**
```
         n
L(w) = - Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
        i=1
```

### 8.3. Đóng góp của phương pháp

✅ Tự động hóa việc đánh giá độ phù hợp
✅ Dự đoán chính xác hành vi người dùng
✅ Cải thiện trải nghiệm matching
✅ Tăng tỷ lệ chuyển đổi (conversion rate)
✅ Tiết kiệm thời gian cho cả client và freelancer
