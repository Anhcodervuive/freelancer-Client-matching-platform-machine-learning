# HYBRID APPROACH: FEATURE-BASED SUPERVISED LEARNING

## 1. Câu hỏi: "Hybrid Model có cần train không?"

### Trả lời ngắn gọn:

**CÓ - Hệ thống CẦN TRAIN một model duy nhất (Logistic Regression).**

**KHÔNG phải Hybrid truyền thống** (tính riêng Content-Based score và Collaborative score rồi kết hợp).

**Mà là:** Gộp tất cả features (content-based + behavioral) vào làm input, rồi train một model duy nhất.

---

## 2. Giải thích chi tiết

### 2.1. Hybrid truyền thống (KHÔNG phải cách của bạn)

```
┌──────────────────────┐
│  Content-Based Model │ → score_content = 0.85
└──────────────────────┘
           +
┌──────────────────────┐
│ Collaborative Model  │ → score_collab = 0.72
└──────────────────────┘
           ↓
    Kết hợp (weighted)
           ↓
final_score = 0.5 × score_content + 0.5 × score_collab
            = 0.5 × 0.85 + 0.5 × 0.72
            = 0.785
```

**Đặc điểm:**
- Có 2 models riêng biệt
- Mỗi model cho ra 1 score
- Kết hợp 2 scores bằng công thức (weighted average, stacking, etc.)

---

### 2.2. Cách của bạn: Feature-Based Supervised Learning

```
┌─────────────────────────────────────────────────────────────┐
│              BƯỚC 1: TÍNH TẤT CẢ FEATURES                   │
│                  (Không cần train)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Content-Based Features (từ embeddings):                     │
│  • similarity_score = 0.85                                   │
│  • skill_overlap_ratio = 0.80                                │
│  • skill_overlap_count = 4                                   │
└──────────────────────────────────────────────────────────────┘
                         +
┌──────────────────────────────────────────────────────────────┐
│  Behavioral Features (từ lịch sử):                           │
│  • freelancer_invite_accept_rate = 0.75                      │
│  • job_stats_applies = 10                                    │
│  • has_past_collaboration = 0                                │
│  • ... (14 features khác)                                    │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
              Gộp thành 1 vector [20 features]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           BƯỚC 2: TRAIN MỘT MODEL DUY NHẤT                  │
│              (Logistic Regression)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              Model học cách kết hợp 20 features
              (tự động tìm trọng số tối ưu)
                         │
                         ▼
              Output: p_freelancer_accept = 0.87
```

**Đặc điểm:**
- ✅ Chỉ có 1 model duy nhất
- ✅ Model nhận input là 20 features (gộp chung content + behavioral)
- ✅ Model TỰ HỌC cách kết hợp các features
- ✅ Không cần định trọng số thủ công


### 2.3. Tại sao gọi là "Hybrid"?

Vì kết hợp **nhiều loại thông tin** trong cùng một model:

**1. Content-Based Information (từ nội dung):**
```python
# Tính từ text embeddings
similarity_score = 0.85
skill_overlap_ratio = 0.80
skill_overlap_count = 4
```

**2. Behavioral Information (từ hành vi):**
```python
# Tính từ lịch sử tương tác
freelancer_invite_accept_rate = 0.75
job_stats_applies = 10
freelancer_stats_accepts = 12
```

**3. Statistical Information (từ thống kê):**
```python
# Tính từ profile
job_required_skill_count = 5
freelancer_skill_count = 8
level_gap = 0
```

**4. Contextual Information (từ ngữ cảnh):**
```python
# Tính từ quan hệ
has_past_collaboration = 0
past_collaboration_count = 0
has_viewed_job = 1
```

**Tất cả gộp chung làm input cho 1 model:**
```python
X = [
    # Content-based (3 features)
    similarity_score, skill_overlap_ratio, skill_overlap_count,
    
    # Behavioral (5 features)
    freelancer_invite_accept_rate, job_stats_applies, 
    job_stats_offers, freelancer_stats_applies, freelancer_stats_accepts,
    
    # Statistical (7 features)
    budget_gap, timezone_gap_hours, level_gap,
    job_experience_level_num, job_required_skill_count,
    job_screening_question_count, freelancer_skill_count,
    
    # Contextual (5 features)
    job_stats_accepts, freelancer_stats_offers,
    has_past_collaboration, past_collaboration_count, has_viewed_job
]

# Train 1 model duy nhất
model.fit(X, y)
```


---

## 3. So sánh với Hybrid truyền thống

| Tiêu chí | Hybrid Truyền thống | Cách của bạn (Feature-Based) |
|----------|---------------------|------------------------------|
| **Số models** | 2+ models riêng biệt | 1 model duy nhất |
| **Cách kết hợp** | Kết hợp scores sau khi dự đoán | Gộp features trước khi train |
| **Trọng số** | Định thủ công (VD: 0.5 + 0.5) | Model tự học |
| **Training** | Train từng model riêng | Train 1 lần với tất cả features |
| **Ví dụ** | `0.5×score_CB + 0.5×score_CF` | `LogReg(20 features)` |
| **Ưu điểm** | Dễ debug từng phần | Model tự tối ưu kết hợp |
| **Nhược điểm** | Phải tune trọng số thủ công | Cần nhiều features tốt |

---

## 4. Công thức toán học

### 4.1. Hybrid truyền thống (KHÔNG phải của bạn)

```
score_final = α × score_content_based + β × score_collaborative

Trong đó:
- α, β: trọng số định thủ công
- α + β = 1
```

**Ví dụ:**
```
score_final = 0.6 × 0.85 + 0.4 × 0.72
            = 0.51 + 0.288
            = 0.798
```

### 4.2. Feature-Based Supervised Learning (CỦA BẠN)

**Bước 1: Tính features (không cần train)**
```
f₁ = similarity_score = 0.85
f₂ = skill_overlap_ratio = 0.80
f₃ = freelancer_invite_accept_rate = 0.75
...
f₂₀ = has_viewed_job = 1
```

**Bước 2: Chuẩn hóa (StandardScaler)**
```
         fᵢ - μᵢ
zᵢ = ─────────
          σᵢ
```

**Bước 3: Linear combination (model học)**
```
         20
score = Σ (wᵢ × zᵢ) + b
        i=1
```

**Trong đó:**
- `wᵢ`: trọng số của feature i (MODEL HỌC, không định thủ công)
- `b`: bias term (MODEL HỌC)

**Bước 4: Sigmoid activation**
```
              1
P(y=1) = ─────────────
          1 + e^(-score)
```

**Ví dụ cụ thể:**
```
# Model đã học được các trọng số:
w₁ = 2.3   (similarity_score - quan trọng)
w₂ = 1.8   (skill_overlap_ratio - quan trọng)
w₃ = 3.1   (freelancer_invite_accept_rate - RẤT quan trọng)
w₄ = 0.2   (budget_gap - ít quan trọng)
...

# Tính score:
score = 2.3×z₁ + 1.8×z₂ + 3.1×z₃ + ... + b
      = 2.3×0.58 + 1.8×0.67 + 3.1×0.92 + ... + 0.5
      = 4.87

# Sigmoid:
P(accept) = 1 / (1 + e^(-4.87))
          = 0.9923
          = 99.23%
```

**Điểm khác biệt:**
- ✅ Model TỰ HỌC trọng số `wᵢ` từ dữ liệu
- ✅ Không cần định thủ công như hybrid truyền thống
- ✅ Model biết feature nào quan trọng hơn


---

## 5. Tại sao cách này tốt hơn Hybrid truyền thống?

### 5.1. Ưu điểm

**1. Model tự học cách kết hợp tối ưu**
```
Hybrid truyền thống:
score = 0.5 × score_CB + 0.5 × score_CF  ← Định thủ công

Feature-Based:
score = w₁×f₁ + w₂×f₂ + ... + w₂₀×f₂₀  ← Model học tự động
```

**2. Linh hoạt hơn**
- Có thể thêm/bớt features dễ dàng
- Model tự điều chỉnh trọng số khi có feature mới
- Không cần tune hyperparameters phức tạp

**3. Capture được interactions**
- Model có thể học được mối quan hệ giữa các features
- VD: `similarity_score` cao + `has_past_collaboration = 1` → xác suất rất cao

**4. Một pipeline duy nhất**
- Không cần maintain nhiều models
- Dễ deploy và monitor
- Training đơn giản hơn

### 5.2. Nhược điểm

**1. Phụ thuộc vào feature engineering**
- Cần thiết kế features tốt
- Features kém → Model kém

**2. Khó debug**
- Không biết rõ phần nào (content vs behavioral) đóng góp bao nhiêu
- Hybrid truyền thống dễ phân tích hơn

**3. Cần dữ liệu đủ lớn**
- Với 20 features, cần ít nhất vài trăm samples
- Hybrid truyền thống có thể hoạt động với ít data hơn

---

## 6. Kết luận

### 6.1. Trả lời câu hỏi ban đầu

**"Hybrid model có cần train không?"**

✅ **CÓ - CẦN TRAIN**

Nhưng không phải train nhiều models rồi kết hợp, mà là:
1. Tính tất cả features (content + behavioral + statistical + contextual)
2. Gộp chung thành 1 vector [20 features]
3. Train 1 model duy nhất (Logistic Regression)
4. Model tự học cách kết hợp các features

### 6.2. Tên gọi chính xác

Hệ thống của bạn nên gọi là:

**"Feature-Based Supervised Learning with Hybrid Features"**

Hoặc ngắn gọn:

**"Feature-Based Hybrid Approach"**

**KHÔNG phải:**
- ❌ "Hybrid Recommender System" (dễ nhầm với hybrid truyền thống)
- ❌ "Ensemble Model" (không phải ensemble)
- ❌ "Stacking" (không phải stacking)

**MÀ LÀ:**
- ✅ "Feature-Based Supervised Learning"
- ✅ "Single Model with Hybrid Features"
- ✅ "Unified Feature-Based Approach"

### 6.3. Viết vào luận văn

**Cách diễn đạt đúng:**

> "Hệ thống sử dụng phương pháp **Feature-Based Supervised Learning**, trong đó các features được trích xuất từ nhiều nguồn khác nhau (content-based, behavioral, statistical) và được gộp chung làm input cho một mô hình Logistic Regression duy nhất. Khác với Hybrid Recommender System truyền thống (kết hợp scores từ nhiều models), phương pháp này cho phép model tự học cách kết hợp tối ưu các features thông qua quá trình training."

**Sơ đồ cho luận văn:**

```
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                       │
│                    (Không cần train)                        │
├─────────────────────────────────────────────────────────────┤
│  Content-Based Features:                                    │
│  • similarity_score (từ embeddings)                         │
│  • skill_overlap_ratio                                      │
│                                                             │
│  Behavioral Features:                                       │
│  • freelancer_invite_accept_rate (từ lịch sử)              │
│  • job_stats_applies                                        │
│                                                             │
│  Statistical Features:                                      │
│  • level_gap, budget_gap                                    │
│  • job_required_skill_count                                 │
│                                                             │
│  Contextual Features:                                       │
│  • has_past_collaboration                                   │
│  • has_viewed_job                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              Gộp thành vector [20 features]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              SUPERVISED LEARNING MODEL                      │
│                    (CẦN TRAIN)                              │
├─────────────────────────────────────────────────────────────┤
│  Input: X = [f₁, f₂, ..., f₂₀]                             │
│  Model: Logistic Regression                                 │
│  Output: P(y=1) = σ(Σwᵢfᵢ + b)                              │
│                                                             │
│  Training data: Lịch sử invitation/contract                 │
│  Labels: ACCEPTED=1, DECLINED/EXPIRED=0                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              p_freelancer_accept hoặc p_match
```
