# TÍNH TOÁN ĐIỂM TƯƠNG ĐỒNG CONTENT-BASED

## 1. Tổng quan

Hệ thống sử dụng **Content-Based Filtering** để tính độ tương đồng giữa Job Post và Freelancer Profile dựa trên nội dung văn bản và kỹ năng. Điểm tương đồng cuối cùng là **similarity_score** trong khoảng [0, 1].

---

## 2. Công thức tổng quát

### 2.1. Similarity Score tổng hợp

```
similarity_score = w₁ × sim_FULL + w₂ × sim_SKILLS + w₃ × sim_DOMAIN
```

**Trong đó:**
- `w₁ = 0.2` (trọng số cho FULL)
- `w₂ = 0.6` (trọng số cho SKILLS) 
- `w₃ = 0.2` (trọng số cho DOMAIN)
- `w₁ + w₂ + w₃ = 1.0`

**Giải thích:** 
- SKILLS có trọng số cao nhất (60%) vì kỹ năng là yếu tố quan trọng nhất trong matching
- FULL và DOMAIN mỗi loại chiếm 20% để bổ sung thông tin về mô tả công việc và lĩnh vực

---

## 3. Tính toán từng thành phần

### 3.1. Cosine Similarity

Tất cả các thành phần đều sử dụng **Cosine Similarity** để đo độ tương đồng giữa hai vector embedding:

```
                    A · B
cos(A, B) = ─────────────────
             ||A|| × ||B||
```

**Trong đó:**
- `A · B` = tích vô hướng (dot product) = Σ(aᵢ × bᵢ)
- `||A||` = độ dài vector A = √(Σaᵢ²)
- `||B||` = độ dài vector B = √(Σbᵢ²)

**Giá trị:** 
- `cos(A, B) = 1`: hai vector giống hệt nhau (tương đồng hoàn toàn)
- `cos(A, B) = 0`: hai vector vuông góc (không liên quan)
- `cos(A, B) = -1`: hai vector ngược hướng (đối lập)

**Code implementation:**
```python
def cosine_similarity(a, b):
    dot_product = sum(aᵢ × bᵢ for i in range(len(a)))
    norm_a = sqrt(sum(aᵢ² for i in range(len(a))))
    norm_b = sqrt(sum(bᵢ² for i in range(len(b))))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a × norm_b)
```

---

### 3.2. FULL Similarity (sim_FULL)

**Mục đích:** Đo độ tương đồng giữa mô tả công việc và hồ sơ freelancer

**Bước 1: Tạo văn bản đầy đủ**

```
text_job = job.title + " " + job.description
text_freelancer = freelancer.title + " " + freelancer.bio
```

**Ví dụ:**
```
text_job = "Senior React Developer Build scalable web applications using React and Node.js"
text_freelancer = "Full-stack Developer Experienced in React, Node.js, and modern web technologies"
```

**Bước 2: Chuyển đổi thành embedding vector**

Sử dụng mô hình **Sentence Transformer** (all-MiniLM-L6-v2):

```
E_job_full = SentenceTransformer.encode(text_job)
E_freelancer_full = SentenceTransformer.encode(text_freelancer)
```

- Mỗi embedding là vector 384 chiều
- Vector được normalize về độ dài = 1

**Bước 3: Tính cosine similarity**

```
sim_FULL = cosine_similarity(E_job_full, E_freelancer_full)
```

---

### 3.3. SKILLS Similarity (sim_SKILLS)

**Mục đích:** Đo độ tương đồng về kỹ năng kỹ thuật (quan trọng nhất - 60%)

**Bước 1: Chuẩn hóa danh sách kỹ năng**

```python
# Normalize và loại bỏ trùng lặp
skills_job = normalize_skill_list(["React", "Node.js", "PostgreSQL"])
# → ["react", "nodejs", "postgresql"]

skills_freelancer = normalize_skill_list(["React", "TypeScript", "Node.js", "MongoDB"])
# → ["mongodb", "nodejs", "react", "typescript"]  # sorted alphabetically
```

**Bảng mapping chuẩn hóa:**
| Input | Output |
|-------|--------|
| ReactJS, React JS | react |
| Node.js, Node JS, Node | nodejs |
| PostgreSQL, Postgres | postgresql |
| TypeScript | ts |
| JavaScript | js |

**Bước 2: Embedding từng skill (Mean Pooling)**

Thay vì embed toàn bộ danh sách, ta embed từng skill riêng lẻ rồi lấy trung bình:

```
# Job skills embedding
E₁ = embed("react")        # vector 384 chiều
E₂ = embed("nodejs")       # vector 384 chiều  
E₃ = embed("postgresql")   # vector 384 chiều

E_job_skills = (E₁ + E₂ + E₃) / 3
```

**Công thức tổng quát:**

```
              n
             Σ embed(skillᵢ)
            i=1
E_skills = ─────────────────
                 n
```

**Bước 3: Normalize lại vector kết quả**

```
E_job_skills_normalized = E_job_skills / ||E_job_skills||
E_freelancer_skills_normalized = E_freelancer_skills / ||E_freelancer_skills||
```

**Bước 4: Tính cosine similarity**

```
sim_SKILLS = cosine_similarity(E_job_skills_normalized, E_freelancer_skills_normalized)
```

**Lý do dùng Mean Pooling:**
- Giảm ảnh hưởng của thứ tự skill trong danh sách
- Mỗi skill đóng góp đều vào vector tổng thể
- Phù hợp khi số lượng skill khác nhau giữa job và freelancer

---

### 3.4. DOMAIN Similarity (sim_DOMAIN)

**Mục đích:** Đo độ tương đồng về lĩnh vực chuyên môn

**Bước 1: Tạo văn bản lĩnh vực**

```
text_job_domain = job.category + " " + job.specialty
text_freelancer_domain = " ".join(freelancer.skills_normalized)
```

**Ví dụ:**
```
text_job_domain = "Web Development Full-stack Development"
text_freelancer_domain = "mongodb nodejs react typescript"
```

**Bước 2: Embedding**

```
E_job_domain = SentenceTransformer.encode(text_job_domain)
E_freelancer_domain = SentenceTransformer.encode(text_freelancer_domain)
```

**Bước 3: Tính cosine similarity**

```
sim_DOMAIN = cosine_similarity(E_job_domain, E_freelancer_domain)
```

---

## 4. Ví dụ tính toán cụ thể

### 4.1. Dữ liệu đầu vào

**Job Post:**
```
Title: "Senior React Developer"
Description: "Build scalable web applications using React, Node.js, and PostgreSQL"
Category: "Web Development"
Specialty: "Full-stack Development"
Required Skills: ["React", "Node.js", "PostgreSQL", "TypeScript"]
```

**Freelancer Profile:**
```
Title: "Full-stack Developer"
Bio: "5 years experience in modern web development with React and Node.js"
Skills: ["React", "Node.js", "MongoDB", "TypeScript", "Docker"]
```

### 4.2. Tính toán từng bước

**Bước 1: FULL Similarity**
```
text_job = "Senior React Developer Build scalable web applications..."
text_freelancer = "Full-stack Developer 5 years experience in modern web..."

E_job_full = [0.023, -0.145, 0.089, ..., 0.234]      # 384 chiều
E_freelancer_full = [0.034, -0.123, 0.076, ..., 0.198]  # 384 chiều

sim_FULL = cosine_similarity(E_job_full, E_freelancer_full)
         = 0.78
```

**Bước 2: SKILLS Similarity**
```
skills_job = ["react", "nodejs", "postgresql", "typescript"]
skills_freelancer = ["docker", "mongodb", "nodejs", "react", "typescript"]

# Embed từng skill
E_react = [0.12, 0.34, ..., -0.23]
E_nodejs = [0.45, -0.12, ..., 0.67]
E_postgresql = [-0.23, 0.56, ..., 0.12]
E_typescript = [0.34, 0.23, ..., -0.45]

# Mean pooling cho job
E_job_skills = (E_react + E_nodejs + E_postgresql + E_typescript) / 4
E_job_skills_norm = normalize(E_job_skills)

# Mean pooling cho freelancer  
E_fr_skills = (E_docker + E_mongodb + E_nodejs + E_react + E_typescript) / 5
E_fr_skills_norm = normalize(E_fr_skills)

sim_SKILLS = cosine_similarity(E_job_skills_norm, E_fr_skills_norm)
           = 0.85
```

**Bước 3: DOMAIN Similarity**
```
text_job_domain = "Web Development Full-stack Development"
text_freelancer_domain = "docker mongodb nodejs react typescript"

E_job_domain = [0.23, -0.45, ..., 0.67]
E_freelancer_domain = [0.19, -0.38, ..., 0.72]

sim_DOMAIN = cosine_similarity(E_job_domain, E_freelancer_domain)
           = 0.72
```

**Bước 4: Tính similarity_score tổng hợp**
```
similarity_score = 0.2 × sim_FULL + 0.6 × sim_SKILLS + 0.2 × sim_DOMAIN
                 = 0.2 × 0.78 + 0.6 × 0.85 + 0.2 × 0.72
                 = 0.156 + 0.510 + 0.144
                 = 0.810
```

**Kết luận:** Job và Freelancer có độ tương đồng **81%** (rất phù hợp)

---

## 5. Mô hình Sentence Transformer

### 5.1. Thông tin mô hình

- **Tên:** `sentence-transformers/all-MiniLM-L6-v2`
- **Kiến trúc:** MiniLM (Mini Language Model) - 6 layers
- **Kích thước embedding:** 384 chiều
- **Training data:** 1 tỷ+ câu từ nhiều nguồn
- **Đặc điểm:**
  - Nhẹ, nhanh (22.7M parameters)
  - Phù hợp cho semantic similarity
  - Đã được fine-tune cho sentence embedding

### 5.2. Quá trình embedding

```
Input Text → Tokenization → BERT Encoder → Mean Pooling → Normalization → Output Vector (384D)
```

**Chi tiết:**
1. **Tokenization:** Tách văn bản thành tokens (subwords)
2. **BERT Encoder:** 6 transformer layers xử lý ngữ cảnh
3. **Mean Pooling:** Lấy trung bình các token embeddings
4. **Normalization:** Chuẩn hóa vector về độ dài = 1

---

## 6. Ưu điểm của phương pháp

### 6.1. Content-Based Filtering
✅ Không cần dữ liệu tương tác (cold start problem)
✅ Giải thích được kết quả (interpretable)
✅ Hoạt động tốt với user/item mới

### 6.2. Weighted Multi-Embedding
✅ Kết hợp nhiều khía cạnh (FULL, SKILLS, DOMAIN)
✅ Linh hoạt điều chỉnh trọng số theo domain
✅ SKILLS có trọng số cao phản ánh tầm quan trọng thực tế

### 6.3. Mean Pooling cho Skills
✅ Không phụ thuộc thứ tự skill
✅ Xử lý tốt khi số skill khác nhau
✅ Mỗi skill đóng góp công bằng

---

## 7. Tham số cấu hình

```python
# Trọng số similarity
DEFAULT_SIMILARITY_WEIGHTS = {
    "FULL": 0.2,      # Mô tả tổng quát
    "SKILLS": 0.6,    # Kỹ năng kỹ thuật (quan trọng nhất)
    "DOMAIN": 0.2     # Lĩnh vực chuyên môn
}

# Mô hình embedding
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Normalize embeddings
NORMALIZE_EMBEDDINGS = True
RENORMALIZE_AFTER_MEAN_POOLING = True
```

---

## 8. Kết luận

Hệ thống Content-Based Filtering sử dụng **similarity_score** làm chỉ số chính để đo độ phù hợp giữa Job và Freelancer. Điểm số này được tính toán dựa trên:

1. **Cosine Similarity** giữa các embedding vectors
2. **Weighted combination** của 3 thành phần (FULL, SKILLS, DOMAIN)
3. **Mean Pooling** cho skill embeddings để xử lý danh sách kỹ năng

Công thức cuối cùng:

```
similarity_score = 0.2 × cos(E_job_full, E_fr_full) 
                 + 0.6 × cos(E_job_skills, E_fr_skills)
                 + 0.2 × cos(E_job_domain, E_fr_domain)
```

Giá trị **similarity_score ∈ [0, 1]**, càng gần 1 càng phù hợp.
