"""
test_skill_similarity.py

Mục đích:
- Tạo vài cặp (job_skills, freelancer_skills) dữ liệu cứng
- Tính:
    - cosine similarity khi embed CẢ LIST skill thành 1 chuỗi
    - cosine similarity theo từng skill (best-match / average)
    - các metric overlap: intersection, jaccard, coverage
- Log ra console cho bạn dễ nhìn, so sánh.

Yêu cầu:
    pip install sentence-transformers
"""

from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple
import math


# ==========================
# 1. Helper function
# ==========================

def normalize_skill(s: str) -> str:
    """
    Chuẩn hóa skill để so sánh overlap:
    - lower-case
    - strip khoảng trắng
    - map vài alias cơ bản (reactjs -> react, node.js -> nodejs, ...)
    """
    s = s.strip().lower()

    alias_map = {
        "reactjs": "react",
        "react js": "react",
        "node.js": "nodejs",
        "node js": "nodejs",
        "node": "nodejs",
        "postgres": "postgresql",
        "rest apis": "rest api",
        "rest": "rest api",
        "typescript": "ts",
        "javascript": "js",
    }

    return alias_map.get(s, s)


def compute_overlap(job_skills: List[str], free_skills: List[str]) -> Dict[str, float]:
    """
    Tính các metric overlap skill dựa trên list sau khi normalize.
    """
    job_norm = [normalize_skill(s) for s in job_skills]
    free_norm = [normalize_skill(s) for s in free_skills]

    set_job = set(job_norm)
    set_free = set(free_norm)

    inter = set_job & set_free
    union = set_job | set_free

    overlap_count = len(inter)
    jaccard = len(inter) / len(union) if union else 0.0
    job_coverage = len(inter) / len(set_job) if set_job else 0.0
    free_coverage = len(inter) / len(set_free) if set_free else 0.0

    return {
        "overlap_count": overlap_count,
        "jaccard": jaccard,
        "job_coverage": job_coverage,
        "freelancer_coverage": free_coverage,
        "intersection": list(inter),
        "job_norm": job_norm,
        "freelancer_norm": free_norm,
    }


def build_skill_text(skills: List[str]) -> str:
    """
    Ghép list skill thành 1 chuỗi để embed (cách bạn đang làm hiện tại).
    """
    return ", ".join(skills)


def cosine(a, b) -> float:
    """
    Tính cosine từ vector đã normalize (nếu dùng model.normalize_embeddings=True thì khỏi cần).
    Ở đây vẫn viết sẵn phòng khi bạn muốn reuse.
    """
    dot = float((a * b).sum())
    return dot


def per_skill_similarity(
    model: SentenceTransformer,
    job_skills: List[str],
    free_skills: List[str],
) -> Dict[str, float]:
    """
    Embed TỪNG skill riêng lẻ, rồi:
    - với mỗi job_skill: tìm max similarity với mọi free_skill
    - tính trung bình các max đó
    - cũng tính trung bình ngược lại (free -> job)
    """
    if not job_skills or not free_skills:
        return {
            "avg_best_job_to_free": 0.0,
            "avg_best_free_to_job": 0.0,
        }

    job_embs = model.encode(job_skills, normalize_embeddings=True)
    free_embs = model.encode(free_skills, normalize_embeddings=True)

    # job -> free
    best_j2f = []
    for i, j_emb in enumerate(job_embs):
        sims = util.cos_sim(j_emb, free_embs)[0]  # shape: (len(free),)
        best_sim = float(sims.max())
        best_j2f.append(best_sim)

    # free -> job
    best_f2j = []
    for i, f_emb in enumerate(free_embs):
        sims = util.cos_sim(f_emb, job_embs)[0]
        best_sim = float(sims.max())
        best_f2j.append(best_sim)

    avg_j2f = sum(best_j2f) / len(best_j2f)
    avg_f2j = sum(best_f2j) / len(best_f2j)

    return {
        "avg_best_job_to_free": avg_j2f,
        "avg_best_free_to_job": avg_f2j,
    }


# ==========================
# 2. Test cases cứng
# ==========================

TEST_CASES = [
    {
        "code": "CASE_1",
        "description": "Stack gần như giống hệt (FE JS)",
        "job_skills": ["Node.js", "React", "TypeScript", "REST API"],
        "freelancer_skills": ["NodeJS", "ReactJS", "TypeScript", "REST APIs"],
    },
    {
        "code": "CASE_2",
        "description": "Stack gần giống nhưng freelancer thêm vài thứ",
        "job_skills": ["Node.js", "React", "TypeScript", "REST API"],
        "freelancer_skills": ["NodeJS", "React", "TypeScript", "Docker", "PostgreSQL"],
    },
    {
        "code": "CASE_3",
        "description": "Overlap 50% skill, 50% khác (Web vs PHP)",
        "job_skills": ["Node.js", "React", "TypeScript", "REST API"],
        "freelancer_skills": ["PHP", "Laravel", "MySQL", "jQuery"],
    },
    {
        "code": "CASE_4",
        "description": "Flutter Mobile app, freelancer rất phù hợp",
        "job_skills": ["Flutter", "Dart", "Firebase", "REST API"],
        "freelancer_skills": ["Flutter", "Dart", "Riverpod", "REST"],
    },
    {
        "code": "CASE_5",
        "description": "AI/NLP vs Data engineer (liên quan xa)",
        "job_skills": ["Python", "NLP", "Transformers", "PyTorch"],
        "freelancer_skills": ["Python", "SQL", "Airflow", "Spark"],
    },
]


# ==========================
# 3. Main
# ==========================

def main():
    print("=== LOAD MODEL: sentence-transformers/all-MiniLM-L6-v2 ===")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("\n=== BẮT ĐẦU CHẠY CÁC CASE TEST SKILL SIMILARITY ===\n")

    for case in TEST_CASES:
        code = case["code"]
        desc = case["description"]
        job_skills = case["job_skills"]
        free_skills = case["freelancer_skills"]

        print("=" * 80)
        print(f"[{code}] {desc}")
        print("-" * 80)
        print(f"Job skills       : {job_skills}")
        print(f"Freelancer skills: {free_skills}")

        # 1) Overlap / Jaccard
        ov = compute_overlap(job_skills, free_skills)
        print("\n[1] OVERLAP / JACCARD (dựa trên list skill normalize)")
        print(f"  - job_norm              : {ov['job_norm']}")
        print(f"  - freelancer_norm       : {ov['freelancer_norm']}")
        print(f"  - intersection          : {ov['intersection']}")
        print(f"  - overlap_count         : {ov['overlap_count']}")
        print(f"  - jaccard               : {ov['jaccard']:.4f}")
        print(f"  - job_coverage          : {ov['job_coverage']:.4f}   (phần trăm skill job được cover)")
        print(f"  - freelancer_coverage   : {ov['freelancer_coverage']:.4f}   (phần trăm skill freelancer cover job)")

        # 2) Cosine khi embed cả list
        print("\n[2] COSINE(EMBED CẢ LIST SKILL THÀNH 1 CHUỖI)")

        job_text = build_skill_text(job_skills)
        free_text = build_skill_text(free_skills)
        job_emb = model.encode(job_text, normalize_embeddings=True)
        free_emb = model.encode(free_text, normalize_embeddings=True)
        cos_full = float(util.cos_sim(job_emb, free_emb)[0][0])

        print(f"  - job_text        : \"{job_text}\"")
        print(f"  - freelancer_text : \"{free_text}\"")
        print(f"  - cosine(full-list embedding) = {cos_full:.4f}")

        # 3) Cosine per-skill (best match)
        print("\n[3] COSINE(PER-SKILL, BEST MATCH)")

        per_skill = per_skill_similarity(model, job_skills, free_skills)
        print(f"  - avg_best_job_to_free  (mỗi skill của job so với best skill của freelancer): {per_skill['avg_best_job_to_free']:.4f}")
        print(f"  - avg_best_free_to_job  (mỗi skill của freelancer so với best skill của job): {per_skill['avg_best_free_to_job']:.4f}")

        print("\n=> Gợi ý:")
        print("   - similarity_score (kiểu bạn đang lưu) có thể dùng cosine(full-list).")
        print("   - nhưng thêm các feature overlap & per-skill cosine sẽ giúp đánh giá 'match' chuẩn hơn.")
        print()

    print("=" * 80)
    print("DONE. Bạn xem log để cảm nhận việc cosine ~0.5 vẫn có thể là match tốt khi overlap nhiều.")
    print("=" * 80)


if __name__ == "__main__":
    main()
