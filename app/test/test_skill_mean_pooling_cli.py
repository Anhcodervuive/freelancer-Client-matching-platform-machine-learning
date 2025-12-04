"""
Chạy thử so sánh cosine khi embed skill bằng 2 cách:
- Ghép toàn bộ list skill thành một chuỗi rồi embed một lần.
- Embed từng skill và lấy trung bình (mean pooling) như server đang dùng cho kind="SKILL".

Mục đích: nhìn nhanh xem mean pooling giúp cosine phản ánh overlap tốt hơn ra sao.

Cách chạy:
    python app/test/test_skill_mean_pooling_cli.py

Yêu cầu:
    pip install sentence-transformers
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List
from sentence_transformers import util

# Giúp chạy trực tiếp từ thư mục app/test bằng cách thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.features.skill_processing import aggregate_skill_embedding, normalize_skill_list
from app.models.ml_models import get_embedding_model


def build_skill_text(skills: List[str]) -> str:
    """Ghép list skill thành chuỗi có dấu phẩy."""

    return ", ".join(skills)


def cosine_from_list(vec1: List[float], vec2: List[float]) -> float:
    """Tính cosine similarity từ hai list số."""

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    denom = norm1 * norm2
    if denom == 0:
        return 0.0
    return float(dot / denom)


TEST_CASES: List[Dict[str, List[str]]] = [
    {
        "code": "CASE_MEAN_1",
        "description": "Stack gần như giống hệt (FE JS)",
        "job_skills": ["Node.js", "React", "TypeScript", "REST API"],
        "freelancer_skills": ["NodeJS", "ReactJS", "TypeScript", "REST APIs"],
    },
    {
        "code": "CASE_MEAN_2",
        "description": "Stack gần giống nhưng freelancer thêm vài thứ",
        "job_skills": ["Node.js", "React", "TypeScript", "REST API"],
        "freelancer_skills": ["NodeJS", "React", "TypeScript", "Docker", "PostgreSQL"],
    },
    {
        "code": "CASE_MEAN_3",
        "description": "Overlap 50% skill, 50% khác (Web vs PHP)",
        "job_skills": ["Node.js", "React", "TypeScript", "REST API"],
        "freelancer_skills": ["PHP", "Laravel", "MySQL", "jQuery"],
    },
]


def main() -> None:
    print("=== LOAD MODEL: sentence-transformers/all-MiniLM-L6-v2 ===")
    model = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")

    print("\n=== SO SÁNH COSINE KHI EMBED CẢ LIST VS MEAN POOLING TỪNG SKILL ===\n")

    for case in TEST_CASES:
        code = case["code"]
        desc = case["description"]
        job_skills = case["job_skills"]
        freelancer_skills = case["freelancer_skills"]

        print("=" * 80)
        print(f"[{code}] {desc}")
        print("-" * 80)
        print(f"Job skills       : {job_skills}")
        print(f"Freelancer skills: {freelancer_skills}")

        # 1) Ghép chuỗi rồi embed một lần
        job_text = build_skill_text(job_skills)
        free_text = build_skill_text(freelancer_skills)
        job_emb_concat = model.encode(job_text, normalize_embeddings=True)
        free_emb_concat = model.encode(free_text, normalize_embeddings=True)
        cosine_concat = float(util.cos_sim(job_emb_concat, free_emb_concat)[0][0])

        # 2) Mean pooling từng skill (có normalize + renormalize)
        job_emb_mean = aggregate_skill_embedding(
            job_skills,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            normalize=True,
            renormalize_output=True,
        )
        free_emb_mean = aggregate_skill_embedding(
            freelancer_skills,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            normalize=True,
            renormalize_output=True,
        )
        cosine_mean = cosine_from_list(job_emb_mean, free_emb_mean)

        print("\n[1] DỮ LIỆU SAU KHI NORMALIZE + SORT + DEDUP")
        print(f"  - job_norm              : {normalize_skill_list(job_skills)}")
        print(f"  - freelancer_norm       : {normalize_skill_list(freelancer_skills)}")

        print("\n[2] COSINE KHI EMBED CẢ LIST (GHÉP CHUỖI)")
        print(f"  - job_text        : \"{job_text}\"")
        print(f"  - freelancer_text : \"{free_text}\"")
        print(f"  - cosine(full-list embedding) = {cosine_concat:.4f}")

        print("\n[3] COSINE KHI MEAN POOLING TỪNG SKILL")
        print("  - Mỗi skill được embed riêng (normalize_embeddings=True), sau đó lấy trung bình và renorm")
        print(f"  - cosine(mean pooling) = {cosine_mean:.4f}")

    print("=" * 80)
    print("DONE. So sánh để thấy mean pooling giữ độ sắc nét tốt hơn khi list dài/khác thứ tự.")
    print("=" * 80)


if __name__ == "__main__":
    main()
