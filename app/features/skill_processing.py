# app/features/skill_processing.py
"""
Các helper xử lý danh sách kỹ năng trước khi embed.

- Normalize alias/viết hoa để giảm nhiễu khi so khớp.
- Dedup + sort ổn định để embedding ít lệ thuộc thứ tự do Node.js gửi lên.
- Hỗ trợ embed từng skill rồi lấy trung bình để đại diện cho toàn bộ stack.
"""

import math
from typing import Iterable, List, Optional

def normalize_skill(skill: str) -> str:
    """Chuẩn hoá tên kỹ năng để so khớp ổn định."""

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

    key = skill.strip().lower()
    return alias_map.get(key, key)


def normalize_skill_list(skills: Iterable[str]) -> List[str]:
    """Chuẩn hoá, loại bỏ skill rỗng và dedup theo thứ tự sắp xếp alpha."""

    cleaned = [normalize_skill(s) for s in skills if s and s.strip()]
    # sort để tránh model bị ảnh hưởng thứ tự danh sách đầu vào
    unique_sorted = sorted(set(cleaned))
    return unique_sorted


def aggregate_skill_embedding(
    skills: Iterable[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    renormalize_output: bool = True,
) -> Optional[List[float]]:
    """
    Embed từng skill rồi lấy trung bình (mean pooling) để đại diện toàn bộ stack.

    * `normalize=True` giữ nguyên hành vi cũ: mỗi vector con được chuẩn hoá L2.
    * `renormalize_output=True` chuẩn hoá lại vector trung bình để tránh bị "mềm" đi
      khi lấy mean (tránh cosine bị giảm vì độ dài vector nhỏ hơn 1).

    Trả về None nếu danh sách sau khi chuẩn hoá trống.
    """

    # Tránh import model nặng ở cấp module
    from app.models.ml_models import get_embedding_model

    normalized_skills = normalize_skill_list(skills)
    if not normalized_skills:
        return None

    model = get_embedding_model(model_name)
    embs = model.encode(normalized_skills, normalize_embeddings=normalize)

    # model.encode trả về numpy array; chuyển sang list để tính trung bình mà không cần numpy
    embs_list = [list(v) for v in embs]
    dim = len(embs_list[0])
    sums = [0.0] * dim
    for vec in embs_list:
        for i, val in enumerate(vec):
            sums[i] += float(val)

    mean_emb = [s / len(embs_list) for s in sums]

    if renormalize_output:
        norm = math.sqrt(sum(v * v for v in mean_emb))
        if norm > 0:
            mean_emb = [v / norm for v in mean_emb]

    return mean_emb
