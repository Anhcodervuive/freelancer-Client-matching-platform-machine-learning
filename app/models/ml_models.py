# app/models/ml_models.py
from functools import lru_cache
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


# Dùng cache để chỉ load model 1 lần
@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load và cache model embedding. Gọi lại nhiều lần cũng chỉ load 1 lần.
    """
    return SentenceTransformer(model_name)


def embed_text(
    text: Union[str, List[str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
) -> Union[List[float], List[List[float]]]:
    """
    Tính embedding cho 1 string hoặc list string.

    - text: câu / đoạn văn hoặc list các đoạn.
    - model_name: tên model (anh có thể lưu vào cột model trong bảng Embedding).
    - normalize: có chuẩn hoá vector về norm = 1 hay không (thường nên bật để cosine đơn giản).

    Trả về:
    - nếu text là str: List[float]
    - nếu text là List[str]: List[List[float]]
    """
    model = get_embedding_model(model_name)

    if isinstance(text, str):
        emb = model.encode(text, normalize_embeddings=normalize)
        return emb.tolist()
    else:
        embs = model.encode(text, normalize_embeddings=normalize)
        return [e.tolist() for e in embs]


# Dummy ranking models giữ lại (sau này mình nâng cấp)
def predict_p_match(job_embedding: List[float], freelancer_embedding: List[float]) -> float:
    """
    Ví dụ tạm: dùng cosine similarity làm p_match.
    Sau này anh thay bằng LightGBM / XGBoost.
    """
    v1 = np.array(job_embedding)
    v2 = np.array(freelancer_embedding)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    score = float(np.dot(v1, v2) / denom)
    return score


def predict_p_accept(job_embedding: List[float], freelancer_embedding: List[float]) -> float:
    """
    Ví dụ tạm: p_accept = 0.9 * p_match.
    Sau này anh train model real.
    """
    base = predict_p_match(job_embedding, freelancer_embedding)
    return float(0.9 * base)
