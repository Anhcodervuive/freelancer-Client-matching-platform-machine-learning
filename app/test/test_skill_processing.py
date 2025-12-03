import sys
from pathlib import Path

import pytest

# Cho phép import module app khi chạy trực tiếp file test
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.features.skill_processing import (
    aggregate_skill_embedding,
    normalize_skill,
    normalize_skill_list,
)


def test_normalize_skill_aliases():
    assert normalize_skill("ReactJS") == "react"
    assert normalize_skill("Node.js") == "nodejs"
    assert normalize_skill("REST APIs") == "rest api"
    assert normalize_skill("TypeScript") == "ts"


def test_normalize_skill_list_dedup_and_sort():
    skills = [" ReactJS ", "node js", "React", "rest", ""]
    normalized = normalize_skill_list(skills)
    # lower + alias + dedup + sort
    assert normalized == ["nodejs", "react", "rest api"]


def test_aggregate_skill_embedding_mean_and_renorm(monkeypatch):
    calls = {}

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            calls["normalize_embeddings"] = normalize_embeddings
            # trả về 2 vector vuông góc để dễ kiểm tra mean + renorm
            return [
                [1.0, 0.0],
                [0.0, 1.0],
            ]

    import sys
    import types

    fake_ml_models = types.ModuleType("app.models.ml_models")
    fake_ml_models.get_embedding_model = lambda _: FakeModel()
    monkeypatch.setitem(sys.modules, "app.models.ml_models", fake_ml_models)

    emb = aggregate_skill_embedding(["ReactJS", "Node.js"], renormalize_output=True)

    # mean -> [0.5, 0.5], sau đó renorm -> [~0.707, ~0.707]
    assert pytest.approx(emb[0], rel=1e-3) == 0.7071
    assert pytest.approx(emb[1], rel=1e-3) == 0.7071
    assert calls["normalize_embeddings"] is True
