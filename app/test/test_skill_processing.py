import sys
from pathlib import Path
from typing import Any, cast

# Bổ sung sys.path trước khi import để tránh lỗi ModuleNotFoundError khi chạy trực tiếp
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:  # pragma: no cover - chỉ để hỗ trợ chạy trực tiếp mà không cài pytest
    import pytest
    from pytest import approx
except ModuleNotFoundError:  # Fallback khi chạy "python test_skill_processing.py"
    import types

    # import sau khi đã thêm ROOT_DIR vào sys.path
    from app.test.pytest_fallback import approx

    pytest = cast(Any, types.ModuleType("pytest"))
    pytest.approx = approx

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
    setattr(fake_ml_models, "get_embedding_model", lambda _: FakeModel())
    monkeypatch.setitem(sys.modules, "app.models.ml_models", fake_ml_models)

    emb = aggregate_skill_embedding(["ReactJS", "Node.js"], renormalize_output=True)

    # mean -> [0.5, 0.5], sau đó renorm -> [~0.707, ~0.707]
    assert emb is not None
    assert approx(emb[0], rel=1e-3) == 0.7071
    assert approx(emb[1], rel=1e-3) == 0.7071
    assert calls["normalize_embeddings"] is True


def test_aggregate_skill_embedding_no_renorm(monkeypatch):
    """Khi renormalize_output=False thì giữ đúng mean gốc."""

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            # vector không vuông góc để mean khác renorm
            return [
                [0.6, 0.8],  # đã là vector chuẩn hoá
                [0.0, 1.0],
            ]

    import sys
    import types

    fake_ml_models = types.ModuleType("app.models.ml_models")
    setattr(fake_ml_models, "get_embedding_model", lambda _: FakeModel())
    monkeypatch.setitem(sys.modules, "app.models.ml_models", fake_ml_models)

    emb = aggregate_skill_embedding(
        ["Docker", "Kubernetes"], renormalize_output=False
    )

    # mean trực tiếp: [(0.6+0)/2, (0.8+1)/2] = [0.3, 0.9]
    assert emb is not None
    assert emb == approx([0.3, 0.9], rel=1e-6)


def test_aggregate_skill_embedding_empty_after_normalize(monkeypatch):
    """Danh sách skill rỗng hoặc toàn khoảng trắng trả về None."""

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):  # pragma: no cover - không được gọi
            raise AssertionError("encode should not be called")

    import sys
    import types

    fake_ml_models = types.ModuleType("app.models.ml_models")
    setattr(fake_ml_models, "get_embedding_model", lambda _: FakeModel())
    monkeypatch.setitem(sys.modules, "app.models.ml_models", fake_ml_models)

    assert aggregate_skill_embedding([" ", ""], renormalize_output=True) is None


def test_normalize_skill_list_stability():
    """Alias + dedup + sort giúp output ổn định dù input lộn xộn."""

    skills_a = [" Node ", "ReactJS", "rest apis", "react", "NODE.js"]
    skills_b = ["react", "REST", "node js"]

    assert normalize_skill_list(skills_a) == ["nodejs", "react", "rest api"]
    # Cùng một tập, dù ít phần tử hơn vẫn cho kết quả giống nhau
    assert normalize_skill_list(skills_b) == ["nodejs", "react", "rest api"]


if __name__ == "__main__":  # pragma: no cover - hỗ trợ chạy trực tiếp bằng python
    test_file = Path(__file__).resolve()
    print(f"Chạy test thủ công cho {test_file} ...")

    try:
        import pytest

        # Gọi pytest để có log chi tiết trên console
        raise SystemExit(pytest.main([str(test_file)]))
    except ModuleNotFoundError:
        print(
            "Không tìm thấy pytest. Cài bằng `pip install -r requirements-dev.txt` "
            "hoặc chạy `pytest app/test/test_skill_processing.py -q`."
        )
        print("Các hàm test sẽ không chạy khi thiếu pytest.")
