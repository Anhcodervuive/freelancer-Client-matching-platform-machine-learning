# app/workers/apply_p_match.py
"""
Apply Logistic Regression model để tính p_match cho bảng match_feature.

Yêu cầu:
    - Đã train xong model:
        app/models/p_match_logreg.joblib
        app/models/p_match_feature_columns.json
    - DATABASE_URL cấu hình như app chính
"""

import asyncio
import json
from typing import List

import joblib
import pandas as pd
from sqlalchemy import text

from app.db.session import async_session  # dùng lại session của bạn

MODEL_PATH = "app/models/p_match_logreg.joblib"
FEATURES_PATH = "app/models/p_match_feature_columns.json"

BATCH_SIZE = 500

_model = None
_feature_cols: List[str] = []


# ==========================
# 1. LOAD MODEL & FEATURE COLS
# ==========================

def load_model_and_features():
    global _model, _feature_cols
    if _model is None:
        print(f"[p_match] Loading model from {MODEL_PATH} ...")
        _model = joblib.load(MODEL_PATH)

    if not _feature_cols:
        print(f"[p_match] Loading feature cols from {FEATURES_PATH} ...")
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            _feature_cols = json.load(f)

    print(f"[p_match] Model & features loaded. {len(_feature_cols)} feature columns.")


# ==========================
# 2. QUERY BATCH MATCH_FEATURE
# ==========================

MATCH_FEATURE_BATCH_QUERY = """
SELECT
  mf.id AS match_feature_id,
  mf.job_id,
  mf.freelancer_id,

  mf.similarity_score,
  mf.budget_gap,
  mf.timezone_gap_hours,
  mf.level_gap,
  mf.job_experience_level_num,
  mf.job_required_skill_count,
  mf.job_screening_question_count,
  mf.job_stats_applies,
  mf.job_stats_offers,
  mf.job_stats_accepts,
  mf.freelancer_skill_count,
  mf.freelancer_stats_applies,
  mf.freelancer_stats_offers,
  mf.freelancer_stats_accepts,
  mf.freelancer_invite_accept_rate,
  mf.freelancer_region,
  mf.skill_overlap_count,
  mf.skill_overlap_ratio,
  mf.has_past_collaboration,
  mf.past_collaboration_count,
  mf.has_viewed_job

FROM match_feature mf
WHERE mf.p_match IS NULL
ORDER BY mf.id
LIMIT :limit
"""


# ==========================
# 3. BUILD FEATURE MATRIX (giống lúc train)
# ==========================

def build_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Rebuild X đúng thứ tự feature_cols đã dùng khi train.
    Đảm bảo KHÔNG còn NaN (fill = 0.0).
    """
    if df.empty:
        return pd.DataFrame(columns=feature_cols)

    # --- 1. Xử lý region ---
    if "freelancer_region" in df.columns:
        df["freelancer_region"] = df["freelancer_region"].fillna("UNKNOWN")
        region_dummies = pd.get_dummies(df["freelancer_region"], prefix="region")
        df = pd.concat([df.drop(columns=["freelancer_region"]), region_dummies], axis=1)

    # --- 2. Bool -> float ---
    for col in ["has_past_collaboration", "has_viewed_job"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(float)

    id_cols = ["match_feature_id", "job_id", "freelancer_id"]

    # --- 3. Fill NaN cho các cột numeric đã chắc chắn là số ---
    numeric_candidate_cols = [
        c
        for c in df.columns
        if c not in id_cols and not c.startswith("region_")
    ]
    # cố gắng convert sang float trước, cái nào convert được sẽ thành numeric
    for c in numeric_candidate_cols:
        # nếu đang là object (do Decimal hoặc string), cố ép sang float
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # giờ fillna cho mọi cột numeric (kể cả region_* nếu có NaN)
    for c in df.columns:
        if c not in id_cols:
            if df[c].dtype != object:
                df[c] = df[c].fillna(0.0)

    # --- 4. Đảm bảo đủ mọi feature_col (cột thiếu -> 0.0) ---
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # --- 5. Lấy đúng thứ tự feature_cols ---
    X = df[feature_cols].astype(float)

    # --- 6. Chốt luôn: nếu còn NaN thì fill 0 & log cảnh báo ---
    if X.isna().any().any():
        bad_cols = list(X.columns[X.isna().any()])
        print("[p_match] WARNING: Found NaN in columns, filling with 0.0:", bad_cols)
        X = X.fillna(0.0)

    return X

# ==========================
# 4. MAIN APPLY LOOP
# ==========================

async def apply_p_match(batch_size: int = BATCH_SIZE):
    load_model_and_features()
    global _model, _feature_cols

    total_updated = 0

    async with async_session() as session:
        while True:
            # 1. Lấy 1 batch match_feature chưa có p_match
            result = await session.execute(
                text(MATCH_FEATURE_BATCH_QUERY),
                {"limit": batch_size},
            )
            rows = result.mappings().all()

            if not rows:
                break

            df = pd.DataFrame(rows)
            if df.empty:
                break

            # 2. Build feature matrix
            X = build_feature_matrix(df, _feature_cols)

            # 3. Predict proba
            p = _model.predict_proba(X)[:, 1]  # xác suất label = 1

            # 4. Chuẩn bị params để update
            update_params = []
            for row, prob in zip(rows, p):
                update_params.append(
                    {
                        "id": row["match_feature_id"],
                        "p_match": float(prob),
                    }
                )

            # 5. UPDATE batch
            await session.execute(
                text("UPDATE match_feature SET p_match = :p_match WHERE id = :id"),
                update_params,
            )
            await session.commit()

            batch_count = len(update_params)
            total_updated += batch_count
            print(f"[p_match] Updated {batch_count} rows (total {total_updated}).")

            # Vì WHERE p_match IS NULL, sau khi commit batch này
            # thì những row vừa update sẽ không xuất hiện ở batch sau nữa.
            # Nên không cần OFFSET.

    print(f"[p_match] Done. Total rows updated: {total_updated}")


async def main():
    await apply_p_match()


if __name__ == "__main__":
    asyncio.run(main())
