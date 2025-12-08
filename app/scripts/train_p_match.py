# scripts/train_p_match.py

import asyncio
import json
from typing import List

import pandas as pd
from sqlalchemy import text

from app.db.session import async_session  # dùng đúng session hiện tại của bạn


# ----------------- STEP 1: Lấy dataset từ DB qua SQLAlchemy async ----------------- #

TRAIN_QUERY = """
SELECT
  mf.id                         AS match_feature_id,
  mf.job_id                     AS job_id,
  mf.freelancer_id              AS freelancer_id,

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
  mf.has_viewed_job,

  CASE
    WHEN c.id IS NOT NULL THEN 1
    WHEN jo_accepted.id IS NOT NULL THEN 1
    ELSE 0
  END AS label
FROM match_feature mf
LEFT JOIN contract c
  ON c.job_post_id = mf.job_id
 AND c.freelancer_id = mf.freelancer_id
LEFT JOIN job_offer jo_all
  ON jo_all.job_id = mf.job_id
 AND jo_all.freelancer_id = mf.freelancer_id
 AND jo_all.is_deleted = 0
LEFT JOIN job_offer jo_accepted
  ON jo_accepted.job_id = mf.job_id
 AND jo_accepted.freelancer_id = mf.freelancer_id
 AND jo_accepted.status = 'ACCEPTED'
 AND jo_accepted.is_deleted = 0
LEFT JOIN job_proposal p
  ON p.job_id = mf.job_id
 AND p.freelancer_id = mf.freelancer_id
WHERE
  c.id IS NOT NULL
  OR p.id IS NOT NULL
  OR jo_all.id IS NOT NULL
"""


async def load_dataset() -> pd.DataFrame:
    async with async_session() as session:
        result = await session.execute(text(TRAIN_QUERY))
        rows = result.mappings().all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)


# ----------------- STEP 2: Tiền xử lý & train Logistic Regression ----------------- #

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os
import joblib


def preprocess(df: pd.DataFrame):
    id_cols = ["match_feature_id", "job_id", "freelancer_id"]
    label_col = "label"

    # Encode freelancer_region → one-hot
    if "freelancer_region" in df.columns:
        df["freelancer_region"] = df["freelancer_region"].fillna("UNKNOWN")
        region_dummies = pd.get_dummies(
            df["freelancer_region"], prefix="region", dummy_na=False
        )
        df = pd.concat([df.drop(columns=["freelancer_region"]), region_dummies], axis=1)

    # Bool → float
    for col in ["has_past_collaboration", "has_viewed_job"]:
        if col in df.columns:
            df[col] = df[col].astype("float").fillna(0.0)

    # Numeric columns (trừ id, label)
    numeric_cols = [
        c for c in df.columns
        if c not in id_cols + [label_col] and df[c].dtype != "object"
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Thêm các cột region_* vào feature list
    region_cols = [c for c in df.columns if c.startswith("region_")]
    feature_cols = numeric_cols + region_cols

    X = df[feature_cols]
    y = df[label_col].astype(int)
    return X, y, feature_cols


def train_sync(df: pd.DataFrame):
    if df.empty:
        print("No data to train p_match (dataset rỗng).")
        return

    X, y, feature_cols = preprocess(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: scale (StandardScaler) + Logistic Regression
    # class_weight="balanced" để xử lý mất cân bằng 0/1
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Predict proba để tính AUC & metrics
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_val_proba)
    print("Validation AUC (Logistic Regression):", auc)
    print(classification_report(y_val, y_val_pred))

    # Lưu model & feature_cols vào folder models/
    os.makedirs("models", exist_ok=True)

    # Lưu pipeline logistic regression
    joblib.dump(pipeline, "models/p_match_logreg.joblib")
    print("Saved models/p_match_logreg.joblib")

    # Lưu danh sách feature_cols để lúc apply đảm bảo đúng cột/đúng thứ tự
    with open("models/p_match_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print("Saved models/p_match_feature_columns.json")


async def main():
    df = await load_dataset()
    print(f"Loaded dataset: {len(df)} rows")
    train_sync(df)


if __name__ == "__main__":
    asyncio.run(main())
