# app/workers/train_p_match.py
"""
Train Logistic Regression cho p_match dựa trên bảng match_feature.

Output:
    models/p_match_logreg.joblib
    models/p_match_feature_columns.json
"""

import asyncio
import json
import os
import pandas as pd
from sqlalchemy import text

from app.db.session import async_session  # giữ nguyên hệ thống hiện tại của bạn


# ==========================
# 1. LOAD DATASET
# ==========================

TRAIN_QUERY = """
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
        return pd.DataFrame(rows)


# ==========================
# 2. PREPROCESS
# ==========================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import joblib


def preprocess(df: pd.DataFrame):
    id_cols = ["match_feature_id", "job_id", "freelancer_id"]
    label_col = "label"

    # one-hot region
    if "freelancer_region" in df.columns:
        df["freelancer_region"] = df["freelancer_region"].fillna("UNKNOWN")
        region_dummies = pd.get_dummies(df["freelancer_region"], prefix="region")
        df = pd.concat([df.drop(columns=["freelancer_region"]), region_dummies], axis=1)

    # bool → float
    for col in ["has_past_collaboration", "has_viewed_job"]:
        df[col] = df[col].astype(float)

    # fill NaN numeric
    numeric_cols = [
        c for c in df.columns
        if c not in id_cols + [label_col] and df[c].dtype != object
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    region_cols = [c for c in df.columns if c.startswith("region_")]
    feature_cols = numeric_cols + region_cols

    X = df[feature_cols]
    y = df[label_col].astype(int)
    return X, y, feature_cols


# ==========================
# 3. TRAIN LOGISTIC REGRESSION
# ==========================

def train_sync(df: pd.DataFrame):
    if df.empty:
        print("[p_match] dataset rỗng, dừng train.")
        return

    X, y, feature_cols = preprocess(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs"
            ))
        ]
    )

    model.fit(X_train, y_train)

    # evaluate
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    print("Validation AUC:", roc_auc_score(y_val, y_val_proba))
    print(classification_report(y_val, y_val_pred))

    # save files
    os.makedirs("app/models", exist_ok=True)

    joblib.dump(model, "app/models/p_match_logreg.joblib")
    print("Saved app/models/p_match_logreg.joblib")

    with open("app/models/p_match_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print("Saved app/models/p_match_feature_columns.json")


# ==========================
# 4. MAIN
# ==========================

async def main():
    df = await load_dataset()
    print(f"Loaded dataset: {len(df)} rows")
    train_sync(df)


if __name__ == "__main__":
    asyncio.run(main())
