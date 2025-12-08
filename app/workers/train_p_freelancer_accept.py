# app/workers/train_p_freelancer_accept.py

"""
Script build dataset + train Logistic Regression cho p_freelancer_accept.

Feature X (hiện tại ~18 chiều), lấy từ bảng match_feature:

    [
        similarity_score,
        budget_gap,
        timezone_gap_hours,
        level_gap,

        job_experience_level_num,
        job_required_skill_count,
        job_screening_question_count,
        job_stats_applies,
        job_stats_offers,
        job_stats_accepts,

        freelancer_skill_count,
        freelancer_stats_applies,
        freelancer_stats_offers,
        freelancer_stats_accepts,
        freelancer_invite_accept_rate,

        skill_overlap_count,
        skill_overlap_ratio,

        has_past_collaboration,   # 0/1
        past_collaboration_count,
        has_viewed_job,           # 0/1
    ]
"""

import asyncio
from typing import List, Dict, Any

import pandas as pd
from sqlalchemy import text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.db.session import async_session
from app.models.ml_models import P_FREELANCER_MODEL_PATH
import joblib


async def build_dataset_df() -> pd.DataFrame:
    async with async_session() as session:
        sql = text(
            """
            SELECT
                ji.job_id,
                ji.freelancer_id,
                ji.status,

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

                mf.skill_overlap_count,
                mf.skill_overlap_ratio,

                mf.has_past_collaboration,
                mf.past_collaboration_count,
                mf.has_viewed_job
            FROM job_invitation ji
            JOIN match_feature mf
              ON mf.job_id = ji.job_id
             AND mf.freelancer_id = ji.freelancer_id
            WHERE ji.status IN ('ACCEPTED', 'DECLINED', 'EXPIRED')
            """
        )

        rows = (await session.execute(sql)).mappings().all()

        data: List[Dict[str, Any]] = []
        for r in rows:
            status = (r["status"] or "").upper()
            label = 1 if status == "ACCEPTED" else 0

            def f(name: str, default: float = 0.0) -> float:
                v = r.get(name)
                # tránh None
                return float(v) if v is not None else default

            def b(name: str) -> int:
                v = r.get(name)
                # convert bool/None -> 0/1
                return 1 if v else 0

            data.append(
                {
                    "job_id": r["job_id"],
                    "freelancer_id": r["freelancer_id"],
                    "label": label,

                    # ---- core similarity / gap ----
                    "similarity_score": f("similarity_score"),
                    "budget_gap": f("budget_gap"),
                    "timezone_gap_hours": f("timezone_gap_hours"),
                    "level_gap": f("level_gap"),

                    # ---- job features ----
                    "job_experience_level_num": f("job_experience_level_num"),
                    "job_required_skill_count": f("job_required_skill_count"),
                    "job_screening_question_count": f("job_screening_question_count"),
                    "job_stats_applies": f("job_stats_applies"),
                    "job_stats_offers": f("job_stats_offers"),
                    "job_stats_accepts": f("job_stats_accepts"),

                    # ---- freelancer features ----
                    "freelancer_skill_count": f("freelancer_skill_count"),
                    "freelancer_stats_applies": f("freelancer_stats_applies"),
                    "freelancer_stats_offers": f("freelancer_stats_offers"),
                    "freelancer_stats_accepts": f("freelancer_stats_accepts"),
                    "freelancer_invite_accept_rate": f("freelancer_invite_accept_rate"),

                    # ---- pairwise features ----
                    "skill_overlap_count": f("skill_overlap_count"),
                    "skill_overlap_ratio": f("skill_overlap_ratio"),
                    "has_past_collaboration": b("has_past_collaboration"),
                    "past_collaboration_count": f("past_collaboration_count"),
                    "has_viewed_job": b("has_viewed_job"),
                }
            )

        df = pd.DataFrame(data)
        return df


async def main() -> None:
    df = await build_dataset_df()
    if df.empty:
        print("[train_p_freelancer_accept] Dataset rỗng, không train được.")
        return

    csv_path = "dataset_p_freelancer_accept.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[train] Saved {len(df)} rows to {csv_path}")

    feature_cols = [
        "similarity_score",
        "budget_gap",
        "timezone_gap_hours",
        "level_gap",

        "job_experience_level_num",
        "job_required_skill_count",
        "job_screening_question_count",
        "job_stats_applies",
        "job_stats_offers",
        "job_stats_accepts",

        "freelancer_skill_count",
        "freelancer_stats_applies",
        "freelancer_stats_offers",
        "freelancer_stats_accepts",
        "freelancer_invite_accept_rate",

        "skill_overlap_count",
        "skill_overlap_ratio",
        "has_past_collaboration",
        "past_collaboration_count",
        "has_viewed_job",
    ]

    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("[train] Classification report:")
    print(classification_report(y_test, y_pred))

    P_FREELANCER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, P_FREELANCER_MODEL_PATH)
    print(f"[train] Saved model to {P_FREELANCER_MODEL_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
