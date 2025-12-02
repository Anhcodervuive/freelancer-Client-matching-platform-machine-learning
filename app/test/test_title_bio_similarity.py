"""
test_title_bio_similarity.py

Mục đích:
- Test mức độ tương đồng giữa:
    + Freelancer: title + bio
    + Job: title + description
- Dùng dữ liệu cứng, nhiều kịch bản:
    + Match rất sát
    + Match cùng stack nhưng khác level
    + Cùng mảng khác stack
    + Khác mảng hoàn toàn
- Log:
    + Cosine(full_text_freelancer, full_text_job)
    + Cosine(title_freelancer, title_job)
    + Cosine(bio_freelancer, description_job)

Yêu cầu:
    pip install sentence-transformers
"""

from sentence_transformers import SentenceTransformer, util
from typing import Dict, List


# ==========================
# Helper
# ==========================

def cosine_sim(model, a: str, b: str) -> float:
    emb_a = model.encode(a, normalize_embeddings=True)
    emb_b = model.encode(b, normalize_embeddings=True)
    sim = float(util.cos_sim(emb_a, emb_b)[0][0])
    return sim


# ==========================
# Test Cases
# ==========================

TEST_CASES: List[Dict] = [
    # 1 — React FE match gần như perfect
    {
        "code": "FE_REACT_STRONG_MATCH",
        "freelancer_title": "Senior Frontend Engineer (React/TypeScript)",
        "freelancer_bio": (
            "5+ years building SPA with React, Next.js, TypeScript. "
            "Strong experience in reusable UI components, design systems, "
            "performance optimization and responsive layout. Used to working in agile teams."
        ),
        "job_title": "React Frontend Developer (TypeScript, Next.js)",
        "job_description": (
            "We are looking for a frontend developer with solid experience in React and TypeScript. "
            "You will build and maintain SPA using Next.js, work closely with designers to "
            "implement pixel-perfect UI and reusable components."
        ),
    },

    # 2 — FE mid-level vs job junior
    {
        "code": "FE_REACT_LEVEL_DIFF",
        "freelancer_title": "Mid-level Frontend Developer (React)",
        "freelancer_bio": (
            "3 years of experience with React and Redux. Comfortable with REST API integration, "
            "form handling, and responsive UI. Some exposure to TypeScript."
        ),
        "job_title": "Junior React Developer",
        "job_description": (
            "We need a junior developer to maintain existing React components, fix bugs, "
            "and implement small UI features. Basic knowledge of JavaScript, React, and CSS required."
        ),
    },

    # 3 — FE React vs job Vue (cùng FE, khác stack)
    {
        "code": "FE_FRAMEWORK_DIFF",
        "freelancer_title": "Frontend Engineer (React/TypeScript)",
        "freelancer_bio": (
            "Experience building web apps with React and TypeScript, integrating REST APIs, "
            "working with modern tooling like Vite and Webpack."
        ),
        "job_title": "Vue.js Frontend Developer",
        "job_description": (
            "We are looking for a frontend developer with production experience in Vue.js and Vuex. "
            "You will work on a large SPA and help refactor legacy components."
        ),
    },

    # 4 — Mobile Flutter match tốt
    {
        "code": "MOBILE_FLUTTER_MATCH",
        "freelancer_title": "Flutter Mobile Developer",
        "freelancer_bio": (
            "Develop cross-platform mobile apps using Flutter and Dart. "
            "Experience with state management (Provider, Bloc), Firebase integration, "
            "authentication, and in-app purchases."
        ),
        "job_title": "Flutter Developer (iOS/Android)",
        "job_description": (
            "We need a Flutter developer to build and maintain mobile apps for both iOS and Android. "
            "Must be familiar with Dart, state management patterns, and Firebase backend services."
        ),
    },

    # 5 — Backend Node vs job Node (match tốt)
    {
        "code": "BACKEND_NODE_MATCH",
        "freelancer_title": "Backend Developer (Node.js/Express)",
        "freelancer_bio": (
            "Building RESTful APIs using Node.js, Express, and MongoDB. "
            "Familiar with authentication, authorization, JWT, and writing unit tests."
        ),
        "job_title": "Backend Engineer Node.js",
        "job_description": (
            "We are hiring a backend engineer to design and implement REST APIs using Node.js and Express. "
            "You will work with MongoDB and integrate third-party services."
        ),
    },

    # 6 — Backend Laravel vs job Node (cùng backend, khác stack)
    {
        "code": "BACKEND_DIFF_STACK",
        "freelancer_title": "PHP Laravel Backend Developer",
        "freelancer_bio": (
            "4 years developing REST APIs with Laravel, MySQL, and Redis. "
            "Strong in authentication, queues, and building admin dashboards."
        ),
        "job_title": "Node.js Backend Engineer",
        "job_description": (
            "We require a Node.js developer to build API services with Express and MongoDB. "
            "Experience in JavaScript and asynchronous programming is required."
        ),
    },

    # 7 — Data Engineer vs Data Engineer
    {
        "code": "DATA_ENGINEER_MATCH",
        "freelancer_title": "Data Engineer (Airflow, BigQuery)",
        "freelancer_bio": (
            "Designing ETL pipelines using Airflow, building data warehouse on BigQuery, "
            "optimizing SQL queries and managing data schemas."
        ),
        "job_title": "Senior Data Engineer",
        "job_description": (
            "We are looking for a senior data engineer to build and maintain ETL pipelines, "
            "work with Airflow, and design data models for analytics."
        ),
    },

    # 8 — ML Engineer vs Data Analyst (cùng data, khác trọng tâm)
    {
        "code": "ML_VS_ANALYST",
        "freelancer_title": "Machine Learning Engineer",
        "freelancer_bio": (
            "Train and deploy ML models using Python, scikit-learn, and PyTorch. "
            "Experience building recommendation systems and classification models."
        ),
        "job_title": "Business Data Analyst",
        "job_description": (
            "You will build dashboards, analyze business metrics, and generate insights "
            "using SQL, Excel, and BI tools."
        ),
    },

    # 9 — DevOps vs Backend (liên quan phần nào)
    {
        "code": "DEVOPS_VS_BACKEND",
        "freelancer_title": "DevOps Engineer (AWS, Docker, CI/CD)",
        "freelancer_bio": (
            "Managing cloud infrastructure on AWS, building CI/CD pipelines, "
            "working with Docker, Kubernetes, and monitoring systems."
        ),
        "job_title": "Backend Developer (REST API)",
        "job_description": (
            "Develop REST APIs, work with relational databases, and collaborate with DevOps "
            "to deploy services to cloud infrastructure."
        ),
    },

    # 10 — UI/UX vs Graphic Design job (gần nhưng khác)
    {
        "code": "UIUX_VS_GRAPHIC",
        "freelancer_title": "UI/UX Designer",
        "freelancer_bio": (
            "Conduct user research, create wireframes, prototypes, and design system using Figma. "
            "Focus on usability and user-centered design."
        ),
        "job_title": "Graphic Designer for Marketing",
        "job_description": (
            "Design banners, social media posts, and marketing materials using Photoshop and Illustrator."
        ),
    },

    # 11 — Full mismatch: ML vs Accounting
    {
        "code": "TOTAL_MISMATCH",
        "freelancer_title": "Machine Learning Engineer (Computer Vision)",
        "freelancer_bio": (
            "Experience training deep learning models for image classification and object detection "
            "using PyTorch and TensorFlow."
        ),
        "job_title": "Senior Accountant",
        "job_description": (
            "We need an accountant with experience in financial reporting, tax, and auditing."
        ),
    },
]


# ==========================
# MAIN
# ==========================

def main():
    print("=== Load model: sentence-transformers/all-MiniLM-L6-v2 ===")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("\n=== TITLE + BIO vs JOB TITLE + DESCRIPTION SIMILARITY TEST ===\n")

    for case in TEST_CASES:
        print("=" * 100)
        print(f"[{case['code']}] Test Case")
        print("-" * 100)

        f_title = case["freelancer_title"]
        f_bio = case["freelancer_bio"]
        j_title = case["job_title"]
        j_desc = case["job_description"]

        f_full = f"{f_title}. {f_bio}"
        j_full = f"{j_title}. {j_desc}"

        print("Freelancer:")
        print(f"  Title : {f_title}")
        print(f"  Bio   : {f_bio}")
        print("\nJob:")
        print(f"  Title       : {j_title}")
        print(f"  Description : {j_desc}")

        # 1. Full text similarity (gộp title+bio vs title+desc)
        sim_full = cosine_sim(model, f_full, j_full)

        # 2. Title vs title
        sim_title = cosine_sim(model, f_title, j_title)

        # 3. Bio vs description
        sim_bio_desc = cosine_sim(model, f_bio, j_desc)

        print("\n[1] Cosine similarity (FULL: freelancer_title+bio  vs  job_title+description)")
        print(f"    → similarity = {sim_full:.4f}")

        print("\n[2] Cosine similarity (TITLE  vs  TITLE)")
        print(f"    → similarity = {sim_title:.4f}")

        print("\n[3] Cosine similarity (BIO  vs  JOB_DESCRIPTION)")
        print(f"    → similarity = {sim_bio_desc:.4f}")

        # Gợi ý cách map về p_match_title_bio nếu sau này bạn cần
        # vd: p_match_text = 0.7 * sim_full + 0.2 * sim_title + 0.1 * sim_bio_desc
        p_match_text = 0.7 * sim_full + 0.2 * sim_title + 0.1 * sim_bio_desc
        print("\n[*] Gợi ý score tổng hợp (p_match_text ~ weighted):")
        print(f"    p_match_text ≈ {p_match_text:.4f}")

        print()  # dòng trống cho dễ đọc


if __name__ == "__main__":
    main()
