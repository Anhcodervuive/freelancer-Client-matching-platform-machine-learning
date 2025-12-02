"""
test_domain_similarity.py

Mục đích:
- Test mức độ tương đồng giữa DOMAIN / INDUSTRY / FIELD (lĩnh vực)
- Dùng dữ liệu cứng nhiều ngành nghề khác nhau
- So sánh cosine(full text) + cosine per-keyword
- Log kết quả rõ ràng để hiểu vì sao similarity có thể chỉ 0.5–0.7

Yêu cầu:
    pip install sentence-transformers
"""

from sentence_transformers import SentenceTransformer, util
from typing import List, Dict


# ==========================
# Helper
# ==========================

def cosine_full(model, a: str, b: str) -> float:
    a_emb = model.encode(a, normalize_embeddings=True)
    b_emb = model.encode(b, normalize_embeddings=True)
    sim = float(util.cos_sim(a_emb, b_emb)[0][0])
    return sim


def cosine_keywords(model, kw1: List[str], kw2: List[str]) -> Dict[str, float]:
    if not kw1 or not kw2:
        return {"avg_best_1_to_2": 0.0, "avg_best_2_to_1": 0.0}

    e1 = model.encode(kw1, normalize_embeddings=True)
    e2 = model.encode(kw2, normalize_embeddings=True)

    # kw1 -> kw2
    best12 = []
    for v1 in e1:
        sims = util.cos_sim(v1, e2)[0]
        best12.append(float(sims.max()))

    # kw2 -> kw1
    best21 = []
    for v2 in e2:
        sims = util.cos_sim(v2, e1)[0]
        best21.append(float(sims.max()))

    return {
        "avg_best_1_to_2": sum(best12) / len(best12),
        "avg_best_2_to_1": sum(best21) / len(best21),
    }


# ==========================
# Domain Test Cases
# ==========================

TEST_DOMAINS = [
    # 1 — Web
    {
        "code": "WEB_FE",
        "job_domain": "Frontend Web Development",
        "job_keywords": ["React", "TypeScript", "SPA", "UI Development"],
        "freelancer_domain": "Frontend Engineer (React/Next.js)",
        "freelancer_keywords": ["ReactJS", "NextJS", "UI Engineering", "JavaScript"],
    },

    # 2 — Mobile
    {
        "code": "MOBILE_FLUTTER",
        "job_domain": "Mobile Development with Flutter",
        "job_keywords": ["Flutter", "Dart", "Mobile UI", "Firebase"],
        "freelancer_domain": "Cross-platform Mobile Developer",
        "freelancer_keywords": ["Flutter", "Dart", "iOS", "Android"],
    },

    # 3 — Data Engineer
    {
        "code": "DATA_ENGINEER",
        "job_domain": "Data Engineering / ETL Pipelines / Big Data",
        "job_keywords": ["ETL", "Airflow", "SQL", "Big Data"],
        "freelancer_domain": "Backend Developer (Laravel + MySQL)",
        "freelancer_keywords": ["MySQL", "PHP", "REST API"],
    },

    # 4 — AI / NLP
    {
        "code": "AI_NLP",
        "job_domain": "AI / Natural Language Processing",
        "job_keywords": ["NLP", "Transformers", "BERT", "PyTorch"],
        "freelancer_domain": "Machine Learning Engineer",
        "freelancer_keywords": ["ML", "Python", "Tensorflow", "Deep Learning"],
    },

    # 5 — Data Science
    {
        "code": "DATA_SCIENCE",
        "job_domain": "Data Science & Predictive Analytics",
        "job_keywords": ["Statistics", "Machine Learning", "Modeling"],
        "freelancer_domain": "Business Data Analyst",
        "freelancer_keywords": ["Excel", "Power BI", "SQL", "Analysis"],
    },

    # 6 — Cloud
    {
        "code": "CLOUD_ENGINEER",
        "job_domain": "Cloud Infrastructure & DevOps",
        "job_keywords": ["AWS", "Terraform", "CI/CD"],
        "freelancer_domain": "AWS Cloud Engineer",
        "freelancer_keywords": ["AWS", "EC2", "Lambda", "Terraform"],
    },

    # 7 — Cybersecurity
    {
        "code": "CYBERSEC",
        "job_domain": "Cybersecurity & Threat Detection",
        "job_keywords": ["PenTest", "SIEM", "Security Audit"],
        "freelancer_domain": "Backend Developer",
        "freelancer_keywords": ["NodeJS", "REST"],
    },

    # 8 — Game Development
    {
        "code": "GAME_DEV",
        "job_domain": "Game Development (Unity/C#)",
        "job_keywords": ["Unity", "C#", "Game Physics"],
        "freelancer_domain": "Mobile Developer",
        "freelancer_keywords": ["Java", "Android"],
    },

    # 9 — Blockchain
    {
        "code": "BLOCKCHAIN",
        "job_domain": "Blockchain & Smart Contracts",
        "job_keywords": ["Solidity", "Web3", "Ethereum"],
        "freelancer_domain": "Fullstack Web Developer",
        "freelancer_keywords": ["NodeJS", "React"],
    },

    # 10 — Embedded
    {
        "code": "EMBEDDED",
        "job_domain": "Embedded Systems Engineering",
        "job_keywords": ["C", "Microcontroller", "RTOS"],
        "freelancer_domain": "IoT Developer",
        "freelancer_keywords": ["Arduino", "IoT", "Sensors"],
    },

    # 11 — QA
    {
        "code": "QA_TEST",
        "job_domain": "Software Testing / QA Automation",
        "job_keywords": ["Selenium", "Test Automation"],
        "freelancer_domain": "Manual Tester",
        "freelancer_keywords": ["Testing", "Bug Tracking"],
    },

    # 12 — UI/UX
    {
        "code": "UI_UX",
        "job_domain": "UI/UX Product Design",
        "job_keywords": ["Figma", "User Research", "Prototyping"],
        "freelancer_domain": "Graphic Designer",
        "freelancer_keywords": ["Photoshop", "Illustrator"],
    },

    # 13 — PM
    {
        "code": "PRODUCT_MANAGER",
        "job_domain": "Product Management",
        "job_keywords": ["Roadmap", "Agile", "Stakeholder"],
        "freelancer_domain": "Scrum Master",
        "freelancer_keywords": ["Agile", "Scrum"],
    },

    # 14 — Business Analyst
    {
        "code": "BA",
        "job_domain": "Business Analysis",
        "job_keywords": ["Requirement", "Process Mapping"],
        "freelancer_domain": "Junior Data Analyst",
        "freelancer_keywords": ["Excel", "SQL"],
    },

    # 15 — Fintech
    {
        "code": "FINTECH",
        "job_domain": "FinTech / Payment System",
        "job_keywords": ["Payment", "PCI DSS", "Banking"],
        "freelancer_domain": "Fullstack Developer",
        "freelancer_keywords": ["NodeJS", "React"],
    },

    # 16 — Healthcare IT
    {
        "code": "HEALTHCARE",
        "job_domain": "Healthcare IT Systems (HL7, FHIR)",
        "job_keywords": ["HL7", "FHIR", "Medical Data"],
        "freelancer_domain": "Python Developer",
        "freelancer_keywords": ["Python", "API", "FastAPI"],
    },

    # 17 — Marketing
    {
        "code": "MARKETING",
        "job_domain": "Digital Marketing",
        "job_keywords": ["SEO", "Content Marketing"],
        "freelancer_domain": "Copywriter",
        "freelancer_keywords": ["Writing", "Content"],
    },

    # 18 — Hardcore mismatch
    {
        "code": "MISMATCH",
        "job_domain": "Robotics Engineering",
        "job_keywords": ["Robotics", "ROS", "Sensors"],
        "freelancer_domain": "Front-end Vue Developer",
        "freelancer_keywords": ["VueJS", "HTML", "CSS"],
    },
]


# ==========================
# MAIN
# ==========================

def main():
    print("=== Load model: all-MiniLM-L6-v2 ===")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("\n=== DOMAIN SIMILARITY TEST ===\n")

    for case in TEST_DOMAINS:
        print("=" * 80)
        print(f"[{case['code']}] Domain Test Case")
        print("-" * 80)

        jd = case["job_domain"]
        fd = case["freelancer_domain"]
        kw1 = case["job_keywords"]
        kw2 = case["freelancer_keywords"]

        print(f"Job domain        : {jd}")
        print(f"Freelancer domain : {fd}")
        print(f"Job keywords      : {kw1}")
        print(f"Freelancer keywords: {kw2}")

        sim_full = cosine_full(model, jd, fd)
        sim_kw = cosine_keywords(model, kw1, kw2)

        print("\n[1] Cosine (full domain text)")
        print(f"    → similarity = {sim_full:.4f}")

        print("\n[2] Cosine (keyword best match)")
        print(f"    avg_best_job_to_free : {sim_kw['avg_best_1_to_2']:.4f}")
        print(f"    avg_best_free_to_job : {sim_kw['avg_best_2_to_1']:.4f}")

        print()


if __name__ == "__main__":
    main()
