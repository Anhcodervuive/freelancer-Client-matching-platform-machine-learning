# app/models/ml_models.py
import numpy as np

def predict_p_match(job_embedding, freelancer_embedding):
    # dummy example: cosine similarity
    vec1 = np.array(job_embedding)
    vec2 = np.array(freelancer_embedding)
    score = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return float(score)

def predict_p_accept(job_embedding, freelancer_embedding):
    # dummy example: random small variation
    return predict_p_match(job_embedding, freelancer_embedding) * 0.9
