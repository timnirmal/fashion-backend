from typing import Dict, List
import numpy as np
from implicit.bpr import BayesianPersonalizedRanking


def train_bpr(user_items_csr, factors=128, reg=1e-3, learning_rate=0.05, epochs=100, random_state=42):
    model = BayesianPersonalizedRanking(
        factors=factors,
        regularization=reg,
        learning_rate=learning_rate,
        iterations=epochs,
        random_state=random_state,
    )
    model.fit(user_items_csr)
    return model


def recommend_users(model: BayesianPersonalizedRanking, user_items_csr, user_ids: List[str], item_ids: List[str], N=200):
    recs: Dict[str, List[str]] = {}
    for u_idx, u_id in enumerate(user_ids):
        user_row = user_items_csr[u_idx]
        ids, scores = model.recommend(u_idx, user_row, N=N, filter_already_liked_items=True)
        recs[u_id] = [item_ids[i] for i in ids]
    return recs


def recommend_with_scores(model: BayesianPersonalizedRanking, user_items_csr, user_ids: List[str], item_ids: List[str], N=200) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for u_idx, u_id in enumerate(user_ids):
        user_row = user_items_csr[u_idx]
        ids, scores = model.recommend(u_idx, user_row, N=N, filter_already_liked_items=True)
        out[u_id] = {item_ids[i]: float(s) for i, s in zip(ids, scores)}
    return out


