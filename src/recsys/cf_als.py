from typing import Dict, List, Tuple
import numpy as np
from implicit.als import AlternatingLeastSquares


def train_als(confidence_csr, factors=128, reg=0.05, alpha=20.0, iterations=30, use_cg=True, random_state=42):
    # implicit expects item-user CSR for training
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iterations,
        use_cg=use_cg,
        random_state=random_state,
    )
    # Confidence matrix
    Cui = confidence_csr.T.tocsr() * alpha
    model.fit(Cui)
    return model


def recommend_users(model: AlternatingLeastSquares, user_items_csr, user_ids: List[str], item_ids: List[str], N=200):
    recs: Dict[str, List[str]] = {}
    for u_idx, u_id in enumerate(user_ids):
        user_row = user_items_csr[u_idx]
        ids, scores = model.recommend(u_idx, user_row, N=N, filter_already_liked_items=True)
        recs[u_id] = [item_ids[i] for i in ids]
    return recs


def recommend_with_scores(model: AlternatingLeastSquares, user_items_csr, user_ids: List[str], item_ids: List[str], N=200) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for u_idx, u_id in enumerate(user_ids):
        user_row = user_items_csr[u_idx]
        ids, scores = model.recommend(u_idx, user_row, N=N, filter_already_liked_items=True)
        out[u_id] = {item_ids[i]: float(s) for i, s in zip(ids, scores)}
    return out


