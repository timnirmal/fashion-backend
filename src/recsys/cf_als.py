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
    max_users = min(len(user_ids), getattr(model, "user_factors", np.zeros((0,))).shape[0], user_items_csr.shape[0])
    for u_idx in range(max_users):
        u_id = user_ids[u_idx]
        user_row = user_items_csr[u_idx]
        ids, scores = model.recommend(u_idx, user_row, N=N, filter_already_liked_items=True)
        valid = [i for i in ids if 0 <= int(i) < len(item_ids)]
        recs[u_id] = [item_ids[int(i)] for i in valid]
    return recs


def recommend_with_scores(model: AlternatingLeastSquares, user_items_csr, user_ids: List[str], item_ids: List[str], N=200) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    max_users = min(len(user_ids), getattr(model, "user_factors", np.zeros((0,))).shape[0], user_items_csr.shape[0])
    for u_idx in range(max_users):
        u_id = user_ids[u_idx]
        user_row = user_items_csr[u_idx]
        ids, scores = model.recommend(u_idx, user_row, N=N, filter_already_liked_items=True)
        pairs = [(int(i), float(s)) for i, s in zip(ids, scores) if 0 <= int(i) < len(item_ids)]
        out[u_id] = {item_ids[i]: s for i, s in pairs}
    return out


