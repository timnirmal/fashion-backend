from typing import Dict, List, Tuple
import numpy as np
try:
    import xgboost as xgb
except Exception:
    xgb = None


def build_feature_matrix(
    user_ids: List[str],
    candidates: Dict[str, List[str]],
    cf_scores: Dict[str, Dict[str, float]] | None,
    cb_tfidf_scores: Dict[str, Dict[str, float]] | None,
    cb_hash_scores: Dict[str, Dict[str, float]] | None,
    pop: Dict[str, float] | None,
    item_numeric: Dict[str, List[float]] | None = None,
):
    X_list = []
    y_list = []
    qid = []
    item_list = []
    for u in user_ids:
        items = candidates.get(u, [])
        if not items:
            continue
        for it in items:
            row = [
                (cf_scores or {}).get(u, {}).get(it, 0.0),
                (cb_tfidf_scores or {}).get(u, {}).get(it, 0.0),
                (cb_hash_scores or {}).get(u, {}).get(it, 0.0),
                np.log1p((pop or {}).get(it, 0.0)),
            ]
            if item_numeric and it in item_numeric:
                row.extend(item_numeric[it])
            X_list.append(row)
            # y to be filled by caller (0/1 rel labels)
            y_list.append(0.0)
            item_list.append(it)
        qid.append(len(items))
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y, qid, item_list


def train_xgb_ranker(X: np.ndarray, y: np.ndarray, qid: List[int]):
    if xgb is None:
        raise ImportError("xgboost is unavailable; skipping LTR training")
    group = qid
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        learning_rate=0.1,
        n_estimators=200,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
    )
    model.fit(X, y, group=group)
    return model


