from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import normalize
from scipy import sparse


def build_item_text(products: pd.DataFrame) -> pd.Series:
    def join_tags(x):
        try:
            import ast

            lst = ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
            return " ".join(map(str, lst))
        except Exception:
            return ""

    text = (
        products[["name", "short_description", "description", "tags", "search_keywords"]]
        .fillna("")
        .assign(tags=lambda d: d["tags"].apply(join_tags))
        .assign(search_keywords=lambda d: d["search_keywords"].apply(join_tags))
    )
    return (
        text["name"].astype(str)
        + " \n"
        + text["short_description"].astype(str)
        + " \n"
        + text["description"].astype(str)
        + " \n"
        + text["tags"].astype(str)
        + " \n"
        + text["search_keywords"].astype(str)
    )


def fit_tfidf(corpus: pd.Series):
    vec = TfidfVectorizer(
        max_features=50000,
        min_df=2,
        ngram_range=(1, 2),
        analyzer="word",
    )
    X = vec.fit_transform(corpus)
    X = normalize(X, norm="l2")
    return vec, X


def transform_tfidf(vec: TfidfVectorizer, corpus: pd.Series):
    X = vec.transform(corpus)
    return normalize(X, norm="l2")


def fit_hashing(corpus: pd.Series):
    vec = HashingVectorizer(n_features=2**18, alternate_sign=False, analyzer="word", ngram_range=(1, 2))
    X = vec.transform(corpus)
    X = normalize(X, norm="l2")
    return vec, X


def build_user_profiles(user_items: Dict[str, List[str]], item_index: Dict[str, int], item_matrix: sparse.csr_matrix, item_weights: Dict[str, float] | None = None) -> Dict[str, np.ndarray]:
    profiles: Dict[str, np.ndarray] = {}
    for u, items in user_items.items():
        idxs = [item_index[i] for i in items if i in item_index]
        if not idxs:
            continue
        mat = item_matrix[idxs]
        if item_weights:
            ws = np.array([item_weights.get(i, 1.0) for i in items if i in item_index])[:, None]
            vec = (mat.multiply(ws)).mean(axis=0)
        else:
            vec = mat.mean(axis=0)
        profiles[u] = np.asarray(vec).ravel()
    return profiles


def recommend_by_profile(user_profiles: Dict[str, np.ndarray], item_matrix: sparse.csr_matrix, item_ids: List[str], exclude: Dict[str, set] | None = None, N=200):
    recs: Dict[str, List[str]] = {}
    item_norm = normalize(item_matrix, norm="l2")
    item_norm = item_norm.tocsr()
    for u, p in user_profiles.items():
        if p is None or np.linalg.norm(p) == 0:
            continue
        scores = item_norm @ p
        scores = np.asarray(scores).ravel()
        if exclude and u in exclude:
            # mask seen
            for seen in exclude[u]:
                # best-effort index lookup
                try:
                    idx = item_ids.index(seen)
                    scores[idx] = -1e9
                except ValueError:
                    pass
        k = min(N, len(scores))
        if k <= 0:
            continue
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        recs[u] = [item_ids[i] for i in top]
    return recs


def recommend_with_scores(user_profiles: Dict[str, np.ndarray], item_matrix: sparse.csr_matrix, item_ids: List[str], exclude: Dict[str, set] | None = None, N=200):
    recs: Dict[str, Dict[str, float]] = {}
    item_norm = normalize(item_matrix, norm="l2")
    item_norm = item_norm.tocsr()
    for u, p in user_profiles.items():
        if p is None or np.linalg.norm(p) == 0:
            continue
        scores = item_norm @ p
        scores = np.asarray(scores).ravel()
        if exclude and u in exclude:
            for seen in exclude[u]:
                try:
                    idx = item_ids.index(seen)
                    scores[idx] = -1e9
                except ValueError:
                    pass
        k = min(N, len(scores))
        if k <= 0:
            continue
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        recs[u] = {item_ids[i]: float(scores[i]) for i in top}
    return recs


