import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..recsys.data import (
    load_frames,
    preprocess_items,
    add_weights_and_recency,
    build_id_maps,
    build_interaction_matrix,
)
from ..recsys.cf_bpr import train_bpr, recommend_users as bpr_recommend, recommend_with_scores as bpr_scores
from ..recsys.cb_text import (
    build_item_text,
    fit_tfidf,
    transform_tfidf,
    build_user_profiles,
    recommend_by_profile,
    recommend_with_scores as cb_scores,
)
from ..recsys.hybrid import blend_scores


class RecommenderService:
    """Facade that loads data and supports generating recommendations for a user.

    This service keeps minimal state so it can be reloaded cheaply at runtime.
    """

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = project_root or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self._loaded = False

        # Data
        self.products = None
        self.interactions = None
        self.user_ids: List[str] = []
        self.item_ids: List[str] = []

        # Matrices and indices
        self.user_items_csr = None
        self.uid2i: Dict[str, int] = {}
        self.iid2i: Dict[str, int] = {}

        # Content-based
        self.tfidf_vec = None
        self.tfidf_X = None
        self.user_profiles: Dict[str, np.ndarray] = {}

        # A lightweight CF model (BPR) for quick online recs
        self.bpr_model = None

        # Popularity for blending
        self.item_popularity: Dict[str, float] = {}

    def load(self) -> None:
        frames = load_frames(self.project_root)
        self.products = preprocess_items(frames["products"])  # active
        interactions = add_weights_and_recency(frames["interactions"])  # weights

        # Keep interactions for active items only
        interactions = interactions[interactions["product_id"].isin(self.products["id"])].copy()
        self.interactions = interactions

        # Build ID maps and matrices
        uid2i, iid2i, user_ids, item_ids = build_id_maps(
            interactions["user_id"], interactions["product_id"]
        )
        self.uid2i = uid2i
        self.iid2i = iid2i
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.user_items_csr = build_interaction_matrix(interactions, uid2i, iid2i)

        # Content: TF-IDF item representations
        corpus = build_item_text(self.products.set_index("id").loc[self.item_ids]).fillna("")
        self.tfidf_vec, self.tfidf_X = fit_tfidf(corpus)

        # Precompute simple user profiles from interactions
        user_items = {
            u: interactions[interactions["user_id"] == u]["product_id"].unique().tolist()
            for u in self.user_ids
        }
        item_index = {pid: i for i, pid in enumerate(self.item_ids)}
        self.user_profiles = build_user_profiles(user_items, item_index, self.tfidf_X)

        # CF: a compact BPR model
        self.bpr_model = train_bpr(self.user_items_csr, factors=64, reg=1e-3, learning_rate=0.05, epochs=30)

        # Popularity for blending
        self.item_popularity = {
            pid: float((interactions[interactions["product_id"] == pid].shape[0]))
            for pid in self.item_ids
        }

        self._loaded = True

    def is_ready(self) -> bool:
        return bool(self._loaded)

    def methods(self) -> List[str]:
        return ["cb_tfidf", "cf_bpr", "blend_bpr_tfidf"]

    def _cb_scores(self, user_id: str, top_k: int) -> Tuple[List[str], Dict[str, float]]:
        profile = self.user_profiles.get(user_id)
        if profile is None or self.tfidf_X is None:
            return [], {}
        # Build recs and scores for single user
        recs = recommend_by_profile({user_id: profile}, self.tfidf_X, self.item_ids, exclude=None, N=top_k)
        scores = cb_scores({user_id: profile}, self.tfidf_X, self.item_ids, exclude=None, N=top_k)
        return recs.get(user_id, []), scores.get(user_id, {})

    def _cf_scores(self, user_id: str, top_k: int) -> Tuple[List[str], Dict[str, float]]:
        if self.bpr_model is None:
            return [], {}
        # Build a tiny user_items matrix row for this user id if present
        if user_id not in self.uid2i:
            return [], {}
        user_idx = self.uid2i[user_id]
        # Use existing matrix row
        row = self.user_items_csr[user_idx]
        ids, scores = self.bpr_model.recommend(user_idx, row, N=top_k, filter_already_liked_items=True)
        valid = [(int(i), float(s)) for i, s in zip(ids, scores) if 0 <= int(i) < len(self.item_ids)]
        rec_items = [self.item_ids[i] for i, _ in valid]
        rec_scores = {self.item_ids[i]: s for i, s in valid}
        return rec_items, rec_scores

    def top_popular(self, top_k: int = 20) -> List[str]:
        if not self.item_popularity:
            return []
        return [
            pid for pid, _ in sorted(
                self.item_popularity.items(), key=lambda kv: kv[1], reverse=True
            )[:top_k]
        ]

    def recommend(self, user_id: str, method: str = "blend_bpr_tfidf", top_k: int = 20) -> List[str]:
        if not self._loaded:
            self.load()

        method = method or "blend_bpr_tfidf"
        method = method.lower()

        if method == "cb_tfidf":
            items, _ = self._cb_scores(user_id, top_k)
            return items[:top_k] if items else self.top_popular(top_k)
        if method == "cf_bpr":
            items, _ = self._cf_scores(user_id, top_k)
            return items[:top_k] if items else self.top_popular(top_k)

        # Default: blend BPR + TF-IDF + small popularity prior
        cb_items, cb_s = self._cb_scores(user_id, max(top_k, 200))
        cf_items, cf_s = self._cf_scores(user_id, max(top_k, 200))
        if not cb_s and not cf_s:
            # Cold-start fallback: popularity
            return self.top_popular(top_k)

        users = [user_id]
        cf_scores = {user_id: cf_s}
        cb_scores_map = {user_id: cb_s}
        blended = blend_scores(cf_scores, cb_scores_map, self.item_popularity, w_cf=0.6, w_cb=0.5, w_pop=0.1, top_k=max(top_k, 200))
        return blended.get(user_id, [])[:top_k]

    def user_exists(self, user_id: str) -> bool:
        return user_id in self.uid2i

    def item_exists(self, item_id: str) -> bool:
        return item_id in self.iid2i

    def get_item(self, item_id: str) -> Optional[dict]:
        if self.products is None:
            return None
        try:
            row = self.products[self.products["id"] == item_id].iloc[0]
            return row.to_dict()
        except Exception:
            return None


