import os
import numpy as np
from .data import load_frames, preprocess_items, add_weights_and_recency, build_id_maps, build_interaction_matrix, temporal_split
from .cf_als import train_als, recommend_users as als_recommend
from .cf_bpr import train_bpr, recommend_users as bpr_recommend
from .cb_text import build_item_text, fit_tfidf, fit_hashing, build_user_profiles, recommend_by_profile
from .metrics import ndcg_at_k, recall_at_k, mrr_at_k


def run_small(root: str):
    frames = load_frames(root)
    products = preprocess_items(frames["products"])  # active only
    interactions = add_weights_and_recency(frames["interactions"])  # weight + decay

    # filter interactions to active products
    interactions = interactions[interactions["product_id"].isin(products["id"])].copy()

    # temporal split
    train_df, val_df, test_df = temporal_split(interactions, 0.7, 0.1)

    # build maps and matrices (train only)
    uid2i, iid2i, user_ids, item_ids = build_id_maps(train_df["user_id"], train_df["product_id"])
    train_mat = build_interaction_matrix(train_df, uid2i, iid2i)

    # CF: ALS
    als = train_als(train_mat, factors=64, reg=0.05, alpha=20.0, iterations=20)
    als_recs = als_recommend(als, train_mat, user_ids, item_ids, N=50)

    # CF: BPR
    bpr = train_bpr(train_mat, factors=64, reg=1e-3, learning_rate=0.05, epochs=50)
    bpr_recs = bpr_recommend(bpr, train_mat, user_ids, item_ids, N=50)

    # CB: TF-IDF and Hashing
    corpus = build_item_text(products.set_index("id").loc[item_ids]).fillna("")
    tfidf_vec, tfidf_X = fit_tfidf(corpus)
    hash_vec, hash_X = fit_hashing(corpus)

    # User profiles from train interactions
    user_items = {u: train_df[train_df["user_id"]==u]["product_id"].unique().tolist() for u in user_ids}
    item_index = {pid: i for i, pid in enumerate(item_ids)}
    tfidf_profiles = build_user_profiles(user_items, item_index, tfidf_X)
    hash_profiles = build_user_profiles(user_items, item_index, hash_X)
    tfidf_recs = recommend_by_profile(tfidf_profiles, tfidf_X, item_ids, exclude={u:set(v) for u,v in user_items.items()}, N=50)
    hash_recs = recommend_by_profile(hash_profiles, hash_X, item_ids, exclude={u:set(v) for u,v in user_items.items()}, N=50)

    # Build ground truth for validation: last interactions per user in val set
    val_last = val_df.sort_values(["user_id","created_at"]).groupby("user_id").tail(1)
    gt = {u: [pid] for u, pid in zip(val_last["user_id"], val_last["product_id"]) if u in set(user_ids)}

    # Evaluate
    metrics = {
        "als_ndcg@20": ndcg_at_k(als_recs, gt, 20),
        "bpr_ndcg@20": ndcg_at_k(bpr_recs, gt, 20),
        "tfidf_ndcg@20": ndcg_at_k(tfidf_recs, gt, 20),
        "hash_ndcg@20": ndcg_at_k(hash_recs, gt, 20),
        "als_recall@20": recall_at_k(als_recs, gt, 20),
        "bpr_recall@20": recall_at_k(bpr_recs, gt, 20),
    }
    return metrics


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = run_small(root)
    print(m)


