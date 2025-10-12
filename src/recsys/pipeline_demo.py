import os
import json
import pickle
import csv
from datetime import datetime
import numpy as np
from .data import (
    load_frames,
    preprocess_items,
    add_weights_and_recency,
    build_id_maps,
    build_interaction_matrix,
    temporal_split,
)
from .cf_als import train_als, recommend_users as als_recommend, recommend_with_scores as als_scores
from .cf_bpr import train_bpr, recommend_users as bpr_recommend, recommend_with_scores as bpr_scores
from .cb_text import (
    build_item_text,
    fit_tfidf,
    fit_hashing,
    build_user_profiles,
    recommend_by_profile,
    recommend_with_scores as cb_scores,
)
from .hybrid import blend_scores, mmr_rerank
from .metrics import metrics_suite
from .ltr import build_feature_matrix, train_xgb_ranker


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
    als_recs = als_recommend(als, train_mat, user_ids, item_ids, N=200)
    als_sc = als_scores(als, train_mat, user_ids, item_ids, N=200)

    # CF: BPR
    bpr = train_bpr(train_mat, factors=64, reg=1e-3, learning_rate=0.05, epochs=50)
    bpr_recs = bpr_recommend(bpr, train_mat, user_ids, item_ids, N=200)
    bpr_sc = bpr_scores(bpr, train_mat, user_ids, item_ids, N=200)

    # CB: TF-IDF and Hashing
    corpus = build_item_text(products.set_index("id").loc[item_ids]).fillna("")
    tfidf_vec, tfidf_X = fit_tfidf(corpus)
    hash_vec, hash_X = fit_hashing(corpus)

    # User profiles from train interactions
    user_items = {u: train_df[train_df["user_id"]==u]["product_id"].unique().tolist() for u in user_ids}
    item_index = {pid: i for i, pid in enumerate(item_ids)}
    tfidf_profiles = build_user_profiles(user_items, item_index, tfidf_X)
    hash_profiles = build_user_profiles(user_items, item_index, hash_X)
    tfidf_recs = recommend_by_profile(tfidf_profiles, tfidf_X, item_ids, exclude={u:set(v) for u,v in user_items.items()}, N=200)
    hash_recs = recommend_by_profile(hash_profiles, hash_X, item_ids, exclude={u:set(v) for u,v in user_items.items()}, N=200)
    tfidf_sc = cb_scores(tfidf_profiles, tfidf_X, item_ids, exclude={u:set(v) for u,v in user_items.items()}, N=200)
    hash_sc = cb_scores(hash_profiles, hash_X, item_ids, exclude={u:set(v) for u,v in user_items.items()}, N=200)

    # Build ground truth for validation: last interactions per user in val set
    val_last = val_df.sort_values(["user_id","created_at"]).groupby("user_id").tail(1)
    gt = {u: [pid] for u, pid in zip(val_last["user_id"], val_last["product_id"]) if u in set(user_ids)}

    # Popularity for novelty and blending
    pop = {pid: float((train_df[train_df["product_id"] == pid].shape[0])) for pid in item_ids}

    # Blending: use BPR and TF-IDF scores; search small grid
    blends = {}
    for w_cf, w_cb, w_pop in [(0.3, 0.7, 0.1), (0.5, 0.5, 0.1), (0.7, 0.3, 0.1)]:
        recs = blend_scores(bpr_sc, tfidf_sc, pop, w_cf=w_cf, w_cb=w_cb, w_pop=w_pop, top_k=200)
        blends[f"wcf{w_cf}_wcb{w_cb}_wpop{w_pop}"] = recs

    # Evaluate methods
    results = {"als": als_recs, "bpr": bpr_recs, "tfidf": tfidf_recs, "hash": hash_recs}
    results.update({f"blend_{k}": v for k, v in blends.items()})

    # Item vectors for diversity (use TF-IDF)
    item_index = {pid: i for i, pid in enumerate(item_ids)}
    item_vectors = {pid: tfidf_X[item_index[pid]].toarray().ravel() for pid in item_ids}

    report = {}
    for name, recs in results.items():
        report[name] = metrics_suite(recs, gt, all_item_ids=item_ids, item_vectors=item_vectors, item_popularity=pop, ks=(10, 20))

    # Learning-to-rank reranker over union candidates (<=400 per user)
    union_candidates = {}
    method_names = ["bpr", "tfidf", "hash"] + [f"blend_{k}" for k in blends.keys()]
    for u in user_ids:
        pool = []
        for name in method_names:
            pool.extend(results.get(name, {}).get(u, [])[:200])
        seen = set()
        uniq = []
        for it in pool:
            if it not in seen:
                uniq.append(it)
                seen.add(it)
        union_candidates[u] = uniq[:400]

    # Build LTR feature matrix; labels from validation ground-truth
    X, y, qid, item_list = build_feature_matrix(
        user_ids=user_ids,
        candidates=union_candidates,
        cf_scores=bpr_sc,
        cb_tfidf_scores=tfidf_sc,
        cb_hash_scores=hash_sc,
        pop=pop,
        item_numeric=None,
    )
    # Set relevance labels
    offset = 0
    for u, q in zip(user_ids, qid):
        gt_items = set(gt.get(u, []))
        if q > 0:
            for i in range(q):
                if item_list[offset + i] in gt_items:
                    y[offset + i] = 1.0
        offset += q

    # Train and apply XGB ranker
    try:
        ranker = train_xgb_ranker(X, y, qid)
        preds = ranker.predict(X)
        reranked = {}
        offset = 0
        for u, q in zip(user_ids, qid):
            seg_items = item_list[offset: offset + q]
            seg_scores = preds[offset: offset + q]
            order = np.argsort(-seg_scores)
            top_items = [seg_items[i] for i in order[:200]]
            # Optional MMR diversification using TF-IDF vectors
            vecs = np.vstack([item_vectors[it] for it in top_items if it in item_vectors]) if top_items else np.zeros((0, len(next(iter(item_vectors.values())))))
            if vecs.shape[0] >= 2:
                top_items = mmr_rerank(top_items, vecs, lambda_div=0.3, k=min(200, len(top_items)))
            reranked[u] = top_items
            offset += q
        results["ltr_xgb"] = reranked
        report["ltr_xgb"] = metrics_suite(reranked, gt, all_item_ids=item_ids, item_vectors=item_vectors, item_popularity=pop, ks=(10, 20))
    except Exception:
        pass

    # Experiment folder with timestamp
    exp_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, "experiments", exp_ts)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Persist config for reproducibility
    cfg = {
        "data": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "num_users": int(len(user_ids)),
            "num_items": int(len(item_ids)),
            "half_life_days": 28,
        },
        "params": {
            "als": {"factors": 64, "reg": 0.05, "alpha": 20.0, "iterations": 20},
            "bpr": {"factors": 64, "reg": 1e-3, "learning_rate": 0.05, "epochs": 50},
            "tfidf": {"max_features": getattr(tfidf_vec, "max_features", None), "min_df": getattr(tfidf_vec, "min_df", None), "ngram_range": getattr(tfidf_vec, "ngram_range", None)},
            "hashing": {"n_features": getattr(hash_vec, "n_features", None)},
            "blend_grid": [(0.3, 0.7, 0.1), (0.5, 0.5, 0.1), (0.7, 0.3, 0.1)],
            "ltr": {"enabled": "ltr_xgb" in report},
        },
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Write README summary
    readme = []
    readme.append(f"# Experiment {exp_ts}\n")
    readme.append("\n## Data\n")
    readme.append(f"Users: {len(user_ids)}, Items: {len(item_ids)}\n")
    readme.append(f"Train/Val/Test rows: {len(train_df)}/{len(val_df)}/{len(test_df)}\n")
    readme.append("\n## Params\n")
    readme.append(json.dumps(cfg["params"], indent=2))
    readme.append("\n## Metrics (ndcg@20)\n")
    for name, m in report.items():
        readme.append(f"- {name}: {m.get('ndcg@20', 0.0):.4f}")
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))

    # Save models/vectorizers under models/
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    try:
        with open(os.path.join(models_dir, "als.pkl"), "wb") as f:
            pickle.dump(als, f)
        with open(os.path.join(models_dir, "bpr.pkl"), "wb") as f:
            pickle.dump(bpr, f)
        with open(os.path.join(models_dir, "tfidf_vec.pkl"), "wb") as f:
            pickle.dump(tfidf_vec, f)
        with open(os.path.join(models_dir, "hash_vec.pkl"), "wb") as f:
            pickle.dump(hash_vec, f)
    except Exception:
        pass

    # Report folder with a machine-readable CSV and a concise summary
    report_dir = os.path.join(out_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    # metrics.csv
    try:
        # collect all metric keys
        metric_keys = sorted({k for m in report.values() for k in m.keys()})
        with open(os.path.join(report_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["model"] + metric_keys)
            for name, m in report.items():
                writer.writerow([name] + [m.get(k, "") for k in metric_keys])
    except Exception:
        pass
    # summary.md
    try:
        best = max(report.items(), key=lambda kv: kv[1].get("ndcg@20", 0.0))
        with open(os.path.join(report_dir, "summary.md"), "w", encoding="utf-8") as fsum:
            fsum.write("# Evaluation Summary\n\n")
            fsum.write(f"Best by ndcg@20: {best[0]} = {best[1].get('ndcg@20', 0.0):.4f}\n\n")
            fsum.write("Top metrics (ndcg@20, recall@20, diversity@20, novelty@20):\n\n")
            for name, m in report.items():
                fsum.write(f"- {name}: ndcg@20={m.get('ndcg@20', 0.0):.4f}, recall@20={m.get('recall@20', 0.0):.3f}, diversity@20={m.get('diversity@20', 0.0):.3f}, novelty@20={m.get('novelty@20', 0.0):.3f}\n")
    except Exception:
        pass

    # Dump candidates per method
    def dump_candidates(name, recs):
        path = os.path.join(out_dir, f"candidates_{name}.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("user_id,product_id\n")
            for u, items in recs.items():
                for it in items[:200]:
                    f.write(f"{u},{it}\n")

    for name, recs in results.items():
        dump_candidates(name, recs)

    # Also keep a latest symlink/copy for convenience
    latest_dir = os.path.join(root, "experiments", "latest")
    try:
        if os.path.islink(latest_dir) or os.path.exists(latest_dir):
            try:
                os.remove(latest_dir)
            except Exception:
                pass
        os.symlink(out_dir, latest_dir)
    except Exception:
        # Fallback: copy metrics.json as latest
        try:
            with open(os.path.join(root, "experiments", "metrics_latest.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass

    return report


if __name__ == "__main__":
    # project root: three levels up from this file (src/recsys/pipeline_demo.py -> project root)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    m = run_small(root)
    print(m)


