import numpy as np


def recall_at_k(recommended, ground_truth, k=10):
    hits = 0
    for u in ground_truth:
        gt = set(ground_truth[u])
        rec = recommended.get(u, [])[:k]
        hits += int(len(gt & set(rec)) > 0)
    return hits / max(len(ground_truth), 1)


def hitrate_at_k(recommended, ground_truth, k=10):
    return recall_at_k(recommended, ground_truth, k)


def dcg(rels):
    return np.sum((2 ** np.array(rels) - 1) / np.log2(np.arange(2, len(rels) + 2)))


def ndcg_at_k(recommended, ground_truth, k=10):
    vals = []
    for u in ground_truth:
        gt = set(ground_truth[u])
        rec = recommended.get(u, [])[:k]
        rels = [1 if r in gt else 0 for r in rec]
        ideal = sorted(rels, reverse=True)
        idcg = dcg(ideal)
        vals.append(0.0 if idcg == 0 else dcg(rels) / idcg)
    return float(np.mean(vals)) if vals else 0.0


def mrr_at_k(recommended, ground_truth, k=10):
    rr = []
    for u in ground_truth:
        gt = set(ground_truth[u])
        rec = recommended.get(u, [])[:k]
        rank = next((i + 1 for i, r in enumerate(rec) if r in gt), None)
        rr.append(0.0 if rank is None else 1.0 / rank)
    return float(np.mean(rr)) if rr else 0.0


# Mean Average Precision at K
def map_at_k(recommended, ground_truth, k=10):
    ap_vals = []
    for u in ground_truth:
        gt = set(ground_truth[u])
        if not gt:
            continue
        rec = recommended.get(u, [])[:k]
        hits = 0
        precisions = []
        for i, r in enumerate(rec, start=1):
            if r in gt:
                hits += 1
                precisions.append(hits / i)
        ap_vals.append(np.mean(precisions) if precisions else 0.0)
    return float(np.mean(ap_vals)) if ap_vals else 0.0


# Fraction of unique recommended items in top-k over catalog
def coverage_at_k(recommended, all_item_ids, k=10):
    if not all_item_ids:
        return 0.0
    seen = set()
    for recs in recommended.values():
        seen.update(recs[:k])
    return len(seen) / float(len(all_item_ids))


# Diversity proxy: 1 - average cosine similarity across pairs within each list (then average users)
def diversity_at_k(recommended, item_vectors, k=10):
    # item_vectors: Dict[item_id, np.ndarray]
    def pairwise_cosine(items):
        if len(items) < 2:
            return 1.0
        vecs = [item_vectors[i] for i in items if i in item_vectors]
        if len(vecs) < 2:
            return 1.0
        X = np.vstack(vecs)
        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms
        sim = X @ X.T
        # take upper triangle without diagonal
        n = sim.shape[0]
        idx = np.triu_indices(n, k=1)
        avg_sim = float(sim[idx].mean()) if idx[0].size > 0 else 0.0
        return 1.0 - avg_sim

    vals = []
    for u, recs in recommended.items():
        vals.append(pairwise_cosine(recs[:k]))
    return float(np.mean(vals)) if vals else 0.0


# Novelty: average -log2(popularity probability) for recommended items
def novelty_at_k(recommended, item_popularity, k=10):
    # item_popularity: Dict[item_id, count]
    total = float(sum(item_popularity.values()))
    if total <= 0:
        return 0.0
    vals = []
    for recs in recommended.values():
        scores = []
        for it in recs[:k]:
            p = item_popularity.get(it, 0.0) / total
            p = max(p, 1e-12)
            scores.append(-np.log2(p))
        if scores:
            vals.append(float(np.mean(scores)))
    return float(np.mean(vals)) if vals else 0.0


def metrics_suite(recommended, ground_truth, all_item_ids, item_vectors=None, item_popularity=None, ks=(10, 20)):
    out = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(recommended, ground_truth, k)
        out[f"ndcg@{k}"] = ndcg_at_k(recommended, ground_truth, k)
        out[f"mrr@{k}"] = mrr_at_k(recommended, ground_truth, k)
        out[f"map@{k}"] = map_at_k(recommended, ground_truth, k)
        out[f"coverage@{k}"] = coverage_at_k(recommended, all_item_ids, k)
        if item_vectors is not None:
            out[f"diversity@{k}"] = diversity_at_k(recommended, item_vectors, k)
        if item_popularity is not None:
            out[f"novelty@{k}"] = novelty_at_k(recommended, item_popularity, k)
    return out

