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


