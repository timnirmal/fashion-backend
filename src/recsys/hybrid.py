from typing import Dict, List
import numpy as np


def zscore(x: np.ndarray):
    mu = np.nanmean(x)
    sigma = np.nanstd(x) + 1e-8
    return (x - mu) / sigma


def blend_scores(cf_scores: Dict[str, Dict[str, float]], cb_scores: Dict[str, Dict[str, float]], pop: Dict[str, float], w_cf=0.5, w_cb=0.5, w_pop=0.0, top_k=20) -> Dict[str, List[str]]:
    users = set(cf_scores.keys()) | set(cb_scores.keys())
    items = set()
    for u in users:
        items |= set((cf_scores.get(u) or {}).keys())
        items |= set((cb_scores.get(u) or {}).keys())
    items = list(items)

    pop_vec = np.array([pop.get(i, 0.0) for i in items], dtype=float)
    pop_vec = np.log1p(pop_vec)
    pop_vec = zscore(pop_vec)

    out: Dict[str, List[str]] = {}
    for u in users:
        cf_vec = np.array([ (cf_scores.get(u) or {}).get(i, np.nan) for i in items ])
        cb_vec = np.array([ (cb_scores.get(u) or {}).get(i, np.nan) for i in items ])
        # normalize independently per user
        cf_n = zscore(np.nan_to_num(cf_vec, nan=np.nanmean(cf_vec)))
        cb_n = zscore(np.nan_to_num(cb_vec, nan=np.nanmean(cb_vec)))
        final = w_cf * cf_n + w_cb * cb_n + w_pop * pop_vec
        top = np.argpartition(final, -top_k)[-top_k:]
        top = top[np.argsort(final[top])[::-1]]
        out[u] = [items[i] for i in top]
    return out


def mmr_rerank(candidates: List[str], item_vectors: np.ndarray, lambda_div=0.3, k=20) -> List[str]:
    # item_vectors aligned with candidates
    if len(candidates) <= k:
        return candidates
    selected = []
    remaining = list(range(len(candidates)))
    sim = item_vectors @ item_vectors.T
    np.fill_diagonal(sim, 0.0)
    relevance = np.diag(sim) if sim.ndim == 2 else np.ones(len(candidates))
    while remaining and len(selected) < k:
        if not selected:
            best = int(np.argmax(relevance))
            selected.append(best)
            remaining.remove(best)
            continue
        max_mmr = -1e9
        best_j = remaining[0]
        for j in remaining:
            div = max(sim[j, s] for s in selected) if selected else 0.0
            mmr = (1 - lambda_div) * relevance[j] - lambda_div * div
            if mmr > max_mmr:
                max_mmr = mmr
                best_j = j
        selected.append(best_j)
        remaining.remove(best_j)
    return [candidates[i] for i in selected]


