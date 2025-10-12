<!-- a1d0be81-97ab-4d1c-bec5-6234d2710193 1bc07ede-546c-450e-bc2d-17a5f16c9530 -->
# Hybrid Recommender: EDA, CF, CB, Hybrid, and Evaluation

## Scope and Data

- Use `generated_data/*.csv` for implicit interactions and profiles; enrich items from `collected_data/products_rows.csv`.
- Item ID: `products_rows.csv.id`; User ID: `profiles.csv.id`; Timestamps: from `product_interactions.csv.created_at`.
- Filter to active/in‑stock items where possible; map variants to product level.

## Event Weights and Recency

- Weights: view=1.0, search_click=1.5, wishlist_add=3.0, add_to_cart=4.0, purchase=6.0.
- Recency decay: weight *= exp(-ln(2) * Δdays / half_life), half_life ∈ {14, 28}.

## Splits

- Temporal split by `created_at`: 70% train, 10% val, 20% test.
- Also do leave-last-1-out per user (LL1O) within val/test to measure next‑item prediction.

## EDA (notebook)

- Row counts per file; null rates; time span; per-user and per-item interaction histograms; event mix; weight distributions; top categories/tags; sparsity level; cold-start ratios.

## CF Experiments

- Common: Build sparse user×item matrix using weighted implicit signals (with decay).
- ALS (implicit): factors ∈ {64,128}, reg ∈ {0.01,0.05}, alpha ∈ {10,20,40}, iters ∈ {20,40}, use_cg=True; tune on val NDCG@20.
- BPR‑MF (implicit): factors ∈ {64,128}, reg ∈ {1e-3,1e-2}, lr ∈ {0.01,0.05}, epochs ∈ {50,100}; nsamples per epoch default; sample negatives by popularity.
- Candidate gen: top‑200 per user (exclude seen), with score outputs.

## CB Experiments

- Text fields: `name`, `short_description`, `description`, `tags`, `search_keywords`.
- TF‑IDF: word + char n‑grams, min_df=2, max_features=50k; cosine similarity for item–item; user vector = weighted mean of interacted item vectors.
- Sparse hashing: `HashingVectorizer` with n_features=2^18, word + char 3–5‑grams; cosine sim; same user profiling.
- Candidate gen: top‑200 by similarity to user profile; keep both TF‑IDF and hashing scores.

## Hybrid Strategies

- Score blending: z‑score normalize `cf_score`, `cb_score`, `pop_score`; grid w_cf∈{0.3,0.5,0.7}, w_cb∈{0.7,0.5,0.3}; pop=log(pop+1).
- LTR rerank: Generate union candidates (≤400). Train XGBoost Ranker with features: cf_score, cb_tfidf, cb_hash, pop_score, price_norm, is_featured, inventory_qty, days_since_release, category_match, style_overlap.

## Metrics

- Ranking: Recall@{10,20}, NDCG@{10,20}, HitRate@{10,20}, MRR@{10,20}, MAP@{10,20}.
- Business: Coverage@K, Diversity (MMR proxy: 1−avg cosine), Novelty@K (−log2 pop), In‑stock rate, Price alignment.
- Segments: cold users (≤2 train events), cold items (no train interactions), by category.
- Runtime: train time, inference latency, memory.

## Deliverables

- Notebooks: `notebooks/eda.ipynb`, `notebooks/eval_report.ipynb`.
- Lib code under `src/recsys/`: loaders, featurizers, CF trainers, CB indexers, blending, LTR reranker, metrics.
- Artifacts in `artifacts/`: models, vectorizers, candidate dumps, metrics JSON.

## Implementation Notes

- Use `pandas`, `numpy/scipy`, `scikit-learn`, `implicit`, `xgboost`; optional `lightgbm`.
- Ensure deterministic seeds; filter inactive/out‑of‑stock items; remove train leakage from val/test.
- Add MMR diversification at final rerank (λ∈{0.2,0.4}).

### To-dos

- [ ] Install libs and create project skeleton under src/recsys
- [ ] Create EDA notebook over generated_data and products
- [ ] Build interaction matrix with weights and recency decay
- [ ] Train/tune implicit ALS; export top-200 candidates and scores
- [ ] Train/tune BPR-MF; export top-200 candidates and scores
- [ ] Fit TF-IDF; build user profiles; export candidates/scores
- [ ] Fit HashingVectorizer; build user profiles; export candidates/scores
- [ ] Implement z-score blending and weight grid search
- [ ] Train XGBoost ranker with features to rerank union candidates
- [ ] Implement metrics: Recall/NDCG/MRR/MAP, coverage, diversity
- [ ] Run eval across methods and segments; save metrics JSON
- [ ] Summarize results and recommendations in eval_report notebook