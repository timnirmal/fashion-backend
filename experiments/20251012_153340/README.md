# Experiment 20251012_153340


## Data

Users: 30, Items: 25

Train/Val/Test rows: 709/101/203


## Params

{
  "als": {
    "factors": 64,
    "reg": 0.05,
    "alpha": 20.0,
    "iterations": 20
  },
  "bpr": {
    "factors": 64,
    "reg": 0.001,
    "learning_rate": 0.05,
    "epochs": 50
  },
  "tfidf": {
    "max_features": 50000,
    "min_df": 2,
    "ngram_range": [
      1,
      2
    ]
  },
  "hashing": {
    "n_features": 262144
  },
  "blend_grid": [
    [
      0.3,
      0.7,
      0.1
    ],
    [
      0.5,
      0.5,
      0.1
    ],
    [
      0.7,
      0.3,
      0.1
    ]
  ],
  "ltr": {
    "enabled": false
  }
}

## Metrics (ndcg@20)

- als: 0.1728
- bpr: 0.2862
- tfidf: 0.2095
- hash: 0.2658
- blend_wcf0.3_wcb0.7_wpop0.1: 0.4242
- blend_wcf0.5_wcb0.5_wpop0.1: 0.4242
- blend_wcf0.7_wcb0.3_wpop0.1: 0.4242