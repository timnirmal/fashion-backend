import math
import os
import pandas as pd
import numpy as np
from scipy import sparse


EVENT_WEIGHTS = {
    "view": 1.0,
    "search_click": 1.5,
    "wishlist_add": 3.0,
    "add_to_cart": 4.0,
    "checkout_start": 4.5,
    "purchase": 6.0,
}


def load_frames(root: str):
    gd = os.path.join(root, "generated_data")
    cd = os.path.join(root, "collected_data")
    profiles = pd.read_csv(os.path.join(gd, "profiles.csv"))
    interactions = pd.read_csv(os.path.join(gd, "product_interactions.csv"))
    orders = pd.read_csv(os.path.join(gd, "orders.csv"))
    order_items = pd.read_csv(os.path.join(gd, "order_items.csv"))
    products = pd.read_csv(os.path.join(cd, "products_rows.csv"))
    variants = pd.read_csv(os.path.join(cd, "product_variants_rows.csv"))
    return {
        "profiles": profiles,
        "interactions": interactions,
        "orders": orders,
        "order_items": order_items,
        "products": products,
        "variants": variants,
    }


def preprocess_items(products: pd.DataFrame) -> pd.DataFrame:
    # Keep active items only
    active = products[products["is_active"].astype(str).str.lower() == "true"].copy()
    # Inventory > 0 if provided
    active["inventory_quantity"] = pd.to_numeric(active["inventory_quantity"], errors="coerce").fillna(0).astype(int)
    active = active[active["inventory_quantity"] >= 0]  # allow zero to support exposure
    return active


def add_weights_and_recency(interactions: pd.DataFrame, half_life_days: int = 28) -> pd.DataFrame:
    df = interactions.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])  # robust to malformed
    max_ts = df["created_at"].max()
    df["event_weight"] = df["interaction_type"].map(EVENT_WEIGHTS).fillna(0.5)
    delta_days = (max_ts - df["created_at"]).dt.total_seconds() / (60 * 60 * 24)
    decay = np.exp(-np.log(2.0) * delta_days / float(half_life_days))
    df["weight"] = df["event_weight"] * decay
    return df


def build_id_maps(users: pd.Series, items: pd.Series):
    user_ids = users.unique().tolist()
    item_ids = items.unique().tolist()
    uid2i = {u: i for i, u in enumerate(user_ids)}
    iid2i = {p: i for i, p in enumerate(item_ids)}
    return uid2i, iid2i, user_ids, item_ids


def build_interaction_matrix(df: pd.DataFrame, uid2i: dict, iid2i: dict):
    rows = df["user_id"].map(uid2i)
    cols = df["product_id"].map(iid2i)
    vals = df["weight"].astype(float)
    mask = rows.notna() & cols.notna()
    rows = rows[mask].astype(int)
    cols = cols[mask].astype(int)
    vals = vals[mask]
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(len(uid2i), len(iid2i))).tocsr()
    return mat


def temporal_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.1):
    df = df.sort_values("created_at")
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def leave_last_one_out(df: pd.DataFrame):
    # For evaluation creation per user
    df = df.sort_values(["user_id", "created_at"]).copy()
    last = df.groupby("user_id").tail(1)
    rest = df.drop(last.index)
    return rest, last


