import csv
import json
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

try:
    from faker import Faker
except ImportError:
    print("Missing dependency: faker. Run: pip install -r requirements.txt", file=sys.stderr)
    raise


# -------------------------
# Config
# -------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT_DIR, "collected_data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "generated_data")

PRODUCTS_CSV = os.path.join(INPUT_DIR, "products_rows.csv")
VARIANTS_CSV = os.path.join(INPUT_DIR, "product_variants_rows.csv")

NUM_USERS = 30
SEED = 42


def utc_now():
    return datetime.now(timezone.utc)


def format_ts(dt: datetime) -> str:
    # Match the input CSV style: 2025-10-11 11:54:01.090122+00
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f+00")


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def parse_list_field(value: str):
    if not value or value == "null":
        return []
    try:
        return json.loads(value)
    except Exception:
        # Fallback to best-effort eval-like parsing
        try:
            import ast

            return ast.literal_eval(value)
        except Exception:
            return []


def load_products(products_csv: str):
    products = {}
    with open(products_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["images"] = parse_list_field(row.get("images", ""))
            row["tags"] = parse_list_field(row.get("tags", ""))
            row["search_keywords"] = parse_list_field(row.get("search_keywords", ""))
            products[row["id"]] = row
    return products


def load_variants(variants_csv: str):
    variants_by_product = {}
    with open(variants_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product_id = row["product_id"]
            variants_by_product.setdefault(product_id, []).append(row)
    return variants_by_product


class Persona:
    def __init__(self, faker: Faker):
        self.age = random.choices(
            population=[
                (18, 24),
                (25, 34),
                (35, 44),
                (45, 54),
                (55, 65),
            ],
            weights=[18, 32, 24, 16, 10],
            k=1,
        )[0]
        self.age_value = random.randint(self.age[0], self.age[1])
        self.gender = random.choices(["male", "female", "non-binary"], weights=[47, 47, 6], k=1)[0]
        # Style affinities influence browsing
        self.style_affinities = random.sample(
            [
                "casual",
                "workwear",
                "streetwear",
                "luxury",
                "sustainable",
                "summer",
                "winter",
                "formal",
                "travel",
                "everyday",
            ],
            k=random.choice([2, 3, 4]),
        )
        # Cross-browse probability between gender-coded catalog segments
        base_cross = 0.18 if self.gender == "male" else 0.14
        self.cross_gender_browse_prob = base_cross + (0.06 if "streetwear" in self.style_affinities else 0.0)
        self.location = {
            "name": faker.name_nonbinary() if self.gender == "non-binary" else faker.name_male() if self.gender == "male" else faker.name_female(),
            "street": faker.street_address(),
            "city": faker.city(),
            "state": faker.state_abbr(),
            "postal_code": faker.postcode(),
            "country": "US",
            "phone": faker.phone_number(),
        }


def sku_gender_hint(sku: str) -> str:
    # Heuristic from provided SKUs: M-*, W-*, A-* (accessories)
    if not sku:
        return "unisex"
    if sku.startswith("M-"):
        return "male"
    if sku.startswith("W-"):
        return "female"
    if sku.startswith("A-"):
        return "unisex"
    return "unisex"


def product_affinity_score(product: dict, persona: Persona) -> float:
    tags = set((product.get("tags") or []) + (product.get("search_keywords") or []))
    style_matches = len(set(persona.style_affinities) & tags)
    sku = product.get("sku", "")
    gender_hint = sku_gender_hint(sku)
    # Gender fit boosts score, but allow cross-browse via probability gate later
    gender_boost = 0.9
    if gender_hint == "unisex":
        gender_boost = 1.0
    elif (gender_hint == "male" and persona.gender == "male") or (
        gender_hint == "female" and persona.gender == "female"
    ):
        gender_boost = 1.15
    base = 1.0 + 0.4 * style_matches
    # Prefer featured/active items a bit
    if product.get("is_featured", "").lower() == "true":
        base *= 1.1
    if product.get("is_active", "").lower() != "true":
        base *= 0.8
    return base * gender_boost


def choose_candidates(products: dict, persona: Persona, k: int) -> list:
    # Weighted sampling by affinity; also inject a small fraction of off-target noise
    items = list(products.values())
    weights = [product_affinity_score(p, persona) for p in items]
    chosen = random.choices(items, weights=weights, k=k)
    # Inject off-target a bit to avoid obvious patterns
    for i in range(len(chosen)):
        if random.random() < 0.2:
            chosen[i] = random.choice(items)
    # Filter by cross-gender browse probability
    filtered = []
    for p in chosen:
        sku = p.get("sku", "")
        hint = sku_gender_hint(sku)
        if hint == "unisex":
            filtered.append(p)
        elif persona.gender == "male" and hint == "female":
            if random.random() < persona.cross_gender_browse_prob:
                filtered.append(p)
        elif persona.gender == "female" and hint == "male":
            if random.random() < persona.cross_gender_browse_prob:
                filtered.append(p)
        else:
            filtered.append(p)
    return filtered or chosen


def choose_variant(variants_by_product: dict, product_id: str) -> dict | None:
    vs = variants_by_product.get(product_id, [])
    if not vs:
        return None
    # Prefer variants with available inventory
    available = [v for v in vs if (int(v.get("inventory_quantity") or 0)) > 0]
    pool = available or vs
    return random.choice(pool)


def rand_session_id() -> str:
    return uuid.uuid4().hex


def rand_order_number(dt: datetime) -> str:
    return f"ORD-{dt.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"


def simulate(
    products: dict,
    variants_by_product: dict,
    num_users: int,
    seed: int,
):
    random.seed(seed)
    faker = Faker("en_US")
    Faker.seed(seed)

    ensure_dir(OUTPUT_DIR)

    # Output collectors
    profiles_rows = []
    interactions_rows = []
    wishlist_rows = []
    cart_rows = []
    orders_rows = []
    order_items_rows = []

    catalog = list(products.values())
    now = utc_now()

    for _ in range(num_users):
        auth_user_id = str(uuid.uuid4())
        persona = Persona(faker)

        # Profile timing
        created_shift_days = random.randint(40, 120)
        profile_created_at = now - timedelta(days=created_shift_days, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        profile_updated_at = profile_created_at + timedelta(days=random.randint(0, 10))

        # Profile data
        display_name = persona.location["name"]
        avatar_url = ""
        phone = persona.location["phone"]
        shipping_address = {
            "name": display_name,
            "line1": persona.location["street"],
            "city": persona.location["city"],
            "state": persona.location["state"],
            "postal_code": persona.location["postal_code"],
            "country": persona.location["country"],
            # Non-PII persona hints live only here to keep dataset self-contained and anonymized
            "persona": {
                "age": persona.age_value,
                "gender": persona.gender,
                "styles": persona.style_affinities,
            },
        }
        billing_address = shipping_address

        profiles_rows.append(
            {
                "id": auth_user_id,
                "display_name": display_name,
                "avatar_url": avatar_url,
                "phone": phone,
                "shipping_address": json.dumps(shipping_address),
                "billing_address": json.dumps(billing_address),
                "created_at": format_ts(profile_created_at),
                "updated_at": format_ts(profile_updated_at),
            }
        )

        # Journeys per user
        num_sessions = random.randint(2, 5)
        for _s in range(num_sessions):
            session_id = rand_session_id()
            # Session start time within last 60 days
            session_start = now - timedelta(days=random.randint(1, 60), hours=random.randint(0, 23), minutes=random.randint(0, 59))
            cursor_time = session_start

            # Simulate a search intent and subsequent clicks/views
            query_seed_product = random.choice(catalog)
            # Query terms are picked from product tags/keywords and persona styles
            query_terms = []
            if query_seed_product.get("search_keywords"):
                query_terms.extend(random.sample(query_seed_product["search_keywords"], k=min(2, len(query_seed_product["search_keywords"])) ))
            query_terms.extend(random.sample(persona.style_affinities, k=random.randint(1, min(2, len(persona.style_affinities)))))
            query = " ".join(dict.fromkeys(query_terms))[:60]

            candidates = choose_candidates(products, persona, k=random.randint(3, 6))
            random.shuffle(candidates)

            # Search result clicks and product views
            rank = 1
            viewed_products = []
            for p in candidates:
                # search_click
                interactions_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "user_id": auth_user_id,
                        "product_id": p["id"],
                        "interaction_type": "search_click",
                        "session_id": session_id,
                        "metadata": json.dumps({"query": query, "rank": rank, "device": random.choice(["mobile", "desktop"])}),
                        "created_at": format_ts(cursor_time),
                    }
                )
                cursor_time += timedelta(seconds=random.randint(3, 18))

                # view
                interactions_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "user_id": auth_user_id,
                        "product_id": p["id"],
                        "interaction_type": "view",
                        "session_id": session_id,
                        "metadata": json.dumps({"referrer": "search", "dwell_sec": random.randint(5, 90)}),
                        "created_at": format_ts(cursor_time),
                    }
                )
                cursor_time += timedelta(seconds=random.randint(5, 60))
                viewed_products.append(p)
                rank += 1

            # Optional wishlist add
            if viewed_products and random.random() < 0.35:
                p = random.choice(viewed_products)
                wishlist_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "user_id": auth_user_id,
                        "product_id": p["id"],
                        "created_at": format_ts(cursor_time),
                    }
                )
                interactions_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "user_id": auth_user_id,
                        "product_id": p["id"],
                        "interaction_type": "wishlist_add",
                        "session_id": session_id,
                        "metadata": json.dumps({}),
                        "created_at": format_ts(cursor_time),
                    }
                )
                cursor_time += timedelta(seconds=random.randint(5, 40))

            # Add to cart and maybe purchase
            will_buy_this_session = random.random() < 0.42
            cart_products = random.sample(viewed_products, k=min(len(viewed_products), random.randint(1, 2))) if viewed_products else []
            order_subtotal = 0.0
            order_items = []

            for p in cart_products:
                if random.random() < 0.75:  # not all viewed products go to cart
                    v = choose_variant(variants_by_product, p["id"])
                    if not v:
                        continue
                    qty = 1 if random.random() < 0.8 else 2
                    unit_price = float(p.get("price") or 0.0) + float(v.get("price_adjustment") or 0.0)
                    line_total = unit_price * qty
                    order_subtotal += line_total

                    # cart_items row
                    cart_row = {
                        "id": str(uuid.uuid4()),
                        "user_id": auth_user_id,
                        "product_id": p["id"],
                        "variant_id": v["id"],
                        "quantity": str(qty),
                        "created_at": format_ts(cursor_time),
                        "updated_at": format_ts(cursor_time + timedelta(seconds=random.randint(10, 120))),
                    }
                    cart_rows.append(cart_row)
                    interactions_rows.append(
                        {
                            "id": str(uuid.uuid4()),
                            "user_id": auth_user_id,
                            "product_id": p["id"],
                            "interaction_type": "add_to_cart",
                            "session_id": session_id,
                            "metadata": json.dumps({"variant_id": v["id"], "quantity": qty}),
                            "created_at": cart_row["created_at"],
                        }
                    )
                    cursor_time += timedelta(seconds=random.randint(5, 45))
                    order_items.append((p, v, qty, unit_price, line_total))

            if will_buy_this_session and order_items:
                # checkout start interaction
                interactions_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "user_id": auth_user_id,
                        "product_id": random.choice(order_items)[0]["id"],
                        "interaction_type": "checkout_start",
                        "session_id": session_id,
                        "metadata": json.dumps({}),
                        "created_at": format_ts(cursor_time),
                    }
                )
                cursor_time += timedelta(minutes=random.randint(1, 8))

                shipping_amount = 0.0 if order_subtotal >= 100 else random.choice([0.0, 4.99, 9.99])
                tax_amount = round(order_subtotal * random.choice([0.075, 0.0825, 0.085, 0.0925]), 2)
                total_amount = round(order_subtotal + tax_amount + shipping_amount, 2)

                order_created_at = cursor_time
                order_updated_at = order_created_at + timedelta(minutes=random.randint(10, 120))

                order_id = str(uuid.uuid4())
                orders_rows.append(
                    {
                        "id": order_id,
                        "user_id": auth_user_id,
                        "order_number": rand_order_number(order_created_at),
                        "status": random.choice(["processing", "completed", "shipped"]),
                        "total_amount": f"{total_amount:.2f}",
                        "subtotal": f"{order_subtotal:.2f}",
                        "tax_amount": f"{tax_amount:.2f}",
                        "shipping_amount": f"{shipping_amount:.2f}",
                        "shipping_address": json.dumps(shipping_address),
                        "billing_address": json.dumps(billing_address),
                        "payment_status": "paid",
                        "created_at": format_ts(order_created_at),
                        "updated_at": format_ts(order_updated_at),
                    }
                )

                for p, v, qty, unit_price, line_total in order_items:
                    order_items_rows.append(
                        {
                            "id": str(uuid.uuid4()),
                            "order_id": order_id,
                            "product_id": p["id"],
                            "variant_id": v["id"],
                            "quantity": str(qty),
                            "unit_price": f"{unit_price:.2f}",
                            "total_price": f"{line_total:.2f}",
                            "created_at": format_ts(order_created_at),
                        }
                    )
                cursor_time = order_updated_at
            else:
                # Abandoned cart signal occasionally
                if order_items and random.random() < 0.5:
                    p = random.choice(order_items)[0]
                    interactions_rows.append(
                        {
                            "id": str(uuid.uuid4()),
                            "user_id": auth_user_id,
                            "product_id": p["id"],
                            "interaction_type": "abandon_cart",
                            "session_id": session_id,
                            "metadata": json.dumps({}),
                            "created_at": format_ts(cursor_time),
                        }
                    )

    # Write outputs
    write_profiles_csv(profiles_rows)
    write_interactions_csv(interactions_rows)
    write_wishlist_csv(wishlist_rows)
    write_cart_csv(cart_rows)
    write_orders_csv(orders_rows)
    write_order_items_csv(order_items_rows)


def write_csv(path: str, fieldnames: list, rows: list) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_profiles_csv(rows: list) -> None:
    path = os.path.join(OUTPUT_DIR, "profiles.csv")
    cols = [
        "id",
        "display_name",
        "avatar_url",
        "phone",
        "shipping_address",
        "billing_address",
        "created_at",
        "updated_at",
    ]
    write_csv(path, cols, rows)


def write_wishlist_csv(rows: list) -> None:
    path = os.path.join(OUTPUT_DIR, "wishlist_items.csv")
    cols = [
        "id",
        "user_id",
        "product_id",
        "created_at",
    ]
    write_csv(path, cols, rows)


def write_cart_csv(rows: list) -> None:
    path = os.path.join(OUTPUT_DIR, "cart_items.csv")
    cols = [
        "id",
        "user_id",
        "product_id",
        "variant_id",
        "quantity",
        "created_at",
        "updated_at",
    ]
    write_csv(path, cols, rows)


def write_orders_csv(rows: list) -> None:
    path = os.path.join(OUTPUT_DIR, "orders.csv")
    cols = [
        "id",
        "user_id",
        "order_number",
        "status",
        "total_amount",
        "subtotal",
        "tax_amount",
        "shipping_amount",
        "shipping_address",
        "billing_address",
        "payment_status",
        "created_at",
        "updated_at",
    ]
    write_csv(path, cols, rows)


def write_order_items_csv(rows: list) -> None:
    path = os.path.join(OUTPUT_DIR, "order_items.csv")
    cols = [
        "id",
        "order_id",
        "product_id",
        "variant_id",
        "quantity",
        "unit_price",
        "total_price",
        "created_at",
    ]
    write_csv(path, cols, rows)


def write_interactions_csv(rows: list) -> None:
    path = os.path.join(OUTPUT_DIR, "product_interactions.csv")
    cols = [
        "id",
        "user_id",
        "product_id",
        "interaction_type",
        "session_id",
        "metadata",
        "created_at",
    ]
    write_csv(path, cols, rows)


def main():
    products = load_products(PRODUCTS_CSV)
    variants_by_product = load_variants(VARIANTS_CSV)
    simulate(products, variants_by_product, NUM_USERS, SEED)
    print(f"Generated synthetic data for {NUM_USERS} users into: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


