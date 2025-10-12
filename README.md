## Synthetic data generator

This project includes a script to generate anonymized user journeys for the fashion catalog.

### Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python .\src\generate_data.py
```

Outputs are written to `generated_data/` as CSV files matching the schema:
- `profiles.csv`
- `product_interactions.csv`
- `wishlist_items.csv`
- `cart_items.csv`
- `orders.csv`
- `order_items.csv`

Notes:
- 30 anonymized profiles are created with consistent personas (age, gender, style affinities).
- Journeys simulate realistic search → view → wishlist/cart → checkout behavior with cross-gender browsing allowed but not dominant.
- Timestamps span recent weeks for temporal variety.
