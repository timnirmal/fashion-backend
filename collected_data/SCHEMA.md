## Database Schema (Tables and Columns)

Note: Derived from `supabase/migrations/*.sql` in this repo.

### public.profiles
- id: UUID (PK, references auth.users)
- display_name: TEXT
- avatar_url: TEXT
- phone: TEXT
- shipping_address: JSONB
- billing_address: JSONB
- created_at: TIMESTAMPTZ
- updated_at: TIMESTAMPTZ

### public.categories
- id: UUID (PK)
- name: TEXT
- slug: TEXT (UNIQUE)
- description: TEXT
- image_url: TEXT
- parent_id: UUID (references public.categories)
- created_at: TIMESTAMPTZ

### public.products
- id: UUID (PK)
- name: TEXT
- slug: TEXT (UNIQUE)
- description: TEXT
- short_description: TEXT
- price: DECIMAL(10,2)
- compare_at_price: DECIMAL(10,2)
- cost_price: DECIMAL(10,2)
- category_id: UUID (references public.categories)
- images: TEXT[]
- is_featured: BOOLEAN
- is_active: BOOLEAN
- inventory_quantity: INTEGER
- sku: TEXT (UNIQUE)
- weight: DECIMAL(8,2)
- tags: TEXT[]
- search_keywords: TEXT[]
- created_at: TIMESTAMPTZ
- updated_at: TIMESTAMPTZ

### public.product_variants
- id: UUID (PK)
- product_id: UUID (references public.products, ON DELETE CASCADE)
- name: TEXT
- value: TEXT
- price_adjustment: DECIMAL(10,2)
- inventory_quantity: INTEGER
- sku: TEXT
- created_at: TIMESTAMPTZ

### public.cart_items
- id: UUID (PK)
- user_id: UUID (references auth.users, ON DELETE CASCADE)
- product_id: UUID (references public.products, ON DELETE CASCADE)
- variant_id: UUID (references public.product_variants, ON DELETE SET NULL)
- quantity: INTEGER
- created_at: TIMESTAMPTZ
- updated_at: TIMESTAMPTZ

### public.wishlist_items
- id: UUID (PK)
- user_id: UUID (references auth.users, ON DELETE CASCADE)
- product_id: UUID (references public.products, ON DELETE CASCADE)
- created_at: TIMESTAMPTZ

### public.orders
- id: UUID (PK)
- user_id: UUID (references auth.users)
- order_number: TEXT (UNIQUE)
- status: TEXT
- total_amount: DECIMAL(10,2)
- subtotal: DECIMAL(10,2)
- tax_amount: DECIMAL(10,2)
- shipping_amount: DECIMAL(10,2)
- shipping_address: JSONB
- billing_address: JSONB
- payment_status: TEXT
- created_at: TIMESTAMPTZ
- updated_at: TIMESTAMPTZ

### public.order_items
- id: UUID (PK)
- order_id: UUID (references public.orders, ON DELETE CASCADE)
- product_id: UUID (references public.products)
- variant_id: UUID (references public.product_variants)
- quantity: INTEGER
- unit_price: DECIMAL(10,2)
- total_price: DECIMAL(10,2)
- created_at: TIMESTAMPTZ

### public.product_interactions
- id: UUID (PK)
- user_id: UUID (references auth.users, ON DELETE SET NULL)
- product_id: UUID (references public.products, ON DELETE CASCADE)
- interaction_type: TEXT
- session_id: TEXT
- metadata: JSONB
- created_at: TIMESTAMPTZ

