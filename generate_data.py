"""
Generate synthetic datasets for all three portfolio projects.
"""
import numpy as np
import pandas as pd
import sqlite3
import os

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. CREDIT RISK / LOAN DEFAULT DATASET
# ─────────────────────────────────────────────
n = 5000

age           = np.random.randint(22, 65, n)
income        = np.random.normal(55000, 18000, n).clip(15000, 150000).astype(int)
loan_amount   = np.random.randint(2000, 50000, n)
loan_term     = np.random.choice([12, 24, 36, 48, 60], n)
interest_rate = np.round(np.random.uniform(5, 24, n), 2)
credit_score  = np.random.randint(300, 850, n)
employment_years = np.random.randint(0, 30, n)
num_credit_lines = np.random.randint(1, 10, n)
debt_to_income = np.round(loan_amount / income, 3)
purpose       = np.random.choice(['debt_consolidation','home_improvement','car','medical','vacation','education'], n)
home_ownership= np.random.choice(['RENT','OWN','MORTGAGE'], n, p=[0.45, 0.2, 0.35])

# Logistic default probability based on features
log_odds = (
    -4
    + 0.015 * debt_to_income * 100
    - 0.006 * (credit_score - 600)
    + 0.03  * (interest_rate - 12)
    - 0.01  * employment_years
    + 0.005 * (loan_amount / 1000)
)
prob_default = 1 / (1 + np.exp(-log_odds))
default       = (np.random.rand(n) < prob_default).astype(int)

loan_df = pd.DataFrame({
    'loan_id':           [f'L{str(i).zfill(5)}' for i in range(1, n+1)],
    'age':               age,
    'annual_income':     income,
    'loan_amount':       loan_amount,
    'loan_term_months':  loan_term,
    'interest_rate':     interest_rate,
    'credit_score':      credit_score,
    'employment_years':  employment_years,
    'num_credit_lines':  num_credit_lines,
    'debt_to_income':    debt_to_income,
    'loan_purpose':      purpose,
    'home_ownership':    home_ownership,
    'default':           default,
})
os.makedirs('credit_risk/data', exist_ok=True)
loan_df.to_csv('credit_risk/data/loan_data.csv', index=False)
print(f"Loan dataset: {loan_df.shape}  |  Default rate: {default.mean():.2%}")

# ─────────────────────────────────────────────
# 2. SQL EXPLORATION DATASET  (retail store)
# ─────────────────────────────────────────────
# Customers
n_cust = 1000
customers = pd.DataFrame({
    'customer_id':   range(1, n_cust+1),
    'name':          [f'Customer_{i}' for i in range(1, n_cust+1)],
    'email':         [f'customer{i}@email.com' for i in range(1, n_cust+1)],
    'city':          np.random.choice(['New York','Los Angeles','Chicago','Houston','Phoenix','Philadelphia'], n_cust),
    'signup_date':   pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0,1460,n_cust), unit='D'),
    'age':           np.random.randint(18, 70, n_cust),
    'segment':       np.random.choice(['Bronze','Silver','Gold','Platinum'], n_cust, p=[0.4,0.3,0.2,0.1]),
})

# Products
categories = ['Electronics','Clothing','Books','Home & Garden','Sports','Beauty','Food']
products = pd.DataFrame({
    'product_id':   range(1, 201),
    'product_name': [f'Product_{i}' for i in range(1, 201)],
    'category':     np.random.choice(categories, 200),
    'price':        np.round(np.random.uniform(5, 500, 200), 2),
    'cost':         np.round(np.random.uniform(2, 200, 200), 2),
    'stock':        np.random.randint(0, 500, 200),
})
products['cost'] = np.minimum(products['cost'], products['price'] * 0.8)

# Orders
n_ord = 8000
order_dates = pd.to_datetime('2022-01-01') + pd.to_timedelta(np.random.randint(0,730,n_ord), unit='D')
orders = pd.DataFrame({
    'order_id':    range(1, n_ord+1),
    'customer_id': np.random.randint(1, n_cust+1, n_ord),
    'order_date':  order_dates,
    'status':      np.random.choice(['Completed','Pending','Cancelled','Returned'], n_ord, p=[0.75,0.1,0.1,0.05]),
    'shipping_cost': np.round(np.random.uniform(0, 25, n_ord), 2),
})

# Order Items
n_items = 15000
order_items = pd.DataFrame({
    'item_id':    range(1, n_items+1),
    'order_id':   np.random.randint(1, n_ord+1, n_items),
    'product_id': np.random.randint(1, 201, n_items),
    'quantity':   np.random.randint(1, 6, n_items),
})
order_items = order_items.merge(products[['product_id','price']], on='product_id')
order_items['line_total'] = order_items['quantity'] * order_items['price']
order_items.drop(columns='price', inplace=True)

# Write to SQLite
os.makedirs('sql_exploration/data', exist_ok=True)
db_path = 'sql_exploration/data/retail_store.db'
conn = sqlite3.connect(db_path)
customers.to_sql('customers',   conn, if_exists='replace', index=False)
products.to_sql('products',     conn, if_exists='replace', index=False)
orders.to_sql('orders',         conn, if_exists='replace', index=False)
order_items.to_sql('order_items', conn, if_exists='replace', index=False)
conn.close()
print(f"SQL DB created: customers={len(customers)}, products={len(products)}, orders={len(orders)}, items={len(order_items)}")

# ─────────────────────────────────────────────
# 3. DATA VIZ – GLOBAL CLIMATE & ENERGY DATASET
# ─────────────────────────────────────────────
years  = np.arange(1990, 2024)
countries = ['United States','China','India','Germany','Brazil','Australia','Canada','United Kingdom','France','Japan']

rows = []
for country in countries:
    base_temp  = np.random.uniform(8, 25)
    base_co2   = np.random.uniform(200, 600)
    base_renew = np.random.uniform(5, 30)
    for y in years:
        t = y - 1990
        rows.append({
            'year':            y,
            'country':         country,
            'avg_temp_c':      round(base_temp + t * 0.03 + np.random.normal(0, 0.3), 2),
            'co2_emissions_mt': round(base_co2 + t * np.random.uniform(-2, 8) + np.random.normal(0, 10), 1),
            'renewable_pct':   round(min(95, base_renew + t * np.random.uniform(0.3, 1.2) + np.random.normal(0, 1)), 1),
            'gdp_billion_usd': round(np.random.uniform(500, 25000) + t * np.random.uniform(10, 400), 1),
            'population_m':    round(np.random.uniform(20, 1400) + t * np.random.uniform(0.1, 15), 1),
            'energy_use_twh':  round(np.random.uniform(100, 6000) + t * np.random.uniform(-5, 50), 1),
        })

climate_df = pd.DataFrame(rows)
os.makedirs('data_viz/data', exist_ok=True)
climate_df.to_csv('data_viz/data/climate_energy.csv', index=False)
print(f"Climate dataset: {climate_df.shape}")

print("\n✅ All datasets generated successfully.")