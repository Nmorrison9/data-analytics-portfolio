"""
Retail Store SQL Exploration
=============================
Connects to a SQLite retail database and runs a series of analytical
SQL queries to evaluate revenue by category, confirm full product-sales
coverage, examine monthly revenue trends, and assess order-status
distribution. Key findings include stable revenue with seasonal peaks
roughly every six months and a ~10 % cancellation rate that warrants
further investigation.

Author : Nathaniel Morrison
Date   : 05/21/2026
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH    = "data/retail_store.db"
PALETTE      = "#2563EB"
GRID_CLR     = "#E5E7EB"
FIG_SIZE     = (10, 5)
USD_FMT      = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")


# ── Database helpers ──────────────────────────────────────────────────────────

def get_connection(path: str = DATA_PATH) -> sqlite3.Connection:
    """Open and return a SQLite connection."""
    return sqlite3.connect(path)


def query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Execute *sql* and return the result as a DataFrame."""
    return pd.read_sql_query(sql, conn)


# ── Exploration queries ───────────────────────────────────────────────────────

def explore_tables(conn: sqlite3.Connection) -> None:
    """
    Print the first rows of every table for a quick data-dictionary check.
    Useful during development; keep or remove for a final portfolio run.
    """
    tables = query(conn, "SELECT name FROM sqlite_master WHERE type='table'")["name"].tolist()

    for table in tables:
        print(f"\n{'─' * 50}")
        print(f"TABLE: {table}")
        print("─" * 50)
        print(query(conn, f"SELECT * FROM {table} LIMIT 5").to_string(index=False))


# ── Analysis functions ────────────────────────────────────────────────────────

def revenue_by_category(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Total revenue per product category, sorted descending.

    JOIN: order_items → products
    """
    sql = """
        SELECT
            category,
            SUM(line_total) AS total_revenue
        FROM order_items
        INNER JOIN products USING (product_id)
        GROUP BY category
        ORDER BY total_revenue DESC
    """
    df = query(conn, sql)

    print("\n── Revenue by Category ───────────────────────────")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(df["category"], df["total_revenue"],
           color=PALETTE, edgecolor="#1E3A8A", linewidth=0.6, width=0.6)

    ax.set_title("Total Revenue by Product Category", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Category", fontsize=11)
    ax.set_ylabel("Total Revenue", fontsize=11)
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    plt.show()

    return df


def check_dead_stock(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Identify products that have never been ordered (dead stock).

    Uses a LEFT JOIN so products with no matching order_items rows
    are retained; filtering on IS NULL isolates those products.

    Finding: all 200 products have been ordered at least once.
    """
    sql = """
        SELECT p.product_name
        FROM products p
        LEFT JOIN order_items oi ON p.product_id = oi.product_id
        WHERE oi.product_id IS NULL
    """
    df = query(conn, sql)

    print("\n── Dead Stock (never ordered) ────────────────────")
    if df.empty:
        print("None — all products have been ordered at least once.")
    else:
        print(df.to_string(index=False))

    return df


def monthly_revenue_trend(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Aggregate revenue by calendar month across all years.

    JOIN: order_items → orders
    Observation: revenue is relatively stable with seasonal peaks
    roughly every six months; a dip in September 2023 warrants
    further investigation.
    """
    sql = """
        SELECT
            strftime('%Y-%m', order_date) AS year_month,
            SUM(line_total)               AS total_revenue
        FROM order_items
        JOIN orders USING (order_id)
        GROUP BY year_month
        ORDER BY year_month
    """
    df = query(conn, sql)

    print("\n── Monthly Revenue Trend ─────────────────────────")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(df["year_month"], df["total_revenue"],
            color=PALETTE, linewidth=2, marker="o", markersize=4)

    ax.set_title("Monthly Revenue — 2022 & 2023", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Total Revenue", fontsize=11)
    ax.yaxis.set_major_formatter(USD_FMT)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    plt.show()

    return df


def order_status_distribution(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Count and percentage breakdown of every order status.

    Observation: a ~10 % cancellation rate warrants investigation.
    The dataset lacks pre- vs. post-shipment distinction, which would
    be critical for understanding the true business impact.
    """
    sql = """
        SELECT
            status,
            COUNT(*)                                          AS order_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 1) AS pct
        FROM orders
        GROUP BY status
        ORDER BY order_count DESC
    """
    df = query(conn, sql)

    print("\n── Order Status Distribution ─────────────────────")
    print(df.to_string(index=False))

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    pd.set_option("display.max_columns", None)

    with get_connection(DATA_PATH) as conn:
        explore_tables(conn)

        revenue_by_category(conn)
        check_dead_stock(conn)
        monthly_revenue_trend(conn)
        order_status_distribution(conn)


if __name__ == "__main__":
    main()